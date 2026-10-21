"""
Extract HACT graphs for all the sample in the BRACS dataset.
"""

import os
from glob import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch 
from dgl.data.utils import save_graphs
import h5py
import math
import openslide

from histocartography.preprocessing import (
    VahadaneStainNormalizer,         # stain normalizer
    NucleiExtractor,                 # nuclei detector 
    DeepFeatureExtractor,            # feature extractor 
    KNNGraphBuilder,                 # kNN graph builder
    ColorMergedSuperpixelExtractor,  # tissue detector
    DeepFeatureExtractor,            # feature extractor
    RAGGraphBuilder,                 # build graph
    AssignmnentMatrixBuilder         # assignment matrix 
)


# BRACS subtype to 7-class label 
TUMOR_TYPE_TO_LABEL = {
    'N': 0,
    'PB': 1,
    'UDH': 2,
    'ADH': 3,
    'FEA': 4,
    'DCIS': 5,
    'IC': 6
}

MIN_NR_PIXELS = 50000
MAX_NR_PIXELS = 50000000  
STAIN_NORM_TARGET_IMAGE = '../target_image/target.png'  # define stain normalization target image. 


patch_size = {
    'cell_input': 72 , 
    'cell_output': 224 , 
    'tissue_input': 144 , 
    'tissue_output': 224 
}



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        help='path to the BRACS data.',
        default='',
        required=False
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='path to save the cell graphs.',
        default='../data/',
        required=False
    )
    parser.add_argument(
        '--exact_image',
        type=str,
        help='(Optional) Path to a single exact image to process (overrides --image_path).',
        default=None,
        required=False
    )
    return parser.parse_args()

class HACTBuilding:

    def __init__(self):

        # 1. define stain normalizer 
        self.normalizer = VahadaneStainNormalizer(target_path=STAIN_NORM_TARGET_IMAGE)

        # 2. define CG builders
        self._build_cg_builders()

        # 3. define TG builders 
        self._build_tg_builders()

        # 4. define assignment matrix builder
        self.assignment_matrix_builder = AssignmnentMatrixBuilder()

        # 5. define var to store image IDs that failed (for whatever reason)
        self.image_ids_failing = []

    def _build_cg_builders(self):
        # a define nuclei extractor
        self.nuclei_detector = NucleiExtractor()

        # b define feature extractor: Extract patches of 72x72 pixels around each
        # nucleus centroid, then resize to 224 to match ResNet input size.
        self.nuclei_feature_extractor = DeepFeatureExtractor(
            architecture='resnet34',
            patch_size=patch_size['cell_input'],
            resize_size=patch_size['cell_output']
        )

        # c define k-NN graph builder with k=5 and thresholding edges longer
        # than 50 pixels. Add image size-normalized centroids to the node features.
        # For e.g., resulting node features are 512 features from ResNet34 + 2
        # normalized centroid features.
        self.knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)

    def _build_tg_builders(self):
        # a define nuclei extractor    
        self.tissue_detector = ColorMergedSuperpixelExtractor(
            superpixel_size=500,
            compactness=20,
            blur_kernel_size=1,
            threshold=0.05,
            downsampling_factor=4
        )

        # b define feature extractor: Extract patches of 144x144 pixels all over 
        # the tissue regions. Each patch is resized to 224 to match ResNet input size.
        self.tissue_feature_extractor = DeepFeatureExtractor(
            architecture='resnet34',
            patch_size=patch_size['tissue_input'],
            resize_size=patch_size['tissue_output']
        )

        # c define RAG builder. Append normalized centroid to the node features. 
        self.rag_graph_builder = RAGGraphBuilder(add_loc_feats=True)

    def _build_cg(self, image):
        nuclei_map, nuclei_centroids = self.nuclei_detector.process(image)
        features = self.nuclei_feature_extractor.process(image, nuclei_map)
        graph = self.knn_graph_builder.process(nuclei_map, features)
        return graph, nuclei_centroids

    def _build_tg(self, image):
        superpixels, _ = self.tissue_detector.process(image)
        features = self.tissue_feature_extractor.process(image, superpixels)
        graph = self.rag_graph_builder.process(superpixels, features)
        return graph, superpixels

    def process(self, image_path, save_path, split):
        # 1. get image path

        nan_cell = []
        nan_tissue = []

        image_fnames = []

        if os.path.isfile(image_path):
            image_fnames.append(image_path)
        else:
            subdirs = os.listdir(image_path)
            for subdir in (subdirs + ['']):  # look for all subdirs AND the base folder
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.svs'):
                    image_fnames += glob(os.path.join(image_path, subdir, ext))


        print('\n\n\n\n\n*** Start analysing {} images ***\n'.format(len(image_fnames)))

        for image_path in tqdm(image_fnames):

            # a. load image & check if already there 
            _, image_name = os.path.split(image_path)
            
            if image_path.lower().endswith('.svs'):
                try:
                    slide = openslide.OpenSlide(image_path)

                    # Parameters for svs processing
                    patch_size = 4096  # size of patch you want
                    level = 0  # openslide level (0 = highest resolution)
                    num_patches = 1  # number of patches to sample, you can set >1 if you want multiple patches per slide

                    width, height = slide.level_dimensions[level]

                    if width < patch_size or height < patch_size:
                        print(f"Warning: {image_path} is too small for patch size {patch_size}. Skipping.")
                        continue

                    # Center patch (same as before)
                    x = (width // 2) - (patch_size // 2)
                    y = (height // 2) - (patch_size // 2)

                    image = np.array(slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB"))

                except Exception as e:
                    print(f"Warning: Failed to load {image_path} with error {e}")
                    self.image_ids_failing.append(image_path)
                    continue
            else:
                image = np.array(Image.open(image_path))




            nr_pixels = image.shape[0] * image.shape[1]
          
            image_label = -1  # Default: unknown label
            try:
                tumor_type = image_name.split('_')[2]
                image_label = TUMOR_TYPE_TO_LABEL[tumor_type]
            except (IndexError, KeyError):
                print(f"Warning: Couldn't extract tumor label from '{image_name}'. Using -1.")


            
            base_name = os.path.splitext(image_name)[0]
            cg_out = os.path.join(save_path, 'cell_graphs', split, base_name + '.bin')
            tg_out = os.path.join(save_path, 'tissue_graphs', split, base_name + '.bin')
            assign_out = os.path.join(save_path, 'assignment_matrices', split, base_name + '.h5')

            # if file was not already created + not too big + not too small, then process 
            if not self._exists(cg_out, tg_out, assign_out) and self._valid_image(nr_pixels):

                # b. stain norm the image 
                try: 
                    image = self.normalizer.process(image)
                except:
                    print('\nWarning: {} failed during stain normalization.\n'.format(image_path))
                    self.image_ids_failing.append(image_path)
                    pass
            

                try:
                    cell_graph, nuclei_centroid = self._build_cg(image)

                    feat = cell_graph.ndata['feat']
                    centroid = cell_graph.ndata['centroid']
                    cell_bool = 0

                    for i in range(len(feat)):
                        for j in range(len(feat[i])):
                            if torch.isnan(feat[i][j]):
                                nan_cell.append(image_name)
                                cell_bool = 1

                    for i in range(len(centroid)):
                        for j in range(len(centroid[i])):
                            if torch.isnan(centroid[i][j]):
                                nan_cell.append(image_name)
                                cell_bool = 1

                    nan_cell = list(set(nan_cell))

                except: pass
            

                try:
                    tissue_graph, tissue_map = self._build_tg(image)
                    
                    feat = tissue_graph.ndata['feat']
                    centroid = tissue_graph.ndata['centroid']
                    tissue_bool = 0 

                    for i in range(len(feat)):
                        for j in range(len(feat[i])):
                            if torch.isnan(feat[i][j]):
                                nan_tissue.append(image_name)
                                tissue_bool = 1

                    for i in range(len(centroid)):
                        for j in range(len(centroid[i])):
                            if torch.isnan(centroid[i][j]):
                                nan_tissue.append(image_name)
                                tissue_bool = 1

                    nan_tissue = list(set(nan_tissue))

                except: pass



                try:
                    if cell_bool == 0 and tissue_bool == 0 :

                        cell_graph, nuclei_centroid = self._build_cg(image)

                        if image_label is not None:
                            save_graphs(
                                filename=cg_out,
                                g_list=[cell_graph],
                                labels={"label": torch.tensor([image_label])}
                            )
                        else:
                            save_graphs(
                                filename=cg_out,
                                g_list=[cell_graph]
    )


                except:
                    print('\nWarning: {} failed during Cell Graph Generation.\n'.format(image_path))
                    self.image_ids_failing.append(image_path)
                    pass


                try: 
                    if cell_bool == 0 and tissue_bool == 0 :

                        tissue_graph, tissue_map = self._build_tg(image)
                        
                  
                        if image_label is not None:
                            save_graphs(
                                filename=tg_out,
                                g_list=[tissue_graph],
                                labels={"label": torch.tensor([image_label])}
                            )
                        else:
                            save_graphs(
                                filename=tg_out,
                                g_list=[tissue_graph]
                            )


                
                except:
                    print('\nWarning: {} failed during Tissue Graph Generation.\n'.format(image_path))
                    self.image_ids_failing.append(image_path)
                    pass


                try: 
                    if cell_bool == 0 and tissue_bool == 0 :
                        assignment_matrix = self.assignment_matrix_builder.process(nuclei_centroid, tissue_map)
                        with h5py.File(assign_out, "w") as output_file:
                            output_file.create_dataset(
                                "assignment_matrix",
                                data=assignment_matrix,
                                compression="gzip",
                                compression_opts=9,
                            )
                except:
                    print('\nWarning: {} failed during assignment matrix generation.\n'.format(image_path))
                    self.image_ids_failing.append(image_path)
                    pass

            else:
                print('\nImage:', image_path, ' was already processed or is too large/small.\n')

        print('\n\n\nOut of {} images, {} successful HACT graph generations.\n'.format(
            len(image_fnames),
            len(image_fnames) - len(self.image_ids_failing)
        ))
        print('\n\nFailing IDs are:', self.image_ids_failing)



        if not nan_cell:
            print('\n\nNo NaN Cell Graphs found\n') 

        else:
            print(f"\n\nCell Graphs {len(nan_cell)} / {len(image_fnames)} have NaN elements\n")
            print(f"\n\nThe following images {nan_cell}\n")


        if not nan_tissue:
            print('\n\nNo NaN Tissue Graphs found\n') 

        else:
            print(f"\n\nTissue Graphs {len(nan_tissue)} / {len(image_fnames)} have NaN elements\n")
            print(f"\n\nThe following images {nan_tissue}\n")




    def _valid_image(self, nr_pixels):
        if nr_pixels > MIN_NR_PIXELS and nr_pixels < MAX_NR_PIXELS:
            return True
        return False

    def _exists(self, cg_out, tg_out, assign_out):
        if os.path.isfile(cg_out) and os.path.isfile(tg_out) and os.path.isfile(assign_out):
            return True
        return False


if __name__ == "__main__":

    # 1. handle i/o
    args = parse_arguments()
    
    if args.exact_image:
        if not os.path.isfile(args.exact_image):
            raise ValueError(f"Exact image {args.exact_image} does not exist.")
        image_source = args.exact_image
    else:
        if not os.path.isdir(args.image_path) or not os.listdir(args.image_path):
            raise ValueError(f"Data directory {args.image_path} is either empty or does not exist.")
        image_source = args.image_path


    split = ''
    if 'train' in args.image_path:
        split = 'train'
    elif  'val' in args.image_path:
        split = 'val'
    else:
        split = 'test'

    os.makedirs(os.path.join(args.save_path, 'cell_graphs', split), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'tissue_graphs', split), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'assignment_matrices', split), exist_ok=True)

    # 2. generate HACT graphs one-by-one, will automatically
    # run on GPU if available.
    hact_builder = HACTBuilding()
    hact_builder.process(image_source, args.save_path, split)
