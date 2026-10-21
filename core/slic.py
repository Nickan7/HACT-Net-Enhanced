import os
import argparse
import numpy as np
import torch
import cv2  
from PIL import Image

from dgl.data.utils import save_graphs
from histocartography.preprocessing import SLICSuperpixelExtractor, RAGGraphBuilder
from histocartography.visualization import OverlayGraphVisualization

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def generate_slic_tissue_graph(image_path, output_dir, nr_superpixels=300, compactness=10):
    # --- Load image ---
    from openslide import OpenSlide

    if image_path.lower().endswith('.svs'):
        slide = OpenSlide(image_path)
        level = 0
        patch_size = 4096

        width, height = slide.level_dimensions[level]
        if width < patch_size or height < patch_size:
            raise ValueError(f"SVS slide {image_path} is too small to extract patch.")

        x = (width // 2) - (patch_size // 2)
        y = (height // 2) - (patch_size // 2)

        image = np.array(slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB"))
    else:
        image = np.array(Image.open(image_path).convert("RGB"))

    # --- Extract superpixels ---
    slic_extractor = SLICSuperpixelExtractor(
        nr_superpixels=nr_superpixels,
        compactness=compactness
    )
    superpixel_mask = slic_extractor.process(image)

    # --- Extract features (mean RGB per superpixel) ---
    features = []
    for rid in np.unique(superpixel_mask):
        mask = (superpixel_mask == rid)
        mean_rgb = image[mask].mean(axis=0)
        features.append(mean_rgb)
    features = torch.tensor(features, dtype=torch.float32)

    # --- Build tissue graph ---
    rag_builder = RAGGraphBuilder()
    tissue_graph = rag_builder.process(superpixel_mask, features)

    # --- Output names ---
    os.makedirs(os.path.join(output_dir, "slic"), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]



    # --- Save visualization ---
    visualizer = OverlayGraphVisualization()


    # Create a random colormap for unique superpixel IDs
    unique_ids = np.unique(superpixel_mask)
    rand_cmap = plt.get_cmap('tab20', len(unique_ids))
    colored_mask = rand_cmap((superpixel_mask - superpixel_mask.min()) % len(unique_ids))

    # Convert to 8-bit image
    colored_mask_img = (colored_mask[:, :, :3] * 255).astype(np.uint8)

    blurred_img = cv2.GaussianBlur(colored_mask_img, (11, 11), sigmaX=5)
    Image.fromarray(blurred_img).save(os.path.join(output_dir, "slic", base_name + "_slic.png"))


    # Image.fromarray(colored_mask_img).save(os.path.join(output_dir, "slic", base_name + "_slic.png"))


    print(f"[✓] Saved visualization to {base_name}_slic.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image (.png)")
    parser.add_argument("--output", type=str, required=True, help="Folder to save graphs and images")
    parser.add_argument("--nr_superpixels", type=int, default=300, help="Number of SLIC superpixels")
    parser.add_argument("--compactness", type=float, default=10.0, help="SLIC compactness parameter")
    args = parser.parse_args()

    generate_slic_tissue_graph(
        image_path=args.input,
        output_dir=args.output,
        nr_superpixels=args.nr_superpixels,
        compactness=args.compactness
    )
