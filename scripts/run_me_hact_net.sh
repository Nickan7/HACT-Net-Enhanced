### Server ###
# nvidia-smi 
# export OMP_NUM_THREADS=4



### pt files for nuclei_extraction.py
# scp /mnt/d/cbml/dissertation/hact-net-main/models/hovernet_monusac.pt kanakis@139.91.190.244:/home/kanakis/miniconda3/envs/nikoskan_1998/lib/python3.7/site-packages/histocartography/preprocessing/

### pt files for train.py
# scp /mnt/d/cbml/dissertation/hact-net-main/models/hovernet_monusac.pt kanakis@139.91.190.244:/media/hdd/kanakis/hactnet/core


# server password: k4n4k1s
ssh kanakis@139.91.190.244
conda activate nikoskan_1998 && cd hactnet/core && export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6


# Create .yml of enviroment
conda env export > environment_hactnet.yml 

# Download enviroment
scp kanakis@139.91.190.244:/media/hdd/kanakis/hactnet/core/environment_hactnet.yml nikoskan@DESKTOP-9DLUH9B:/mnt/d/cbml/dissertation/hact-net-main/fixed_files/


nvidia-smi
screen -ls

ctrl A, D # sceen log out
ctrl A, [ # scroll up and down with arrays, then q for quit



### If it gets stuck or has a lot of bugs due to the GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 && export CUDA_LAUNCH_BLOCKING=1 && python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()"





#--------------------------Seperate data to folders train, test and val-------------------------
python3 seperate_data_to_folders.py --data_folder ../data/... --percentance 100







56


screen -r cuda_0


#-------------------------Step 1 - HieArchical Cell to Tissue Generation - BRACS ROI Dataset-----------------------


# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/val/0_N_val/ --save_path ../roi_bracs_trained_0/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/test/0_N_test/ --save_path ../roi_bracs_trained_0/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/train/0_N_train/ --save_path ../roi_bracs_trained_0/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/val/1_PB_val/ --save_path ../roi_bracs_trained_1/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/test/1_PB_test/ --save_path ../roi_bracs_trained_1/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/train/1_PB_train/ --save_path ../roi_bracs_trained_1/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/val/2_UDH_val/ --save_path ../roi_bracs_trained_2/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/test/2_UDH_test/ --save_path ../roi_bracs_trained_2/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/train/2_UDH_train/ --save_path ../roi_bracs_trained_2/ &&

# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/val/3_FEA_val/ --save_path ../roi_bracs_trained_3/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/test/3_FEA_test/ --save_path ../roi_bracs_trained_3/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/train/3_FEA_train/ --save_path ../roi_bracs_trained_3/ 
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/val/4_ADH_val/ --save_path ../roi_bracs_trained_4/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/test/4_ADH_test/ --save_path ../roi_bracs_trained_4/ &&
# CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/train/4_ADH_train/ --save_path ../roi_bracs_trained_4/ 

# CUDA_VISIBLE_DEVICES=2 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/val/5_DCIS_val/ --save_path ../roi_bracs_trained_5/ &&
# CUDA_VISIBLE_DEVICES=2 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/test/5_DCIS_test/ --save_path ../roi_bracs_trained_5/ &&
# CUDA_VISIBLE_DEVICES=2 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/train/5_DCIS_train/ --save_path ../roi_bracs_trained_5/ &&
# CUDA_VISIBLE_DEVICES=2 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/val/6_IC_val/ --save_path ../roi_bracs_trained_6/ &&
# CUDA_VISIBLE_DEVICES=2 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/test/6_IC_test/ --save_path ../roi_bracs_trained_6/ &&
# CUDA_VISIBLE_DEVICES=2 python3 generate_hact_graphs.py --image_path ../data/bracs_roi_data/train/6_IC_train/ --save_path ../roi_bracs_trained_6/





#-------------------------Step 2 - Find the NaN .bin files-----------------------------------------------------------------

# python3 find_corrupt_bins.py --path ../bracs_output/cell_graphs/train/
# python3 find_corrupt_bins.py --path ../bracs_output/cell_graphs/test/
# python3 find_corrupt_bins.py --path ../bracs_output/cell_graphs/val/

# python3 find_corrupt_bins.py --path ../bracs_output/tissue_graphs/train/
# python3 find_corrupt_bins.py --path ../bracs_output/tissue_graphs/test/
# python3 find_corrupt_bins.py --path ../bracs_output/tissue_graphs/val/

# python3 find_corrupt_h5.py --path ../bracs_output/assignment_matrices/train/
# python3 find_corrupt_h5.py --path ../bracs_output/assignment_matrices/test/
# python3 find_corrupt_h5.py --path ../bracs_output/assignment_matrices/val/




#--------------------------Step 3 - Delete the NaN .bin and .h5 files---------------------------------------------------------

# for f in $(python3 find_corrupt_bins.py --path ../bracs_output/cell_graphs/train/ | grep "Corrupt file" | awk '{print $NF}'); do rm "$f"; done
# for f in $(python3 find_corrupt_bins.py --path ../bracs_output/tissue_graphs/train/ | grep "Corrupt file" | awk '{print $NF}'); do rm "$f"; done
# for f in $(python3 find_corrupt_h5.py --path ../bracs_output/assignment_matrices/train/ | grep "Corrupt file" | awk '{print $NF}'); do rm "$f"; done






#-------------------------Step 4 - Train Model - BRACS ROI Dataset-------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_0/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_0/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_0/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_0/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --tg_path ../roi_bracs_trained_0/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_0/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_0/cell_graphs/ --tg_path ../roi_bracs_trained_0/tissue_graphs/ --assign_mat_path ../roi_bracs_trained_0/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_0/ 
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_1/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_1/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_1/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_1/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --tg_path ../roi_bracs_trained_1/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_1/
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_1/cell_graphs/ --tg_path ../roi_bracs_trained_1/tissue_graphs/ --assign_mat_path ../roi_bracs_trained_1/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_1/
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_2/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_2/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_2/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_2/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --tg_path ../roi_bracs_trained_2/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_2/
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_2/cell_graphs/ --tg_path ../roi_bracs_trained_2/tissue_graphs/ --assign_mat_path ../roi_bracs_trained_2/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_2/
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_3/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_3/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_3/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_3/ &&
# CUDA_VISIBLE_DEVICES=1 python3 train.py --tg_path ../roi_bracs_trained_3/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_3/
# CUDA_VISIBLE_DEVICES=1 python3 train.py --cg_path ../roi_bracs_trained_3/cell_graphs/ --tg_path ../roi_bracs_trained_3/tissue_graphs/ --assign_mat_path ../roi_bracs_trained_3/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_3/
# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_4/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_4/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_4/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_4/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --tg_path ../roi_bracs_trained_4/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_4/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_4/cell_graphs/ --tg_path ../roi_bracs_trained_4/tissue_graphs/ --assign_mat_path ../roi_bracs_trained_4/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_4/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_5/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_5/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_5/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_5/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --tg_path ../roi_bracs_trained_5/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_5/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_5/cell_graphs/ --tg_path ../roi_bracs_trained_5/tissue_graphs/ --assign_mat_path ../roi_bracs_trained_5/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_5/ &&

# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_6/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_6/ &&
CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_6/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_6/
# CUDA_VISIBLE_DEVICES=2 python3 train.py --tg_path ../roi_bracs_trained_6/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_6/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../roi_bracs_trained_6/cell_graphs/ --tg_path ../roi_bracs_trained_6/tissue_graphs/ --assign_mat_path ../roi_bracs_trained_6/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../roi_bracs_trained_6/ 

# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../bracs_trained_all/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../bracs_trained_all/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --cg_path ../bracs_trained_all/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml -b 8 --epochs 60 -l 0.0005 --out_path ../bracs_trained_all/ &&
# CUDA_VISIBLE_DEVICES=2 python3 train.py --tg_path ../bracs_trained_all/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../bracs_trained_all/
CUDA_VISIBLE_DEVICES=0 python3 train.py --cg_path ../bracs_trained_all/cell_graphs/ --tg_path ../bracs_trained_all/tissue_graphs/ --assign_mat_path ../bracs_trained_all/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml -b 8 --epochs 60 -l 0.0005 --out_path ../bracs_trained_all/ 




















#------------------------------Step 5 - Generate Hact Graphs - BRACS WSI Dataset----------------------------------------------------
# add the type of tumor like generate_hact.py first lines
# for f in BRACS_*.svs; do mv "$f" "${f%.svs}_N_0.svs 

CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_wsi_data/0_N/ --save_path ../wsi_bracs_trained_0/ &&
CUDA_VISIBLE_DEVICES=1 python3 generate_hact_graphs.py --image_path ../data/bracs_wsi_data/1_PB/ --save_path ../wsi_bracs_trained_1/ 

# CUDA_VISIBLE_DEVICES=2 python3 generate_hact_graphs.py --image_path ../data/bracs_wsi_data/2_UDH/ --save_path ../wsi_bracs_trained_2/ &&
# CUDA_VISIBLE_DEVICES=2 python3 generate_hact_graphs.py --image_path ../data/bracs_wsi_data/3_FEA/ --save_path ../wsi_bracs_trained_3/ 

CUDA_VISIBLE_DEVICES=3 python3 generate_hact_graphs.py --image_path ../data/bracs_wsi_data/4_ADH/ --save_path ../wsi_bracs_trained_4/ &&
CUDA_VISIBLE_DEVICES=3 python3 generate_hact_graphs.py --image_path ../data/bracs_wsi_data/5_DCIS/ --save_path ../wsi_bracs_trained_5/ &&
CUDA_VISIBLE_DEVICES=3 python3 generate_hact_graphs.py --image_path ../data/bracs_wsi_data/6_IC/ --save_path ../wsi_bracs_trained_6/


#------------------------------Step 6 - Nuclei Detection Graphs------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python3 nuclei_detection.py --input ../data/bracs_wsi_data/6_IC/BRACS_1411_IC_6.svs --output ../wsi_bracs_trained_6/




#------------------------------Step 6 - SLIC Tissue Graphs------------------------------------------------------------
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
CUDA_VISIBLE_DEVICES=0 python3 slic.py --input ../data/bracs_wsi_data/6_IC/BRACS_1411_IC_6.svs --output ../wsi_bracs_trained_6/ --nr_superpixels 300 --compactness 10


#------------------------------Step 7 - Visualize Assignment Matrix------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python3 assign_mat.py --input ../wsi_bracs_trained_6/assignment_matrices/test/BRACS_1411_IC_6.h5  --output ../wsi_bracs_trained_6/





#-----------------------------Step 8 - Inference on HACTNet------------------------------------------------

# From Generate use cg.bin, tg.bin, ass_mat.h5 and from Train use models
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_0/ --cg_path ../wsi_bracs_trained_0/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --model_path ../roi_bracs_trained_0/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_0/ --cg_path ../wsi_bracs_trained_0/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_0/cell_model_pna/model_best_val_weighted_f1_score.pt && 
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_0/ --tg_path ../wsi_bracs_trained_0/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_0/tissue_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_0/ --cg_path ../wsi_bracs_trained_0/cell_graphs/ --tg_path ../wsi_bracs_trained_0/tissue_graphs/ --assign_mat_path ../wsi_bracs_trained_0/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml --model_path ../roi_bracs_trained_0/hactnet_model_pna/model_best_val_weighted_f1_score.pt &&

# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_1/ --cg_path ../wsi_bracs_trained_1/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --model_path ../roi_bracs_trained_1/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_1/ --cg_path ../wsi_bracs_trained_1/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_1/cell_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_1/ --tg_path ../wsi_bracs_trained_1/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_1/tissue_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_1/ --cg_path ../wsi_bracs_trained_1/cell_graphs/ --tg_path ../wsi_bracs_trained_1/tissue_graphs/ --assign_mat_path ../wsi_bracs_trained_1/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml --model_path ../roi_bracs_trained_1/hactnet_model_pna/model_best_val_weighted_f1_score.pt &&

# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_2/ --cg_path ../wsi_bracs_trained_2/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --model_path ../roi_bracs_trained_2/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_2/ --cg_path ../wsi_bracs_trained_2/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_2/cell_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_2/ --tg_path ../wsi_bracs_trained_2/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_2/tissue_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_2/ --cg_path ../wsi_bracs_trained_2/cell_graphs/ --tg_path ../wsi_bracs_trained_2/tissue_graphs/ --assign_mat_path ../wsi_bracs_trained_2/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml --model_path ../roi_bracs_trained_2/hactnet_model_pna/model_best_val_weighted_f1_score.pt

# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_3/ --cg_path ../wsi_bracs_trained_3/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --model_path ../roi_bracs_trained_3/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_3/ --cg_path ../wsi_bracs_trained_3/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_3/cell_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_3/ --tg_path ../wsi_bracs_trained_3/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_3/tissue_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_3/ --cg_path ../wsi_bracs_trained_3/cell_graphs/ --tg_path ../wsi_bracs_trained_3/tissue_graphs/ --assign_mat_path ../wsi_bracs_trained_3/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml --model_path ../roi_bracs_trained_3/hactnet_model_pna/model_best_val_weighted_f1_score.pt &&

# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_4/ --cg_path ../wsi_bracs_trained_4/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --model_path ../roi_bracs_trained_4/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_4/ --cg_path ../wsi_bracs_trained_4/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_4/cell_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_4/ --tg_path ../wsi_bracs_trained_4/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_4/tissue_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_4/ --cg_path ../wsi_bracs_trained_4/cell_graphs/ --tg_path ../wsi_bracs_trained_4/tissue_graphs/ --assign_mat_path ../wsi_bracs_trained_4/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml --model_path ../roi_bracs_trained_4/hactnet_model_pna/model_best_val_weighted_f1_score.pt &&

# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_5/ --cg_path ../wsi_bracs_trained_5/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --model_path ../roi_bracs_trained_5/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_5/ --cg_path ../wsi_bracs_trained_5/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_5/cell_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_5/ --tg_path ../wsi_bracs_trained_5/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_5/tissue_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_5/ --cg_path ../wsi_bracs_trained_5/cell_graphs/ --tg_path ../wsi_bracs_trained_5/tissue_graphs/ --assign_mat_path ../wsi_bracs_trained_5/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml --model_path ../roi_bracs_trained_5/hactnet_model_pna/model_best_val_weighted_f1_score.pt &&

CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_6/ --cg_path ../wsi_bracs_trained_6/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --model_path ../roi_bracs_trained_6/cell_model_gin/model_best_val_weighted_f1_score.pt 
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_6/ --cg_path ../wsi_bracs_trained_6/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_6/cell_model_pna/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_6/ --tg_path ../wsi_bracs_trained_6/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml --model_path ../roi_bracs_trained_6/tissue_model_pna/model_best_val_weighted_f1_score.pt && 
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../wsi_bracs_trained_6/ --cg_path ../wsi_bracs_trained_6/cell_graphs/ --tg_path ../wsi_bracs_trained_6/tissue_graphs/ --assign_mat_path ../wsi_bracs_trained_6/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml --model_path ../roi_bracs_trained_6/hactnet_model_pna/model_best_val_weighted_f1_score.pt

# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../bracs_trained_all/ --cg_path ../bracs_trained_all/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --model_path ../bracs_trained_all/cell_model_gin/model_best_val_weighted_f1_score.pt
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../bracs_trained_all/ --cg_path ../bracs_trained_all/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_pna.yml --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt
# CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../bracs_trained_all/ --tg_path ../bracs_trained_all/tissue_graphs/ --config_fpath ./config/bracs_tggnn_7_classes_pna.yml --model_path ../bracs_trained_all/tissue_model_pna/model_best_val_weighted_f1_score.pt
CUDA_VISIBLE_DEVICES=1 python3 inference.py --save_path ../bracs_trained_all/ --cg_path ../bracs_trained_all/cell_graphs/ --tg_path ../bracs_trained_all/tissue_graphs/ --assign_mat_path ../bracs_trained_all/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml --model_path ../bracs_trained_all/hactnet_model_pna/model_best_val_weighted_f1_score.pt






#-----------------------------Step 9 - Multi Inference the .pt Models----------------------------------------------------

CUDA_VISIBLE_DEVICES=1 python3 multi_inference_driver.py --save_path ../bracs_trained_all/ --model_path ../bracs_trained_all/hactnet_model_pna/ --cg_path ../bracs_trained_all/cell_graphs/ --tg_path ../bracs_trained_all/tissue_graphs/ --assign_mat_path ../bracs_trained_all/assignment_matrices/ --config_fpath ./config/bracs_hact_7_classes_pna.yml

CUDA_VISIBLE_DEVICES=1 python3 multi_inference_driver.py --save_path ../wsi_bracs_trained_6/ --model_path ../bracs_trained_all/cell_model_gin/ --cg_path ../wsi_bracs_trained_6/cell_graphs/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml  






#-----------------------------Step 10 - Predict Tumor Type----------------------------------------------------

# CUDA_VISIBLE_DEVICES=2 python3 predict_tumor_type.py --image_path ../data/bracs_wsi_data/0_N/ --save_path ../wsi_bracs_trained_0/ --cg_path ../wsi_bracs_trained_0/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=1 python3 predict_tumor_type.py --image_path ../data/bracs_wsi_data/1_PB/ --save_path ../wsi_bracs_trained_1/ --cg_path ../wsi_bracs_trained_1/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=2 python3 predict_tumor_type.py --image_path ../data/bracs_wsi_data/2_UDH/ --save_path ../wsi_bracs_trained_2/ --cg_path ../wsi_bracs_trained_2/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=2 python3 predict_tumor_type.py --image_path ../data/bracs_wsi_data/3_FEA/ --save_path ../wsi_bracs_trained_3/ --cg_path ../wsi_bracs_trained_3/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=2 python3 predict_tumor_type.py --image_path ../data/bracs_wsi_data/4_ADH/ --save_path ../wsi_bracs_trained_4/ --cg_path ../wsi_bracs_trained_4/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=2 python3 predict_tumor_type.py --image_path ../data/bracs_wsi_data/5_DCIS/ --save_path ../wsi_bracs_trained_5/ --cg_path ../wsi_bracs_trained_5/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=2 python3 predict_tumor_type.py --image_path ../data/bracs_wsi_data/6_IC/ --save_path ../wsi_bracs_trained_6/ --cg_path ../wsi_bracs_trained_6/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml


# CUDA_VISIBLE_DEVICES=2 python3 predict_tumor_type.py --exact_image ../data/bracs_wsi_data/0_N/BRACS_1579_N_0.svs --save_path ../wsi_bracs_trained_0/ --cg_path ../wsi_bracs_trained_0/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=1 python3 predict_tumor_type.py --exact_image ../data/bracs_wsi_data/1_PB/BRACS_1412_PB_1.svs --save_path ../wsi_bracs_trained_1/ --cg_path ../wsi_bracs_trained_1/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=1 python3 predict_tumor_type.py --exact_image ../data/bracs_wsi_data/2_UDH/BRACS_1334_UDH_2.svs --save_path ../wsi_bracs_trained_2/ --cg_path ../wsi_bracs_trained_2/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=1 python3 predict_tumor_type.py --exact_image ../data/bracs_wsi_data/3_FEA/BRACS_1813_FEA_3.svs --save_path ../wsi_bracs_trained_3/ --cg_path ../wsi_bracs_trained_3/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=1 python3 predict_tumor_type.py --exact_image ../data/bracs_wsi_data/4_ADH/BRACS_1003691_ADH_4.svs --save_path ../wsi_bracs_trained_4/ --cg_path ../wsi_bracs_trained_4/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=1 python3 predict_tumor_type.py --exact_image ../data/bracs_wsi_data/5_DCIS/BRACS_1283_DCIS_5.svs --save_path ../wsi_bracs_trained_5/ --cg_path ../wsi_bracs_trained_5/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml
# CUDA_VISIBLE_DEVICES=1 python3 predict_tumor_type.py --exact_image ../data/bracs_wsi_data/6_IC/BRACS_1411_IC_6.svs --save_path ../wsi_bracs_trained_6/ --cg_path ../wsi_bracs_trained_6/cell_graphs/test/ --model_path ../bracs_trained_all/cell_model_pna/model_best_val_weighted_f1_score.pt --config_fpath ./config/bracs_cggnn_7_classes_pna.yml



#------------------------------Step 11 - Explain Graphs - BRACS WSI Dataset----------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python3 explain_graphs.py --exact_image ../data/bracs_wsi_data/0_N/BRACS_1579_N_0.svs --save_path ../wsi_bracs_trained_0/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_0/cell_graphs/test/ --tg_path ../wsi_bracs_trained_0/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_0/assignment_matrices/test/ --model_name ../roi_bracs_trained_0/models/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=0 python3 explain_graphs.py --exact_image ../data/bracs_wsi_data/1_PB/BRACS_1412_PB_1.svs --save_path ../wsi_bracs_trained_1/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_1/cell_graphs/test/ --tg_path ../wsi_bracs_trained_1/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_1/assignment_matrices/test/ --model_name ../roi_bracs_trained_1/models/cell_model_gin/model_best_val_weighted_f1_score.pt

# CUDA_VISIBLE_DEVICES=1 python3 explain_graphs.py --exact_image ../data/bracs_wsi_data/2_UDH/BRACS_1334_UDH_2.svs --save_path ../wsi_bracs_trained_2/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_2/cell_graphs/test/ --tg_path ../wsi_bracs_trained_2/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_2/assignment_matrices/test/ --model_name ../roi_bracs_trained_2/models/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 explain_graphs.py --exact_image ../data/bracs_wsi_data/3_FEA/BRACS_1813_FEA_3.svs --save_path ../wsi_bracs_trained_3/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_3/cell_graphs/test/ --tg_path ../wsi_bracs_trained_3/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_3/assignment_matrices/test/ --model_name ../roi_bracs_trained_3/models/cell_model_gin/model_best_val_weighted_f1_score.pt 

# CUDA_VISIBLE_DEVICES=2 python3 explain_graphs.py --exact_image ../data/bracs_wsi_data/4_ADH/BRACS_1003691_ADH_4.svs --save_path ../wsi_bracs_trained_4/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_4/cell_graphs/test/ --tg_path ../wsi_bracs_trained_4/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_4/assignment_matrices/test/ --model_name ../roi_bracs_trained_4/models/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=2 python3 explain_graphs.py --exact_image ../data/bracs_wsi_data/5_DCIS/BRACS_1283_DCIS_5.svs --save_path ../wsi_bracs_trained_5/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_5/cell_graphs/test/ --tg_path ../wsi_bracs_trained_5/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_5/assignment_matrices/test/ --model_name ../roi_bracs_trained_5/models/cell_model_gin/model_best_val_weighted_f1_score.pt

# CUDA_VISIBLE_DEVICES=3 python3 explain_graphs.py --exact_image ../data/bracs_wsi_data/6_IC/BRACS_1284_IC_6.svs --save_path ../wsi_bracs_trained_6/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_6/cell_graphs/test/ --tg_path ../wsi_bracs_trained_6/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_6/assignment_matrices/test/ --model_name ../roi_bracs_trained_6/models/cell_model_gin/model_best_val_weighted_f1_score.pt





# CUDA_VISIBLE_DEVICES=0 python3 explain_graphs.py --image_path ../data/bracs_wsi_data/0_N/ --save_path ../wsi_bracs_trained_0/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_0/cell_graphs/test/ --tg_path ../wsi_bracs_trained_0/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_0/assignment_matrices/test/ --model_name ../roi_bracs_trained_0/models/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=0 python3 explain_graphs.py --image_path ../data/bracs_wsi_data/1_PB/ --save_path ../wsi_bracs_trained_1/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_1/cell_graphs/test/ --tg_path ../wsi_bracs_trained_1/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_1/assignment_matrices/test/ --model_name ../roi_bracs_trained_1/models/cell_model_gin/model_best_val_weighted_f1_score.pt

# CUDA_VISIBLE_DEVICES=1 python3 explain_graphs.py --image_path ../data/bracs_wsi_data/2_UDH/ --save_path ../wsi_bracs_trained_2/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_2/cell_graphs/test/ --tg_path ../wsi_bracs_trained_2/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_2/assignment_matrices/test/ --model_name ../roi_bracs_trained_2/models/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=1 python3 explain_graphs.py --image_path ../data/bracs_wsi_data/3_FEA/ --save_path ../wsi_bracs_trained_3/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_3/cell_graphs/test/ --tg_path ../wsi_bracs_trained_3/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_3/assignment_matrices/test/ --model_name ../roi_bracs_trained_3/models/cell_model_gin/model_best_val_weighted_f1_score.pt 

# CUDA_VISIBLE_DEVICES=2 python3 explain_graphs.py --image_path ../data/bracs_wsi_data/4_ADH/ --save_path ../wsi_bracs_trained_4/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_4/cell_graphs/test/ --tg_path ../wsi_bracs_trained_4/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_4/assignment_matrices/test/ --model_name ../roi_bracs_trained_4/models/cell_model_gin/model_best_val_weighted_f1_score.pt &&
# CUDA_VISIBLE_DEVICES=2 python3 explain_graphs.py --image_path ../data/bracs_wsi_data/5_DCIS/ --save_path ../wsi_bracs_trained_5/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_5/cell_graphs/test/ --tg_path ../wsi_bracs_trained_5/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_5/assignment_matrices/test/ --model_name ../roi_bracs_trained_5/models/cell_model_gin/model_best_val_weighted_f1_score.pt

CUDA_VISIBLE_DEVICES=2 python3 explain_graphs.py --image_path ../data/bracs_wsi_data/6_IC/ --save_path ../wsi_bracs_trained_6/ --config_fpath ./config/bracs_cggnn_7_classes_gin.yml --cg_path ../wsi_bracs_trained_6/cell_graphs/test/ --tg_path ../wsi_bracs_trained_6/tissue_graphs/test/ --as_mat ../wsi_bracs_trained_6/assignment_matrices/test/ --model_name ../roi_bracs_trained_6/models/cell_model_gin/model_best_val_weighted_f1_score.pt
