#!/usr/bin/env python3

import torch
import mlflow
import os
import torch.nn.functional as F
import pickle
import uuid
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import mlflow.pytorch
import numpy as np
from collections import Counter
import pandas as pd
import shutil
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow.pytorch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel
import inference
from dataloader import make_data_loader

# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 514

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cg_path', type=str, default=None)
    parser.add_argument('--tg_path', type=str, default=None)
    parser.add_argument('--assign_mat_path', type=str, default=None)
    parser.add_argument('-conf', '--config_fpath', type=str, default='')
    parser.add_argument('--in_ram', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='../../data/graphs')
    parser.add_argument('--model_path', type=str, default='../../data/graphs')
    return parser.parse_args()

def main():
    args = parse_arguments()

    if 'bracs_cggnn' in args.config_fpath:
        subfolder = 'cell_model'
    elif 'bracs_tggnn' in args.config_fpath:
        subfolder = 'tissue_model'
    elif 'bracs_hact' in args.config_fpath:
        subfolder = 'hactnet_model'
    else:
        raise ValueError('Model type not recognized. Options are: TG, CG or HACT.')

    if 'pna' in args.config_fpath:
        model_gnn = 'pna'
    elif 'gin' in args.config_fpath:
        model_gnn = 'gin'
    else:
        raise ValueError('Model type not recognized. Options are: PNA or GIN.')



    model_dir = os.path.join(args.model_path)

    model_files = [
        "model_best_val_loss.pt",
        "model_best_val_accuracy.pt",
        "model_best_val_weighted_f1_score.pt"
    ]

    model_score = 'multi_eval'

    all_acc = []
    all_f1 = []
    all_loss = []

    for model_file in model_files:
        print(f"\n--- Evaluating {model_file} ---")
        args.model_path = os.path.join(model_dir, model_file)
        acc, f1, loss = inference.main(args, return_metrics=True)
        all_acc.append(acc)
        all_f1.append(f1)
        all_loss.append(loss)

    df_metrics = pd.DataFrame({
        'Model': list(range(1, len(all_acc) + 1)) * 3,
        'Metric': ['F1 Score'] * len(all_f1) + ['Accuracy'] * len(all_acc)  + ['Loss'] * len(all_loss),
        'Value': all_f1 + all_acc +  all_loss
    })

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Metric', y='Value', data=df_metrics, palette='Set1')
    plt.title("Performance Variability Across Models")
    plt.tight_layout()

    result_dir = os.path.join(args.save_path, "inference", f"{subfolder}_{model_gnn}_{model_score}")
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, f'model_metrics_boxplot_{model_gnn}.png'))
    plt.close()


if __name__ == "__main__":
    main()
