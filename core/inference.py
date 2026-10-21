#!/usr/bin/env python3

"""
Script for testing CG-GNN, TG-GNN and HACT models
"""
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

from dataloader import make_data_loader

# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
NODE_DIM = 514

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--tg_path',
        type=str,
        help='path to tissue graphs.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--assign_mat_path',
        type=str,
        help='path to the assignment matrices.',
        default=None,
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='',
        required=False
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to model to test.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '--pretrained',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=1,
        required=False
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='path to where the output data are saved (currently only for the interpretability).',
        default='../../data/graphs',
        required=False
    )

    return parser.parse_args()



def main(args, return_metrics=False):
    """
    Test HACTNet, CG-GNN or TG-GNN.
    Args:
        args (Namespace): parsed arguments.
    """
    # Ensure that only one of 'pretrained' or 'model_path' is specified, not both.
    assert not(args.pretrained and args.model_path is not None), "Provide a model path or set pretrained. Not both."
    assert (args.pretrained or args.model_path is not None), "Provide either a model path or set pretrained."

    # Load the configuration settings from the YAML file specified by 'config_fpath'.
    with open(args.config_fpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    # Create a data loader for the test dataset, loading from specified paths if provided.
    dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'test') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'test') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
        shuffle=False
    )





    # Initialize the model according to the configuration file type (CG-GNN, TG-GNN, or HACT).
    if 'bracs_cggnn' in args.config_fpath:
        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            num_classes=7,
            pretrained=args.pretrained
        ).to(DEVICE)

    elif 'bracs_tggnn' in args.config_fpath:
        model = TissueGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            num_classes=7,
            pretrained=args.pretrained
        ).to(DEVICE)

    elif 'bracs_hact' in args.config_fpath:
        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            classification_params=config['classification_params'],
            cg_node_dim=NODE_DIM,
            tg_node_dim=NODE_DIM,
            num_classes=7,
            pretrained=args.pretrained
        ).to(DEVICE)
    else:
        # Raise an error if an unrecognized model type is specified.
        raise ValueError('Model type not recognized. Options are: TG, CG or HACT.')



    # Create a unique directory for saving model checkpoints, ensuring it exists. 
    # Define subdirectory name based on config file name
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
    

    if 'f1' in args.model_path:
        model_score = 'f1'
    elif 'loss' in args.model_path:
        model_score = 'loss'
    elif 'accuracy' in args.model_path:
        model_score = 'accuracy'


    # Update the model path to include this subfolder
    model_path = os.path.join(args.save_path, "inference", f"{subfolder}_{model_gnn}_{model_score}")
    os.makedirs(model_path, exist_ok=True)



    # Load the saved model weights if a model path is provided and pretrained is not set. 
    if not args.pretrained:
        model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Begin testing: iterate through the test data loader and collect predictions and labels.
    all_test_logits = []
    all_test_labels = []
    all_model_accuracies = []
    all_model_f1_scores = []
    losses = []

   


    for batch in tqdm(dataloader, desc='Testing', unit='batch'):
        labels = batch[-1]
        data = batch[:-1]

        for i in range(len(data)):
            element = data[i]

            # If it's a DGL graph
            if hasattr(element, 'ndata'):
                if 'feat' in element.ndata:
                    element.ndata['feat'] = torch.nan_to_num(element.ndata['feat'], nan=0.0)
                if 'centroid' in element.ndata:
                    element.ndata['centroid'] = torch.nan_to_num(element.ndata['centroid'], nan=0.0)

            # If it's a tensor
            elif isinstance(element, torch.Tensor):
                data[i] = torch.nan_to_num(element, nan=0.0)

        # Forward pass through the model without gradients (faster and saves memory).
        with torch.no_grad():
            try:
                logits = model(*data)

                if logits is None:
                    print("Skipping batch due to assignment/feature shape mismatch.")
                    continue
            except RuntimeError as e:
                print(f"[ERROR] Runtime failure during validation/test forward pass: {e}")
                print("Skipping this batch.")
                continue

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        losses.append(loss.item())
        # Store the model's raw output logits and true labels for further evaluation.
        all_test_logits.append(logits)
        all_test_labels.append(labels)
        


    # Concatenate logits and labels across all batches.
    all_test_logits = torch.cat(all_test_logits).cpu()
    all_test_preds = torch.argmax(all_test_logits, dim=1)
    all_test_labels = torch.cat(all_test_labels).cpu()

    accuracy = accuracy_score(all_test_labels, all_test_preds)
    weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')

    all_model_accuracies.append(accuracy)
    all_model_f1_scores.append(weighted_f1_score)


    # Detach predictions and labels from GPU and convert to numpy arrays for metric computation.
    all_test_preds = all_test_preds.detach().numpy()
    all_test_labels = all_test_labels.detach().numpy()


    # Filter out dummy labels (-1) before computing metrics
    valid_indices = all_test_labels != -1
    all_test_preds = all_test_preds[valid_indices]
    all_test_labels = all_test_labels[valid_indices]




    if 'pna' in args.config_fpath and 'bracs_hact' in args.config_fpath:

        # Binarize the output labels for multi-class AUC
        n_classes = 7
        y_true = label_binarize(all_test_labels, classes=list(range(n_classes)))
        y_score = all_test_logits.numpy()  # these are logits, softmax is usually better

        # Optional: apply softmax
        
        y_score = F.softmax(torch.tensor(y_score), dim=1).numpy()

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure(figsize=(8, 6))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown'])
        label_names = ['N', 'PB', 'UDH', 'ADH', 'FEA', 'DCIS', 'IC']

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{label_names[i]} (AUC = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class ROC Curve (AUC)')
        plt.legend(loc="lower right")

        # Save to file
        result_dir = os.path.join(args.save_path, "inference", f"{subfolder}_{model_gnn}_{model_score}")
        os.makedirs(result_dir, exist_ok=True)
        auc_path = os.path.join(result_dir, f'auc_curve_{model_gnn}.png')
        plt.tight_layout()
        plt.savefig(auc_path)





    if len(all_test_labels) > 0:
        accuracy = accuracy_score(all_test_labels, all_test_preds)
        weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')
        report = classification_report(all_test_labels, all_test_preds)
    else:
        accuracy = -1
        weighted_f1_score = -1
        report = "No valid labels found (all were -1). Skipping metrics."

    # Build result folder: inference/{model_type}_{gnn}_{metric}
    result_dir = os.path.join(args.save_path, "inference", f"{subfolder}_{model_gnn}_{model_score}")
    os.makedirs(result_dir, exist_ok=True)

    # Generate a tight classification report image (zoomed in)
    from PIL import Image, ImageDraw, ImageFont

    # Use monospaced font
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size=20)
    except:
        font = ImageFont.load_default()

    report_lines = str(report).splitlines()
    line_height = font.getsize("A")[1]
    max_line_width = max(font.getsize(line)[0] for line in report_lines)

    # Set padding around the text
    padding = 20
    img_width = max_line_width + 2 * padding
    img_height = line_height * len(report_lines) + 2 * padding

    # Create image
    canvas = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Draw each line
    for i, line in enumerate(report_lines):
        y = padding + i * line_height
        x = padding

        # Check if line has class metrics and all >= 0.5
        tokens = line.split()
        if len(tokens) >= 5:
            try:
                precision_val = float(tokens[1])
                recall_val = float(tokens[2])
                f1_val = float(tokens[3])
                if precision_val >= 0.5 and recall_val >= 0.5 and f1_val >= 0.5:
                    bold_font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", size=20) if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf") else font
                    draw.text((x, y), line, fill="red", font=bold_font)
                    continue
            except:
                pass

        draw.text((x, y), line, fill="black", font=font)

        # Save output
        report_img_path = os.path.join(result_dir, f'classification_report_of_WSI_{model_gnn}.png')
        canvas.save(report_img_path)







    # ---- Boxplot of Validation Accuracy/F1 Across Models ----
    # You need to accumulate this data externally or across multiple runs.
    # Here's a mockup (replace with real history if available)

    df_metrics = pd.DataFrame({
        'Fold': list(range(1, len(all_model_accuracies) + 1)) * 2,
        'Metric': ['Accuracy'] * len(all_model_accuracies) + ['F1 Score'] * len(all_model_f1_scores),
        'Value': all_model_accuracies + all_model_f1_scores
    })

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Metric', y='Value', data=df_metrics, palette='Set3')
    plt.title("Performance Variability Across Runs")
    plt.tight_layout()
    result_dir = os.path.join(args.save_path, "inference", f"{subfolder}_{model_gnn}_{model_score}")
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, f'model_metrics_boxplot_{model_gnn}.png'))
    plt.close()

    avg_loss = np.mean(losses)

    if return_metrics:
        return accuracy, weighted_f1_score, avg_loss
    
    
    # Print the evaluation results.
    print('\nTest weighted F1 score {}'.format(weighted_f1_score))
    print('Test accuracy {}'.format(accuracy))
    print('Test classification report {}\n'.format(report))


# Main entry point to execute the testing script.
if __name__ == "__main__":
    main(args=parse_arguments())
