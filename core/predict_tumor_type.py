import os
import argparse
import torch
import numpy as np
from dgl.data.utils import load_graphs
from histocartography.ml import (
    CellGraphModel,
    TissueGraphModel,
    HACTModel
)
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import yaml
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Label mapping (reverse of training)
LABELS = ['N', 'PB', 'UDH', 'ADH', 'FEA', 'DCIS', 'IC']
IS_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if IS_CUDA else 'cpu')
NODE_DIM = 514

def load_cell_graph(bin_path):
    graphs, labels = load_graphs(bin_path)
    return graphs[0]

def predict(model, graph):
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        probs = F.softmax(logits, dim=-1).squeeze()
        pred_class = torch.argmax(probs).item()
    return pred_class, probs.cpu().numpy()

def plot_boxplot(prediction_scores, save_path):
    predict_dir = os.path.join(save_path, "predict")
    os.makedirs(predict_dir, exist_ok=True)

    data = []
    for image, probs in prediction_scores.items():
        for i, score in enumerate(probs):
            data.append({'Tumor Type': LABELS[i], 'Confidence': score})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Tumor Type', y='Confidence', palette='Reds')
    plt.title('Confidence Distribution per Tumor Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(predict_dir, 'boxplot_confidence.png'))
    plt.close()

def plot_scatterplot(prediction_scores, save_path):
    predict_dir = os.path.join(save_path, "predict")
    os.makedirs(predict_dir, exist_ok=True)

    data = []
    for image, probs in prediction_scores.items():
        pred_class = np.argmax(probs)
        data.append({
            'Image': image,
            'Confidence': probs[pred_class],
            'Predicted Tumor Type': LABELS[pred_class]
        })

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='Image', y='Confidence', hue='Predicted Tumor Type', palette='tab10')
    plt.title('Prediction Confidence per Image')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(predict_dir, 'scatterplot_confidence.png'))
    plt.close()


def plot_prediction_bar(image_id, probs, pred_label, pred_class, save_path):
    predict_dir = os.path.join(save_path, "predict")
    os.makedirs(predict_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.bar(LABELS, probs, color='tomato')
    plt.ylabel("Confidence")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.figtext(0.5, -0.15, f"The image '{image_id}' was predicted as tumor type: '{pred_label}' (class {pred_class})",
                wrap=True, horizontalalignment='center', fontsize=13)
    plt.tight_layout()
    save_name = os.path.join(predict_dir, f"prediction_{image_id}.png")
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction histogram as {save_name}")

def plot_countplot(predictions, save_path):
    predict_dir = os.path.join(save_path, "predict")
    os.makedirs(predict_dir, exist_ok=True)

    labels = [LABELS[p] for p in predictions]
    df = pd.DataFrame({'Predicted Tumor Type': labels})

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Predicted Tumor Type', palette='Reds')
    plt.title('Prediction Count per Tumor Type')
    plt.ylabel('Number of Predictions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(predict_dir, 'countplot_predictions.png'))
    plt.close()


def plot_heatmap(predictions_dict, save_path, image_base_path, summary_text):
    heatmap_data = np.array(list(predictions_dict.values()))
    image_names = list(predictions_dict.keys())

    predict_dir = os.path.join(save_path, "predict")
    os.makedirs(predict_dir, exist_ok=True)
    save_file = os.path.join(predict_dir, "heatmap_confidences.png")

    fig = plt.figure(figsize=(len(LABELS) * 2.5, max(4, len(image_names)*0.6)), constrained_layout=True)
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])

    sns.heatmap(heatmap_data, annot=True, cmap='Reds', xticklabels=LABELS, yticklabels=image_names, vmin=0, vmax=1,ax=ax, cbar_kws={'shrink': 0.6, 'aspect': 20})
    plt.title("Confidence Scores per Image")
    plt.xlabel("Tumor Type")
    plt.ylabel("Image")
    plt.subplots_adjust(bottom=0.1)
    plt.figtext(0.5, 0.01, summary_text, wrap=True, horizontalalignment='center', fontsize=19)
    plt.savefig(save_file)
    plt.close()
    print(f"Saved heatmap as {save_file}")

def main(args):
    with open(args.config_fpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if 'bracs_cggnn' in args.config_fpath:
        model = CellGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)
    elif 'bracs_tggnn' in args.config_fpath:
        model = TissueGraphModel(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)
    elif 'bracs_hact' in args.config_fpath:
        model = HACTModel(
            cg_gnn_params=config['cg_gnn_params'],
            tg_gnn_params=config['tg_gnn_params'],
            classification_params=config['classification_params'],
            cg_node_dim=NODE_DIM,
            tg_node_dim=NODE_DIM,
            num_classes=7
        ).to(DEVICE)

    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    predictions = []
    prediction_scores = {}

    if args.exact_image:
        image_names = [os.path.basename(args.exact_image)]
        image_base_path = os.path.dirname(args.exact_image)
    elif args.image_path:
        image_names = [f for f in os.listdir(args.image_path) if f.endswith(".svs") or f.endswith(".png") or f.endswith(".bin")]
        image_base_path = args.image_path
    else:
        raise ValueError("Either --image_path or --exact_image must be provided.")

    for image_name in image_names:
        image_id = os.path.splitext(os.path.basename(image_name))[0]
        cg_bin = os.path.join(args.cg_path, image_id + '.bin')

        if not os.path.exists(cg_bin):
            print(f"[WARNING] Graph for {image_id} not found. Skipping.")
            continue

        cell_graph = load_cell_graph(cg_bin).to(DEVICE)
        pred_class, probs = predict(model, cell_graph)
        predictions.append(pred_class)
        prediction_scores[image_id] = probs

        pred_label = LABELS[pred_class]
        print(f"\nPredicted tumor type for image '{image_id}': {pred_label} (class {pred_class})")
        print("Confidence per class:")
        for i, (label, prob) in enumerate(zip(LABELS, probs)):
            print(f"  {label} ({i}): {prob:.4f}")

        if args.exact_image:
            plot_prediction_bar(image_id, probs, pred_label, pred_class, args.save_path)

    # Summary and heatmap (only for --image_path)
    if args.image_path and predictions:
        counter = Counter(predictions)
        most_common_class, count = counter.most_common(1)[0]
        summary_text = f"In the folder '{image_base_path}', the most frequently predicted tumor type is '{LABELS[most_common_class]}' (class {most_common_class}), found in {count}/{len(predictions)} images."
        print(f"\n{summary_text}\n")

        plot_countplot(predictions, args.save_path)
        plot_boxplot(prediction_scores, args.save_path)
        plot_scatterplot(prediction_scores, args.save_path)
        plot_heatmap(prediction_scores, args.save_path, image_base_path, summary_text)
  
    elif args.image_path:
        print("\nNo predictions were made.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict tumor type from WSI folder or single image")
    parser.add_argument('--image_path', type=str, help='Path to folder with .svs or .bin files')
    parser.add_argument('--exact_image', type=str, help='Path to a single image file')
    parser.add_argument('--cg_path', type=str, required=True, help='Folder containing cell graph .bin files')
    parser.add_argument('--model_path', type=str, required=True, help='Trained .pt model')
    parser.add_argument('--save_path', type=str, required=True, help='Folder to save prediction plots and outputs')
    parser.add_argument('--config_fpath', type=str, required=True, help='YAML config used for training')
    args = parser.parse_args()
    main(args)
