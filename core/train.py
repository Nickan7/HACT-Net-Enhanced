#!/usr/bin/env python3

"""
Script for training CG-GNN, TG-GNN and HACT models
"""
import torch
import mlflow
import os
import uuid
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import pandas as pd
import shutil
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel

from dataloader import make_data_loader

import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True
cudnn.allow_tf32 = False  
torch.backends.cudnn.enabled = False



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
        help='path to where the model is saved.',
        default='',
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
        required=False
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
        '--epochs', type=int, help='epochs.', default=10, required=False
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='learning rate.',
        default=10e-3,
        required=False
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path to where the output data are saved (currently only for the interpretability).',
        default='../../data/graphs',
        required=False
    )
    parser.add_argument(
        '--logger',
        type=str,
        help='Logger type. Options are "mlflow" or "none"',
        required=False,
        default='none'
    )

    return parser.parse_args()

def main(args):
    """
    Train HACTNet, CG-GNN or TG-GNN.
    Args:
        args (Namespace): parsed arguments.
    """

    # load config file
    with open(args.config_fpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # log parameters to logger
    if args.logger == 'mlflow':
        mlflow.log_params({
            'batch_size': args.batch_size
        })
        df = pd.io.json.json_normalize(config)
        rep = {"graph_building.": "", "model_params.": "", "gnn_params.": ""}  # replacement for shorter key names
        for i, j in rep.items():
            df.columns = df.columns.str.replace(i, j)
        flatten_config = df.to_dict(orient='records')[0]
        for key, val in flatten_config.items():
            mlflow.log_params({key: str(val)})


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

    # Update the model path to include this subfolder
    model_path = os.path.join(args.out_path, f"{subfolder}_{model_gnn}")
    os.makedirs(model_path, exist_ok=True)


    # make data loaders (train, validation & test)
    train_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'train') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'train') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'train') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    val_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'val') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'val') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'val') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    test_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'test') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'test') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )

    # declare model 
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
    else:
        raise ValueError('Model type not recognized. Options are: TG, CG or HACT.')

    # build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=5e-4
    )

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # training loop
    step = 0
    best_val_loss = 100000
    best_val_accuracy = 0.
    best_val_weighted_f1_score = 0.

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    val_f1_history = [] 
    train_f1_history = []


    
    for epoch in range(args.epochs):
        # A.) train for 1 epoch
        model = model.to(DEVICE)
        model.train()

        train_epoch_losses = []
        train_epoch_correct = 0
        train_epoch_total = 0

        

        for batch in tqdm(train_dataloader, desc='\n\nEpoch training {}'.format(epoch), unit='batch'):

            # 1. forward pass
            data = batch[:-1]
            

            for i in range(len(data)):
                #element = data[i]

                # If it's a DGL graph
                if hasattr(data[i], 'ndata'):
                    if 'feat' in data[i].ndata:
                        data[i].ndata['feat'] = torch.nan_to_num(data[i].ndata['feat'], nan=0.0)
                        data[i].ndata['feat'] = torch.clamp(data[i].ndata['feat'], min=-1e3, max=1e3)

                    elif 'centroid' in data[i].ndata:
                        data[i].ndata['centroid'] = torch.nan_to_num(data[i].ndata['centroid'], nan=0.0)
                        data[i].ndata['centroid'] = torch.clamp(data[i].ndata['centroid'], min=-1e3, max=1e3)

                # If it's a tensor
                elif isinstance(data[i], torch.Tensor):
                    data[i] = torch.nan_to_num(data[i], nan=0.0)
              
            labels = batch[-1]
            labels = torch.nan_to_num(labels, nan=0.0)

            try:
                logits = model(*data)
                if logits is None:
                    print("Skipping batch due to assignment/feature shape mismatch.")
                    continue
            except RuntimeError as e:
                print(f"[ERROR] Runtime failure during forward pass: {e}")
                print("Skipping this batch.")
                continue


            # 2. backward pass
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_losses.append(loss)

            preds = torch.argmax(logits, dim=1)
            train_epoch_correct += (preds == labels).sum().item()
            train_epoch_total += labels.size(0)



            # 3. log training loss 
            if args.logger == 'mlflow':
                mlflow.log_metric('train_loss', loss.item(), step=step)

            # 4. increment step
            step += 1
            
       
        
        # B.) validate
        model.eval()
        all_val_logits = []
        all_val_labels = []

        for batch in tqdm(val_dataloader, desc='\n\nEpoch validation {}'.format(epoch), unit='batch'):
            
            data = batch[:-1]

            for i in range(len(data)):
                #element = data[i]

                # If it's a DGL graph
                if hasattr(data[i], 'ndata'):
                    if 'feat' in data[i].ndata:
                        data[i].ndata['feat'] = torch.nan_to_num(data[i].ndata['feat'], nan=0.0)
                        data[i].ndata['feat'] = torch.clamp(data[i].ndata['feat'], min=-1e3, max=1e3)

                    elif 'centroid' in data[i].ndata:
                        data[i].ndata['centroid'] = torch.nan_to_num(data[i].ndata['centroid'], nan=0.0)
                        data[i].ndata['centroid'] = torch.clamp(data[i].ndata['centroid'], min=-1e3, max=1e3)


                # If it's a tensor
                elif isinstance(data[i], torch.Tensor):
                    data[i] = torch.nan_to_num(data[i], nan=0.0)

            labels = batch[-1]
            labels = torch.nan_to_num(labels, nan=0.0)

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


            all_val_logits.append(logits)
            all_val_labels.append(labels)

        

        all_val_logits = torch.cat(all_val_logits).cpu()
        all_val_preds = torch.argmax(all_val_logits, dim=1)
        all_val_labels = torch.cat(all_val_labels).cpu()


        
        # compute & store loss + model
        with torch.no_grad():
            loss = loss_fn(all_val_logits, all_val_labels).item()
        if args.logger == 'mlflow':
            mlflow.log_metric('val_loss', loss, step=step)
            
        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_loss.pt'))

        # compute & store accuracy + model
        all_val_preds = all_val_preds.detach().numpy()
        all_val_labels = all_val_labels.detach().numpy()
        accuracy = accuracy_score(all_val_labels, all_val_preds)


        if args.logger == 'mlflow':
            mlflow.log_metric('val_accuracy', accuracy, step=step)
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_accuracy.pt'))



        # compute & store weighted f1-score + model
        weighted_f1_score = f1_score(all_val_labels, all_val_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('val_weighted_f1_score', weighted_f1_score, step=step)
        if weighted_f1_score > best_val_weighted_f1_score:
            best_val_weighted_f1_score = weighted_f1_score
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_weighted_f1_score.pt'))

        print('\nVal loss {}'.format(loss))
        print('Val weighted F1 score {}'.format(weighted_f1_score))
        print('Val accuracy {}\n'.format(accuracy))

        avg_train_loss = np.mean([l.item() for l in train_epoch_losses])
        train_accuracy = train_epoch_correct / train_epoch_total
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)

        # Compute and append train F1 score
        train_preds = torch.argmax(logits, dim=1)
        train_labels = labels
        train_f1_score = f1_score(train_labels.cpu(), train_preds.cpu(), average='weighted')
        train_f1_history.append(train_f1_score)

        # Append validation metrics after each epoch
        val_loss_history.append(loss)
        val_acc_history.append(accuracy)
        val_f1_history.append(weighted_f1_score)


    # testing loop
    model.eval()
    for metric in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:
        
        print('\n\n*** Start testing w/ {} model ***'.format(metric))
        
        model_name = [f for f in os.listdir(model_path) if f.endswith(".pt") and metric in f][0]
        model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

        all_test_logits = []
        all_test_labels = []

        for batch in tqdm(test_dataloader, desc='Testing: {}'.format(metric), unit='batch'):
            
            data = batch[:-1]

            for i in range(len(data)):
                
                # If it's a DGL graph
                if hasattr(data[i], 'ndata'):
                    if 'feat' in data[i].ndata:
                        data[i].ndata['feat'] = torch.nan_to_num(data[i].ndata['feat'], nan=0.0)
                        data[i].ndata['feat'] = torch.clamp(data[i].ndata['feat'], min=-1e3, max=1e3)

                    if 'centroid' in data[i].ndata:
                        data[i].ndata['centroid'] = torch.nan_to_num(data[i].ndata['centroid'], nan=0.0)
                        data[i].ndata['centroid'] = torch.clamp(data[i].ndata['centroid'], min=-1e3, max=1e3)

                # If it's a tensor
                elif isinstance(data[i], torch.Tensor):
                    data[i] = torch.nan_to_num(data[i], nan=0.0)

            labels = batch[-1]
            labels = torch.nan_to_num(labels, nan=0.0)

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


            
            all_test_logits.append(logits)
            all_test_labels.append(labels)

        

        all_test_logits = torch.cat(all_test_logits).cpu()
        all_test_preds = torch.argmax(all_test_logits, dim=1)
        all_test_labels = torch.cat(all_test_labels).cpu()

        # compute & store loss
        with torch.no_grad():
            loss = loss_fn(all_test_logits, all_test_labels).item()
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_loss_' + metric, loss)

        # compute & store accuracy
        all_test_preds = all_test_preds.detach().numpy()
        all_test_labels = all_test_labels.detach().numpy()
        accuracy = accuracy_score(all_test_labels, all_test_preds)

        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_accuracy_' + metric, accuracy, step=step)

        # compute & store weighted f1-score
        weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_weighted_f1_score_' + metric, weighted_f1_score, step=step)

        # compute and store classification report 
        report = classification_report(all_test_labels, all_test_preds)
        out_path = os.path.join(model_path, 'classification_report_of_ROIs.txt')
        with open(out_path, "w") as f:
            f.write(report)

        if args.logger == 'mlflow':
            artifact_path = 'evaluators/class_report_{}'.format(metric)
            mlflow.log_artifact(out_path, artifact_path=artifact_path)

        # log MLflow models
        mlflow.pytorch.log_model(model, 'model_' + metric)

        val_loss_history.append(loss)
        val_acc_history.append(accuracy)



        print('\nTest loss {}'.format(loss))
        print('Test weighted F1 score {}'.format(weighted_f1_score))
        print('Test accuracy {}\n'.format(accuracy))


    if args.logger == 'mlflow':
        shutil.rmtree(model_path)



    epochs = list(range(1, len(train_loss_history) + 1))

    plt.figure(figsize=(12, 6))

    # Only plot if lengths match
    if len(train_loss_history) == len(epochs):
        plt.plot(epochs, train_loss_history, label='Train Loss', linestyle='-', marker='o', color='blue')
    if len(val_loss_history) == len(epochs):
        plt.plot(epochs, val_loss_history, label='Val Loss', linestyle='--', marker='o', color='dodgerblue')
    if len(train_acc_history) == len(epochs):
        plt.plot(epochs, train_acc_history, label='Train Accuracy', linestyle='-', marker='s', color='orange')
    if len(val_acc_history) == len(epochs):
        plt.plot(epochs, val_acc_history, label='Val Accuracy', linestyle='--', marker='s', color='darkorange')
    if len(val_f1_history) == len(epochs):
        plt.plot(epochs, val_f1_history, label='Val F1 Score', linestyle='-', marker='^', color='red')
    if len(train_f1_history) == len(epochs):
        plt.plot(epochs, train_f1_history, label='Train F1 Score', linestyle='--', marker='^', color='green')

    plt.title('Training and Validation Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.grid(True)
    plt.tight_layout()


    # Ensure the folder exists (not the file path)
    os.makedirs(args.out_path, exist_ok=True)

    # Create the 'lineplot' folder inside your output path
    plot_folder = os.path.join(args.out_path, f'lineplot_{model_gnn}')
    os.makedirs(plot_folder, exist_ok=True)

    # Define the full path to the plot file
    plot_path = os.path.join(plot_folder, f'training_validation_lineplot_{model_gnn}.png')
    plt.savefig(plot_path)

    plt.close()



if __name__ == "__main__":
    main(args=parse_arguments())