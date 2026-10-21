import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl

class PGExplainer(nn.Module):
    def __init__(self, model, num_hops=2, lr=0.01, num_epochs=30, temp=1.0):
        super(PGExplainer, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.temp = temp
        self.mask_generator = None

    def train(self, g, feat, label):
        self.model.eval()

        edge_index = g.edges()
        num_edges = g.number_of_edges()

        self.mask_generator = nn.Sequential(
            nn.Linear(feat.shape[1] * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        optimizer = torch.optim.Adam(self.mask_generator.parameters(), lr=self.lr)

        src, dst = edge_index
        device = next(self.mask_generator.parameters()).device
        edge_features = torch.cat([feat[src], feat[dst]], dim=1).to(device)
        label = label.to(device)

        for epoch in range(self.num_epochs):
            try:
                mask_logits = self.mask_generator(edge_features).squeeze()
                edge_mask = torch.sigmoid(mask_logits / self.temp)

                # Apply mask to edge weights
                g.edata['mask'] = edge_mask
                g.edata['masked_weight'] = edge_mask

                # Forward pass with edge mask
                output = self.model(g)
                if output is None:
                    print("[PGExplainer] Skipping graph: model output is None.")
                    return  # Skip training

                pred_loss = F.cross_entropy(output, label)

                # Encourage sparsity in edge mask
                mask_loss = edge_mask.mean()
                loss = pred_loss + 0.005 * mask_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except RuntimeError as e:
                print(f"[PGExplainer] Runtime error during training: {e}")
                print("Skipping PGExplainer training for this graph.")
                return  # Skip training

    def explain_graph(self, g, feat):
        self.model.eval()
        with torch.no_grad():
            src, dst = g.edges()
            device = next(self.mask_generator.parameters()).device
            edge_features = torch.cat([feat[src], feat[dst]], dim=1).to(device)
            mask_logits = self.mask_generator(edge_features).squeeze()
            edge_mask = torch.sigmoid(mask_logits / self.temp)
            return edge_mask
