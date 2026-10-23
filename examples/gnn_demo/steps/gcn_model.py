"""The real model for the train step: a PyTorch Geometric GCN for graph-level regression.

Needs torch + torch_geometric (the toy pipeline runs without them). mltree's GNN view reads this file
statically (no import) and shows the layers: 3 x GCNConv, hidden 64, global mean pooling, a linear head.
"""
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

NUM_ATOM_FEATURES = 7  # 6 element one-hots + aromatic flag (see featurize.py)


class MolGCN(nn.Module):
    def __init__(self, in_channels=NUM_ATOM_FEATURES, hidden_channels=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.norm = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = self.dropout(self.norm(global_mean_pool(x, batch)))
        return self.head(x).squeeze(-1)
