"""Graph neural network classifier.

Matches paper §Graph Module: 2-layer GCN (or GraphSAGE) over article nodes
with BERT CLS embeddings as features, producing per-article logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        conv: str = "gcn",
        heads: int = 4,
    ):
        super().__init__()
        self.conv_type = conv.lower()
        if self.conv_type == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif self.conv_type == "sage":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif self.conv_type == "gat":
            # multi-head attention on layer 1, concat -> (hidden * heads);
            # layer 2 averages heads to stay at hidden
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                                 dropout=dropout)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                                 heads=1, concat=False, dropout=dropout)
        else:
            raise ValueError(f"unknown conv type: {conv}")
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embedding: bool = False):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
