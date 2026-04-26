"""Dynamic attention fusion of transformer + GNN outputs.

Replaces the static scalar alpha of Kumar et al. with a per-example
attention over the two predictive branches:

    h_t = W_t * e_t                     # text projection
    h_g = W_g * e_g                     # graph projection
    K   = stack(h_t, h_g)                # (B, 2, d)
    a   = softmax(K @ q)                 # (B, 2)
    y   = a[0] * p_T + a[1] * p_G        # weighted ensemble probs

Optionally also produces a classifier head on the fused embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFusion(nn.Module):
    def __init__(self, dim_t: int = 768, dim_g: int = 128,
                 hidden: int = 128, num_classes: int = 2,
                 dropout: float = 0.3, use_fused_head: bool = True):
        super().__init__()
        self.proj_t = nn.Linear(dim_t, hidden)
        self.proj_g = nn.Linear(dim_g, hidden)
        self.query = nn.Parameter(torch.randn(hidden) * 0.01)
        self.dropout = nn.Dropout(dropout)
        self.use_fused_head = use_fused_head
        if use_fused_head:
            self.head = nn.Linear(hidden, num_classes)

    def attention_weights(self, e_t: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        h_t = torch.tanh(self.proj_t(e_t))
        h_g = torch.tanh(self.proj_g(e_g))
        K = torch.stack([h_t, h_g], dim=1)              # (B, 2, hidden)
        scores = (K * self.query.view(1, 1, -1)).sum(-1)  # (B, 2)
        return F.softmax(scores, dim=-1)

    def fused_embedding(self, e_t, e_g):
        h_t = torch.tanh(self.proj_t(e_t))
        h_g = torch.tanh(self.proj_g(e_g))
        K = torch.stack([h_t, h_g], dim=1)
        scores = (K * self.query.view(1, 1, -1)).sum(-1)
        a = F.softmax(scores, dim=-1)
        fused = (a.unsqueeze(-1) * K).sum(dim=1)
        return fused, a

    def forward(self, e_t, e_g, logits_t=None, logits_g=None):
        fused, a = self.fused_embedding(e_t, e_g)
        fused = self.dropout(fused)
        logits_head = self.head(fused) if self.use_fused_head else None

        if logits_t is not None and logits_g is not None:
            p_t = F.softmax(logits_t, dim=-1)
            p_g = F.softmax(logits_g, dim=-1)
            probs_mixed = a[:, 0:1] * p_t + a[:, 1:2] * p_g
        else:
            probs_mixed = None

        return {
            "attention": a,           # (B, 2)
            "fused_embedding": fused, # (B, hidden)
            "logits_head": logits_head,
            "probs_mixed": probs_mixed,
        }
