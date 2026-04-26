"""Meta-learned ensemble: y = alpha * pT + (1 - alpha) * pG.

Matches paper Eqs. (2)-(4). Alpha is a scalar sigmoid-parameterised weight,
learned end-to-end on the validation set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # raw parameter, sigmoid-ed to stay in (0,1)
        self._alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha)

    def forward(self, logits_transformer: torch.Tensor, logits_gnn: torch.Tensor) -> torch.Tensor:
        pT = F.softmax(logits_transformer, dim=1)
        pG = F.softmax(logits_gnn, dim=1)
        return self.alpha * pT + (1.0 - self.alpha) * pG
