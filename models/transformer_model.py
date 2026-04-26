"""Generic transformer classifier.

Matches paper text module (BERT CLS token -> dropout -> linear). The same
class now also supports RoBERTa / DistilRoBERTa / DeBERTa via AutoModel,
so we can drop in any HF encoder for the ensemble.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertModel


class TransformerClassifier(nn.Module):
    """Wrap any HF encoder as a binary CLS-head classifier.

    `pretrained_name` is any HF model id (e.g. "bert-base-uncased",
    "distilroberta-base", "roberta-base", "microsoft/deberta-v3-small").
    """

    def __init__(self, pretrained_name: str = "bert-base-uncased",
                 dropout: float = 0.3, num_classes: int = 2):
        super().__init__()
        self.pretrained_name = pretrained_name
        try:
            self.backbone = AutoModel.from_pretrained(pretrained_name)
        except Exception:
            # fall-back for older caches
            self.backbone = BertModel.from_pretrained(pretrained_name)
        self.config = self.backbone.config
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    # kept for backward compatibility
    @property
    def bert(self):
        return self.backbone

    def load_state_dict(self, state_dict, strict: bool = True):
        """Accept state dicts saved with the legacy 'bert.*' prefix."""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("bert."):
                remapped["backbone." + k[len("bert."):]] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)

    def _cls(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Most encoders (BERT, RoBERTa, DeBERTa) expose .last_hidden_state.
        # DistilBert/DistilRoberta do as well. The CLS-equivalent is index 0.
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask, return_embedding: bool = False):
        cls_output = self._cls(input_ids, attention_mask)
        logits = self.classifier(self.dropout(cls_output))
        if return_embedding:
            return logits, cls_output
        return logits

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        self.eval()
        return self._cls(input_ids, attention_mask)
