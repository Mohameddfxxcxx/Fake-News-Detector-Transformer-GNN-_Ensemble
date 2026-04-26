"""Unified predictor used by the Gradio app and batch-CSV endpoint.

The predictor lazy-loads every trained checkpoint (BERT, DistilRoBERTa, GCN,
GraphSAGE, GAT, static-alpha ensemble, dynamic fusion) and exposes:

  * predict_text(text)       -> dict of per-model probabilities for one article
  * predict_batch(texts)     -> per-model predictions for N articles

Graph predictions for unseen articles use a simulated graph built on the fly
(article connected only to the K most similar articles from the training
corpus); metadata edges are dropped since a single free-text input has no
speaker/subject. This still lets the GNN contribute its learned aggregation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


OUT = Path("results")
CKPT = OUT / "checkpoints"

# model tag -> HF id
TEXT_MODELS = {
    "BERT":          ("transformer",   "bert-base-uncased"),
    "DistilRoBERTa": ("distilroberta", "distilroberta-base"),
}
GRAPH_MODELS = {
    "GCN":        ("gnn",  "gcn",  128),
    "GraphSAGE":  ("sage", "sage", 128),
    "GAT":        ("gat",  "gat",  64),
}


class Predictor:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._text_models: Dict[str, "torch.nn.Module"] = {}
        self._graph_models: Dict[str, "torch.nn.Module"] = {}
        self._tokenizers: Dict[str, "transformers.PreTrainedTokenizerBase"] = {}
        self._graph_cache = None  # features, edge_index, labels tensors
        self._ensemble_alpha: Optional[float] = None
        self._fusion = None
        self._fusion_cache = None

    # ----- loaders -----------------------------------------------------------
    def _load_text(self, label: str):
        if label in self._text_models:
            return self._text_models[label]
        tag, hf_name = TEXT_MODELS[label]
        ckpt = CKPT / f"{tag}_best.pt"
        if not ckpt.exists():
            return None
        from models.transformer_model import TransformerClassifier
        from utils.preprocess import get_tokenizer
        m = TransformerClassifier(pretrained_name=hf_name).to(self.device)
        m.load_state_dict(torch.load(ckpt, map_location=self.device))
        m.eval()
        self._text_models[label] = m
        self._tokenizers[label] = get_tokenizer(hf_name)
        return m

    def _load_graph_cache(self):
        if self._graph_cache is not None:
            return
        path = CKPT / "graph_data.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}. run train_gnn.py first.")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        self._graph_cache = cached

    def _load_graph(self, label: str):
        if label in self._graph_models:
            return self._graph_models[label]
        tag, conv, hidden = GRAPH_MODELS[label]
        ckpt = CKPT / f"{tag}_best.pt"
        if not ckpt.exists():
            return None
        from models.gnn_model import GNNClassifier
        self._load_graph_cache()
        in_channels = self._graph_cache["features"].size(1)
        g = GNNClassifier(in_channels=in_channels, hidden_channels=hidden,
                          conv=conv).to(self.device)
        g.load_state_dict(torch.load(ckpt, map_location=self.device))
        g.eval()
        self._graph_models[label] = g
        return g

    def _load_static_alpha(self) -> Optional[float]:
        if self._ensemble_alpha is not None:
            return self._ensemble_alpha
        ckpt = CKPT / "ensemble_best.pt"
        if not ckpt.exists():
            return None
        from models.ensemble import EnsembleClassifier
        e = EnsembleClassifier()
        e.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self._ensemble_alpha = float(e.alpha.item())
        return self._ensemble_alpha

    def _load_fusion(self):
        if self._fusion is not None:
            return self._fusion
        ckpt = CKPT / "fusion_best.pt"
        if not ckpt.exists():
            return None
        cache_path = CKPT / "fusion_cache_transformer_gnn.pt"
        if not cache_path.exists():
            return None
        self._fusion_cache = torch.load(cache_path, map_location="cpu",
                                        weights_only=False)
        from models.fusion import DynamicFusion
        fusion = DynamicFusion(
            dim_t=self._fusion_cache["dim_t"],
            dim_g=self._fusion_cache["dim_g"],
            hidden=128,
        ).to(self.device)
        fusion.load_state_dict(torch.load(ckpt, map_location=self.device))
        fusion.eval()
        self._fusion = fusion
        return fusion

    # ----- per-text helpers --------------------------------------------------
    @torch.no_grad()
    def _text_forward(self, label: str, texts: Sequence[str]):
        """Return logits, cls (both numpy arrays on CPU) for the given label."""
        m = self._load_text(label)
        if m is None:
            return None, None
        tok = self._tokenizers[label]
        enc = tok(list(texts), truncation=True, padding="max_length",
                  max_length=128, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits, cls = m(enc["input_ids"], enc["attention_mask"],
                        return_embedding=True)
        return logits.cpu().numpy(), cls.cpu().numpy()

    @torch.no_grad()
    def _graph_forward_with_neighbors(self, label: str, cls_feats: np.ndarray,
                                      k: int = 16) -> np.ndarray:
        """Inject new article(s) into the trained graph as k-nearest neighbours.

        Returns the per-article logits from the graph model (np array shape [N, 2]).
        """
        g = self._load_graph(label)
        if g is None:
            return None
        self._load_graph_cache()
        base_feats = self._graph_cache["features"].to(self.device).float()
        # normalize both for cosine knn
        bf = base_feats / (base_feats.norm(dim=1, keepdim=True) + 1e-12)
        new = torch.tensor(cls_feats, device=self.device).float()
        nn = new / (new.norm(dim=1, keepdim=True) + 1e-12)
        sims = nn @ bf.T                                           # [B, N_base]
        topk = sims.topk(k, dim=1).indices                         # [B, k]

        # build extended graph
        n_base = base_feats.size(0)
        n_new = new.size(0)
        full_feats = torch.cat([base_feats, new], dim=0)
        # copy base edges
        base_ei = self._graph_cache["edge_index"].to(self.device)
        src, dst = [base_ei[0]], [base_ei[1]]
        for b in range(n_new):
            new_id = n_base + b
            for j in topk[b].tolist():
                src.append(torch.tensor([new_id, j], device=self.device))
                dst.append(torch.tensor([j, new_id], device=self.device))
        src = torch.cat(src)
        dst = torch.cat(dst)
        full_ei = torch.stack([src, dst], dim=0)

        logits = g(full_feats, full_ei)                     # [N_total, 2]
        return logits[n_base:].cpu().numpy()

    # ----- public ------------------------------------------------------------
    def predict(self, texts: Sequence[str]) -> List[dict]:
        """Return a list of dicts (one per input) with probabilities per model."""
        # 1. text forward for every loaded text model
        text_logits: Dict[str, np.ndarray] = {}
        text_cls: Dict[str, np.ndarray] = {}
        for label in TEXT_MODELS:
            logits, cls = self._text_forward(label, texts)
            if logits is not None:
                text_logits[label] = logits
                text_cls[label] = cls

        # 2. graph forwards use BERT CLS as the feature vector
        graph_logits: Dict[str, np.ndarray] = {}
        ref_cls = text_cls.get("BERT")  # graph trained on BERT CLS
        if ref_cls is not None:
            for label in GRAPH_MODELS:
                gl = self._graph_forward_with_neighbors(label, ref_cls)
                if gl is not None:
                    graph_logits[label] = gl

        # 3. ensembles
        alpha = self._load_static_alpha()
        fusion = self._load_fusion()

        def softmax(x):
            e = np.exp(x - x.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        results = []
        for i, text in enumerate(texts):
            row = {"text": text, "per_model": {}}
            for label, logits in text_logits.items():
                p = softmax(logits)[i]
                row["per_model"][label] = {
                    "prob_fake": float(p[0]), "prob_real": float(p[1]),
                    "pred": "REAL" if p[1] >= 0.5 else "FAKE",
                    "confidence": float(max(p)),
                }
            for label, logits in graph_logits.items():
                p = softmax(logits)[i]
                row["per_model"][label] = {
                    "prob_fake": float(p[0]), "prob_real": float(p[1]),
                    "pred": "REAL" if p[1] >= 0.5 else "FAKE",
                    "confidence": float(max(p)),
                }
            # GETE (static alpha) — BERT + GCN
            if alpha is not None and "BERT" in text_logits and "GCN" in graph_logits:
                p_t = softmax(text_logits["BERT"])[i]
                p_g = softmax(graph_logits["GCN"])[i]
                p_e = alpha * p_t + (1 - alpha) * p_g
                row["per_model"]["GETE (static α)"] = {
                    "prob_fake": float(p_e[0]), "prob_real": float(p_e[1]),
                    "pred": "REAL" if p_e[1] >= 0.5 else "FAKE",
                    "confidence": float(max(p_e)), "alpha": alpha,
                }
            # Fusion
            if fusion is not None and "BERT" in text_cls and "GCN" in graph_logits:
                with torch.no_grad():
                    e_t = torch.tensor(text_cls["BERT"][i:i+1], device=self.device)
                    self._load_graph_cache()
                    # use the neighbours' average GCN hidden as graph embedding approx
                    # (we don't have hidden from the extended graph here). Fall back
                    # to projecting logits-as-embedding if fusion expects dim_g=2.
                    dim_g = self._fusion_cache["dim_g"]
                    if dim_g == 2:
                        e_g = torch.tensor(graph_logits["GCN"][i:i+1],
                                           device=self.device, dtype=torch.float32)
                    else:
                        # build GCN hidden on the extended graph
                        gcn = self._load_graph("GCN")
                        base_feats = self._graph_cache["features"].to(self.device).float()
                        bf = base_feats / (base_feats.norm(dim=1, keepdim=True) + 1e-12)
                        nn_t = torch.tensor(text_cls["BERT"][i:i+1],
                                            device=self.device).float()
                        nn_n = nn_t / (nn_t.norm(dim=1, keepdim=True) + 1e-12)
                        sims = nn_n @ bf.T
                        topk = sims.topk(16, dim=1).indices
                        n_base = base_feats.size(0)
                        full_feats = torch.cat([base_feats, nn_t], dim=0)
                        base_ei = self._graph_cache["edge_index"].to(self.device)
                        src, dst = [base_ei[0]], [base_ei[1]]
                        for j in topk[0].tolist():
                            src.append(torch.tensor([n_base, j], device=self.device))
                            dst.append(torch.tensor([j, n_base], device=self.device))
                        full_ei = torch.stack([torch.cat(src), torch.cat(dst)])
                        _, hidden = gcn(full_feats, full_ei, return_embedding=True)
                        e_g = hidden[n_base:n_base+1]

                    logits_t = torch.tensor(text_logits["BERT"][i:i+1],
                                            device=self.device)
                    logits_g = torch.tensor(graph_logits["GCN"][i:i+1],
                                            device=self.device)
                    out = fusion(e_t.float(), e_g.float(),
                                 logits_t.float(), logits_g.float())
                    p_e = out["probs_mixed"][0].cpu().numpy()
                    a = out["attention"][0].cpu().numpy()
                    row["per_model"]["GETE (dynamic fusion)"] = {
                        "prob_fake": float(p_e[0]), "prob_real": float(p_e[1]),
                        "pred": "REAL" if p_e[1] >= 0.5 else "FAKE",
                        "confidence": float(max(p_e)),
                        "alpha_text": float(a[0]),
                        "alpha_graph": float(a[1]),
                    }
            results.append(row)
        return results


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
