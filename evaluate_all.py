"""Evaluate every trained model (old + new) on the LIAR test split.

Loaded models:
  - BERT, DistilRoBERTa (text)
  - GCN, GraphSAGE, GAT (graph, over BERT CLS features)
  - GETE static-alpha ensemble (BERT + GCN)
  - GETE dynamic-fusion ensemble (BERT + GCN)

Outputs (merging into results/):
  figures/model_comparison_bar_all.png
  figures/roc_curves_all.png
  figures/pr_curves_all.png
  figures/confusion_matrix_<tag>.png
  figures/dynamic_alpha_distribution.png
  tables/metrics_summary_all.csv
  tables/per_class_report_<tag>.csv
  metrics/metrics_all_full.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ensemble import EnsembleClassifier
from models.fusion import DynamicFusion
from models.gnn_model import GNNClassifier
from models.transformer_model import TransformerClassifier
from utils.preprocess import LIARDataset


OUT  = Path("results")
FIGS = OUT / "figures"
TABS = OUT / "tables"
METS = OUT / "metrics"
LOGS = OUT / "logs"
CKPT = OUT / "checkpoints"
for d in (FIGS, TABS, METS, LOGS, CKPT):
    d.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 140


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def metrics_row(name: str, y_true, y_pred, y_prob) -> dict:
    return {
        "model": name,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "pr_auc":    average_precision_score(y_true, y_prob),
    }


def save_cm(y_true, y_pred, title: str, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"],
                ax=ax, cbar=False)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


# -----------------------------------------------------------------------------
def compute_text_model_logits(text_tag: str, model_name: str, device):
    """Return (logits_te numpy, cls_te numpy, labels numpy) on test set."""
    ckpt = CKPT / f"{text_tag}_best.pt"
    if not ckpt.exists():
        return None
    test_ds = LIARDataset("data/test.tsv", tokenizer_name=model_name)
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=16)
    m = TransformerClassifier(pretrained_name=model_name).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    logits, cls = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lo, c = m(ids, mask, return_embedding=True)
            logits.append(lo.cpu()); cls.append(c.cpu())
    del m; torch.cuda.empty_cache()
    return (torch.cat(logits).numpy(),
            torch.cat(cls).numpy(),
            np.array(test_ds.labels))


def compute_graph_model_logits(tag: str, conv: str, hidden: int, device):
    ckpt = CKPT / f"{tag}_best.pt"
    if not ckpt.exists():
        return None
    graph = torch.load(CKPT / "graph_data.pt", map_location="cpu", weights_only=False)
    m = GNNClassifier(in_channels=graph["features"].size(1),
                      hidden_channels=hidden, conv=conv).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    n_tr, n_va, n_te = graph["sizes"]
    with torch.no_grad():
        logits = m(graph["features"].to(device), graph["edge_index"].to(device)).cpu().numpy()
    return logits[n_tr + n_va:]


# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_all] device={device}")

    text_models = {
        "BERT":          ("transformer",   "bert-base-uncased"),
        "DistilRoBERTa": ("distilroberta", "distilroberta-base"),
    }
    graph_models = {
        "GCN":       ("gnn",  "gcn",  128),
        "GraphSAGE": ("sage", "sage", 128),
        "GAT":       ("gat",  "gat",   64),
    }

    logits: Dict[str, np.ndarray] = {}
    cls: Dict[str, np.ndarray] = {}
    y_te = None

    for name, (tag, hf) in text_models.items():
        print(f"[eval_all] text: {name} ({hf})...")
        out = compute_text_model_logits(tag, hf, device)
        if out is None:
            print(f"[eval_all] skip {name} (missing {tag}_best.pt)")
            continue
        lo, c, y = out
        logits[name] = lo; cls[name] = c; y_te = y

    for name, (tag, conv, hidden) in graph_models.items():
        print(f"[eval_all] graph: {name} ({conv}, h={hidden})...")
        lo = compute_graph_model_logits(tag, conv, hidden, device)
        if lo is None:
            print(f"[eval_all] skip {name} (missing {tag}_best.pt)")
            continue
        logits[name] = lo

    # --- ensembles ----------------------------------------------------------
    # static alpha
    alpha_ckpt = CKPT / "ensemble_best.pt"
    if alpha_ckpt.exists() and "BERT" in logits and "GCN" in logits:
        e = EnsembleClassifier()
        e.load_state_dict(torch.load(alpha_ckpt, map_location="cpu"))
        alpha = float(e.alpha.item())
        p_e = alpha * softmax(logits["BERT"]) + (1 - alpha) * softmax(logits["GCN"])
        logits["GETE (static α=%.3f)" % alpha] = np.log(p_e + 1e-12)
    else:
        alpha = None

    # dynamic fusion
    fusion_ckpt = CKPT / "fusion_best.pt"
    fusion_cache = CKPT / "fusion_cache_transformer_gnn.pt"
    if fusion_ckpt.exists() and fusion_cache.exists():
        cache = torch.load(fusion_cache, map_location="cpu", weights_only=False)
        fusion = DynamicFusion(dim_t=cache["dim_t"], dim_g=cache["dim_g"],
                               hidden=128).to(device)
        fusion.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        fusion.eval()
        with torch.no_grad():
            e_t = cache["embeddings"]["text"]["test"].to(device)
            e_g = cache["embeddings"]["graph"]["test"].to(device)
            l_t = cache["logits"]["text"]["test"].to(device)
            l_g = cache["logits"]["graph"]["test"].to(device)
            out = fusion(e_t, e_g, l_t, l_g)
            probs_mixed = out["probs_mixed"].cpu().numpy()
            logits_head = out["logits_head"].cpu().numpy()
            attention = out["attention"].cpu().numpy()
        logits["GETE (dynamic fusion, mixed)"] = np.log(probs_mixed + 1e-12)
        logits["GETE (dynamic fusion, head)"]  = logits_head
    else:
        attention = None

    # --- compute metrics ----------------------------------------------------
    rows = []
    probs_pos = {}
    predictions = {}
    order = [
        "BERT", "DistilRoBERTa", "GCN", "GraphSAGE", "GAT",
    ] + [k for k in logits if k.startswith("GETE")]
    order = [k for k in order if k in logits]

    for name in order:
        p = softmax(logits[name])
        pred = p.argmax(-1)
        rows.append(metrics_row(name, y_te, pred, p[:, 1]))
        probs_pos[name] = p[:, 1]
        predictions[name] = pred

    df = pd.DataFrame(rows)
    df.to_csv(TABS / "metrics_summary_all.csv", index=False)
    print("[eval_all] leaderboard:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # per-class reports
    for name, pred in predictions.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")\
                   .replace("α=", "alpha_").replace(".", "").replace(",", "")
        rep = classification_report(y_te, pred, target_names=["FAKE", "REAL"],
                                    output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(TABS / f"per_class_report_{safe}.csv")
        save_cm(y_te, pred, f"Confusion Matrix - {name}",
                FIGS / f"confusion_matrix_{safe}.png")

    # --- aggregate plots ---------------------------------------------------
    # ROC
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in order:
        fpr, tpr, _ = roc_curve(y_te, probs_pos[name])
        auc = roc_auc_score(y_te, probs_pos[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — all models on LIAR test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "roc_curves_all.png"); plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name in order:
        prec, rec, _ = precision_recall_curve(y_te, probs_pos[name])
        ap = average_precision_score(y_te, probs_pos[name])
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", lw=1.8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - all models")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "pr_curves_all.png"); plt.close(fig)

    # bar comparison
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(df)), 5))
    width = 0.2
    x = np.arange(len(df))
    ax.bar(x - 1.5*width, df["accuracy"],  width, label="Accuracy")
    ax.bar(x - 0.5*width, df["precision"], width, label="Precision")
    ax.bar(x + 0.5*width, df["recall"],    width, label="Recall")
    ax.bar(x + 1.5*width, df["f1"],        width, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score"); ax.set_title("All-model leaderboard on LIAR test")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    fig.tight_layout(); fig.savefig(FIGS / "model_comparison_bar_all.png",
                                    bbox_inches="tight"); plt.close(fig)

    # dynamic alpha distribution
    if attention is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        alpha_text = attention[:, 0]
        ax.hist(alpha_text, bins=30, color="tab:purple", edgecolor="k", alpha=0.85)
        ax.axvline(alpha_text.mean(), ls="--", color="tab:orange",
                   label=f"mean={alpha_text.mean():.3f}")
        ax.set_xlabel("Attention on text branch (α) — dynamic fusion")
        ax.set_ylabel("# test articles")
        ax.set_title("Per-example α distribution (dynamic fusion)")
        ax.legend()
        fig.tight_layout(); fig.savefig(FIGS / "dynamic_alpha_distribution.png")
        plt.close(fig)

    # --- JSON summary ------------------------------------------------------
    summary = {
        "static_alpha": alpha,
        "dynamic_alpha_text_mean":
            float(attention[:, 0].mean()) if attention is not None else None,
        "dynamic_alpha_text_std":
            float(attention[:, 0].std())  if attention is not None else None,
        "per_model": {r["model"]: {k: v for k, v in r.items() if k != "model"}
                      for r in rows},
        "test_size": int(len(y_te)),
        "class_balance": {"FAKE": int((y_te == 0).sum()),
                          "REAL": int((y_te == 1).sum())},
    }
    with open(METS / "metrics_all_full.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval_all] wrote {METS / 'metrics_all_full.json'}")


if __name__ == "__main__":
    main()
