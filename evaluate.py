"""Evaluate all three models and generate every paper figure/table.

Outputs (under results/):
  figures/training_curves_transformer.png
  figures/training_curves_gnn.png
  figures/alpha_progression.png
  figures/confusion_matrix_proposed.png
  figures/confusion_matrix_<model>.png (bert / gnn / ensemble)
  figures/roc_curves.png
  figures/pr_curves.png
  figures/model_comparison_bar.png
  figures/pca_embeddings.png
  tables/metrics_summary.csv
  tables/per_class_report_<model>.csv
  metrics/metrics_all.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ensemble import EnsembleClassifier


OUT = Path("results")
FIGS, TABS, METS, LOGS, CKPT = (
    OUT / "figures", OUT / "tables", OUT / "metrics", OUT / "logs", OUT / "checkpoints",
)
for d in (FIGS, TABS, METS, LOGS, CKPT):
    d.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 140


# --------------------------------------------------------------------------
def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                y_prob: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def save_cm(y_true, y_pred, title: str, path: Path, cmap: str = "Blues"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax,
                cbar=False)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_per_class_report(y_true, y_pred, path: Path):
    rep = classification_report(
        y_true, y_pred, target_names=["FAKE", "REAL"], output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(rep).T.to_csv(path, index=True)


# --------------------------------------------------------------------------
def main():
    # cached logits from train_ensemble
    cache = torch.load(CKPT / "logits_cache.pt", map_location="cpu", weights_only=False)
    t_te = cache["transformer_logits"]["test"].numpy()
    g_te = cache["gnn_logits"]["test"].numpy()
    y_te = cache["labels"]["test"].numpy()

    # Load trained ensemble for alpha
    ens = EnsembleClassifier()
    ens.load_state_dict(torch.load(CKPT / "ensemble_best.pt", map_location="cpu"))
    alpha = float(ens.alpha.item())
    print(f"[eval] loaded alpha = {alpha:.4f}")

    # probabilities
    def softmax_np(x):
        x = x - x.max(axis=1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=1, keepdims=True)

    pT = softmax_np(t_te)
    pG = softmax_np(g_te)
    pE = alpha * pT + (1 - alpha) * pG

    predictions = {
        "BERT": pT.argmax(-1),
        "GCN": pG.argmax(-1),
        "GETE (ours)": pE.argmax(-1),
    }
    probs_pos = {  # probability of REAL class (label 1) for ROC/PR
        "BERT": pT[:, 1],
        "GCN": pG[:, 1],
        "GETE (ours)": pE[:, 1],
    }

    # --- summary metrics table ---------------------------------------------
    rows = []
    for name in ["BERT", "GCN", "GETE (ours)"]:
        rows.append(metrics_row(name, y_te, predictions[name], probs_pos[name]))
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(TABS / "metrics_summary.csv", index=False)
    print("[eval] metrics table:")
    print(df_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- per-class classification reports ---------------------------------
    for name, pred in predictions.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_per_class_report(y_te, pred, TABS / f"per_class_report_{safe}.csv")

    # --- confusion matrices ------------------------------------------------
    for name, pred in predictions.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_cm(y_te, pred, f"Confusion Matrix — {name}",
                FIGS / f"confusion_matrix_{safe}.png")
    # paper-style proposed CM
    save_cm(y_te, predictions["GETE (ours)"], "Confusion Matrix - Proposed Model",
            FIGS / "confusion_matrix_proposed.png", cmap="Blues")

    # --- ROC curve overlay -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs_pos.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        auc = roc_auc_score(y_te, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on LIAR Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "roc_curves.png")
    plt.close(fig)

    # --- PR curve overlay --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs_pos.items():
        prec, rec, _ = precision_recall_curve(y_te, p)
        ap = average_precision_score(y_te, p)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "pr_curves.png")
    plt.close(fig)

    # --- comparison bar chart ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.2
    x = np.arange(len(rows))
    ax.bar(x - 1.5 * width, df_metrics["accuracy"], width, label="Accuracy")
    ax.bar(x - 0.5 * width, df_metrics["precision"], width, label="Precision")
    ax.bar(x + 0.5 * width, df_metrics["recall"], width, label="Recall")
    ax.bar(x + 1.5 * width, df_metrics["f1"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics["model"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison (LIAR test set)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.20))
    for xi, acc in zip(x, df_metrics["accuracy"]):
        ax.text(xi - 1.5 * width, acc + 0.01, f"{acc:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "model_comparison_bar.png", bbox_inches="tight")
    plt.close(fig)

    # --- training curves: transformer --------------------------------------
    with open(LOGS / "transformer_train.json") as f:
        tr_log = json.load(f)
    th = tr_log["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(th["train_loss"]) + 1)
    axes[0].plot(epochs, th["train_loss"], "o-", label="train")
    axes[0].plot(epochs, th["valid_loss"], "s-", label="valid")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Transformer loss"); axes[0].legend()
    axes[1].plot(epochs, th["train_acc"], "o-", label="train")
    axes[1].plot(epochs, th["valid_acc"], "s-", label="valid")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Transformer accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "training_curves_transformer.png")
    plt.close(fig)

    # --- training curves: gnn ---------------------------------------------
    with open(LOGS / "gnn_train.json") as f:
        gnn_log = json.load(f)
    gh = gnn_log["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(gh["train_loss"]) + 1)
    axes[0].plot(epochs, gh["train_loss"])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("GCN loss")
    axes[1].plot(epochs, gh["train_acc"], label="train")
    axes[1].plot(epochs, gh["valid_acc"], label="valid")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("GCN accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "training_curves_gnn.png")
    plt.close(fig)

    # --- alpha progression -------------------------------------------------
    with open(LOGS / "ensemble_train.json") as f:
        ens_log = json.load(f)
    alphas = ens_log["history"]["alpha"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(alphas) + 1), alphas, "o-", color="tab:purple")
    ax.axhline(ens_log["final_alpha"], ls="--", color="gray", alpha=0.6,
               label=f"final α={ens_log['final_alpha']:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight for Transformer (α)")
    ax.set_title("Ensemble Weight Progression Over Epochs")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "alpha_progression.png")
    plt.close(fig)

    # --- PCA of article embeddings ----------------------------------------
    graph = torch.load(CKPT / "graph_data.pt", map_location="cpu", weights_only=False)
    feats = graph["features"].numpy()
    labels = graph["labels"].numpy()
    test_mask = graph["test_mask"].numpy()
    feats_te = feats[test_mask]
    labels_te = labels[test_mask]
    pca = PCA(n_components=2)
    proj = pca.fit_transform(feats_te)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, color, marker in [(0, "tab:red", "x"), (1, "tab:blue", "o")]:
        sel = labels_te == cls
        ax.scatter(proj[sel, 0], proj[sel, 1], c=color, s=10, alpha=0.6,
                   label=f"Class {cls} ({'FAKE' if cls == 0 else 'REAL'})",
                   marker=marker)
    ax.set_xlabel("PCA Dim 1"); ax.set_ylabel("PCA Dim 2")
    ax.set_title("PCA Visualization of Article Embeddings (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "pca_embeddings.png")
    plt.close(fig)

    # --- dump JSON of every metric ----------------------------------------
    all_metrics = {
        "alpha": alpha,
        "per_model": {r["model"]: {k: v for k, v in r.items() if k != "model"}
                      for r in rows},
        "test_size": int(len(y_te)),
        "class_balance": {"FAKE": int((y_te == 0).sum()),
                          "REAL": int((y_te == 1).sum())},
    }
    with open(METS / "metrics_all.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[eval] all figures written to {FIGS}")
    print(f"[eval] tables written to {TABS}")
    print(f"[eval] metrics written to {METS}")


if __name__ == "__main__":
    main()
