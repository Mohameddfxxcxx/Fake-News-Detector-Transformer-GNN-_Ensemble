"""Meta-learned ensemble training.

Loads the best BERT and GCN, precomputes their logits on every LIAR article,
then fits the scalar alpha on the VALIDATION set by minimizing cross-entropy
on the combined probabilities. Alpha is reported per-epoch so we can produce
the paper's Fig. 9 (ensemble weight progression).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemble import EnsembleClassifier
from models.gnn_model import GNNClassifier
from models.transformer_model import TransformerClassifier
from utils.preprocess import LIARDataset


@torch.no_grad()
def bert_logits(model, dataset, device, batch_size: int = 32) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size)
    out = []
    model.eval()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out.append(model(ids, mask).cpu())
    return torch.cat(out, 0)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ensemble] device={device}")

    # --- transformer logits on all splits -----------------------------------
    print("[ensemble] computing transformer logits...")
    train_ds = LIARDataset(f"{args.data_dir}/train.tsv")
    valid_ds = LIARDataset(f"{args.data_dir}/valid.tsv")
    test_ds = LIARDataset(f"{args.data_dir}/test.tsv")
    bert = TransformerClassifier().to(device)
    bert.load_state_dict(torch.load(f"{args.out_dir}/checkpoints/transformer_best.pt",
                                    map_location=device))
    t_tr = bert_logits(bert, train_ds, device)
    t_va = bert_logits(bert, valid_ds, device)
    t_te = bert_logits(bert, test_ds, device)
    del bert
    torch.cuda.empty_cache()

    # --- gnn logits ---------------------------------------------------------
    graph = torch.load(f"{args.out_dir}/checkpoints/graph_data.pt",
                       map_location=device, weights_only=False)
    gnn = GNNClassifier(in_channels=graph["features"].size(1)).to(device)
    gnn.load_state_dict(torch.load(f"{args.out_dir}/checkpoints/gnn_best.pt",
                                   map_location=device))
    gnn.eval()
    with torch.no_grad():
        g_all = gnn(graph["features"].to(device), graph["edge_index"].to(device)).cpu()
    n_tr, n_va, n_te = graph["sizes"]
    g_tr = g_all[:n_tr]
    g_va = g_all[n_tr:n_tr + n_va]
    g_te = g_all[n_tr + n_va:]

    y_tr = torch.tensor(train_ds.labels, dtype=torch.long)
    y_va = torch.tensor(valid_ds.labels, dtype=torch.long)
    y_te = torch.tensor(test_ds.labels, dtype=torch.long)

    # sanity
    assert g_tr.size(0) == t_tr.size(0) == y_tr.size(0)
    assert g_va.size(0) == t_va.size(0) == y_va.size(0)
    assert g_te.size(0) == t_te.size(0) == y_te.size(0)

    # --- fit alpha ----------------------------------------------------------
    model = EnsembleClassifier().to("cpu")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"alpha": [], "valid_loss": [], "valid_acc": []}
    start = time.time()
    for epoch in range(args.epochs):
        optim.zero_grad()
        probs = model(t_va, g_va)
        loss = F.nll_loss(torch.log(probs + 1e-12), y_va)
        loss.backward()
        optim.step()
        with torch.no_grad():
            preds = probs.argmax(-1)
            acc = (preds == y_va).float().mean().item()
        history["alpha"].append(float(model.alpha.item()))
        history["valid_loss"].append(loss.item())
        history["valid_acc"].append(acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[ensemble] epoch {epoch+1:02d} α={model.alpha.item():.4f} "
                  f"valid_loss={loss.item():.4f} valid_acc={acc:.4f}")

    total_sec = time.time() - start
    with torch.no_grad():
        probs_te = model(t_te, g_te)
        acc_te = (probs_te.argmax(-1) == y_te).float().mean().item()

    # save
    os.makedirs(f"{args.out_dir}/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"{args.out_dir}/checkpoints/ensemble_best.pt")
    torch.save(
        {
            "transformer_logits": {"train": t_tr, "valid": t_va, "test": t_te},
            "gnn_logits": {"train": g_tr, "valid": g_va, "test": g_te},
            "labels": {"train": y_tr, "valid": y_va, "test": y_te},
        },
        f"{args.out_dir}/checkpoints/logits_cache.pt",
    )

    summary = {
        "final_alpha": float(model.alpha.item()),
        "test_acc": acc_te,
        "history": history,
        "total_training_seconds": total_sec,
        "epochs": args.epochs, "lr": args.lr,
    }
    with open(f"{args.out_dir}/logs/ensemble_train.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ensemble] α={model.alpha.item():.4f} test_acc={acc_te:.4f}")


if __name__ == "__main__":
    main()
