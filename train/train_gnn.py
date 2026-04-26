"""Train the GCN/GraphSAGE over an article-relational graph.

Workflow
--------
1. Load the best-trained BERT, freeze it, encode every LIAR statement to a
   768-d CLS embedding -> node features x.
2. Build the article graph (similarity + speaker/subject/party edges) via
   utils.graph_builder.build_article_graph.
3. Train a 2-layer GCN on GPU with a train/val/test mask derived from the
   LIAR split. Masks cover all articles (train/val/test), labels come from
   the binarized LIAR targets.
4. Save checkpoint + features + edge_index for the ensemble step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn_model import GNNClassifier
from models.transformer_model import TransformerClassifier
from utils.graph_builder import build_article_graph
from utils.preprocess import LIARDataset


def encode_dataset(model, dataset, device, batch_size: int = 32) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size)
    feats = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cls = model.encode(ids, mask).cpu()
            feats.append(cls)
    return torch.cat(feats, dim=0)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--sim_topk", type=int, default=8)
    parser.add_argument("--sim_threshold", type=float, default=0.85)
    parser.add_argument("--conv", type=str, default="gcn",
                        choices=["gcn", "sage", "gat"])
    parser.add_argument("--tag", type=str, default=None,
                        help="checkpoint/log suffix; default = --conv value "
                             "('gcn' also aliased to legacy 'gnn')")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tag = args.tag or ("gnn" if args.conv == "gcn" else args.conv)  # 'gcn','sage','gat'
    args.tag = tag

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{tag}] device={device}  conv={args.conv}")

    os.makedirs(f"{args.out_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.out_dir}/logs", exist_ok=True)

    graph_path = f"{args.out_dir}/checkpoints/graph_data.pt"
    if os.path.exists(graph_path):
        print(f"[{tag}] loading cached graph data from {graph_path}")
        cached = torch.load(graph_path, map_location="cpu", weights_only=False)
        features   = cached["features"]
        edge_index = cached["edge_index"]
        labels     = cached["labels"]
        train_mask = cached["train_mask"]
        valid_mask = cached["valid_mask"]
        test_mask  = cached["test_mask"]
        n_tr, n_va, n_te = cached["sizes"]
        total = n_tr + n_va + n_te
        print(f"[{tag}] cached: nodes={total} edges={edge_index.shape[1]} feats={tuple(features.shape)}")
    else:
        print(f"[{tag}] loading LIAR splits...")
        train_ds = LIARDataset(f"{args.data_dir}/train.tsv")
        valid_ds = LIARDataset(f"{args.data_dir}/valid.tsv")
        test_ds  = LIARDataset(f"{args.data_dir}/test.tsv")
        n_tr, n_va, n_te = len(train_ds), len(valid_ds), len(test_ds)
        total = n_tr + n_va + n_te
        print(f"[{tag}] sizes train={n_tr} valid={n_va} test={n_te}  total={total}")

        best_bert = f"{args.out_dir}/checkpoints/transformer_best.pt"
        if not os.path.exists(best_bert):
            raise SystemExit(f"missing {best_bert}. train transformer first.")
        bert = TransformerClassifier().to(device)
        bert.load_state_dict(torch.load(best_bert, map_location=device))

        print(f"[{tag}] encoding articles to CLS embeddings...")
        t0 = time.time()
        f_tr = encode_dataset(bert, train_ds, device, args.batch_size)
        f_va = encode_dataset(bert, valid_ds, device, args.batch_size)
        f_te = encode_dataset(bert, test_ds, device, args.batch_size)
        print(f"[{tag}] encoded in {time.time()-t0:.1f}s")
        del bert
        torch.cuda.empty_cache()

        features = torch.cat([f_tr, f_va, f_te], dim=0)
        df_all = pd.concat(
            [train_ds.df, valid_ds.df, test_ds.df], ignore_index=True,
        )
        labels = torch.tensor(df_all["binary_label"].tolist(), dtype=torch.long)

        train_mask = torch.zeros(total, dtype=torch.bool)
        valid_mask = torch.zeros(total, dtype=torch.bool)
        test_mask  = torch.zeros(total, dtype=torch.bool)
        train_mask[:n_tr] = True
        valid_mask[n_tr:n_tr + n_va] = True
        test_mask[n_tr + n_va:]      = True

        print(f"[{tag}] building article graph...")
        t0 = time.time()
        edge_index = build_article_graph(
            df_all, features,
            sim_topk=args.sim_topk,
            sim_threshold=args.sim_threshold,
        )
        print(f"[{tag}] built graph with {edge_index.shape[1]} edges in {time.time()-t0:.1f}s")

    # move to device
    x = features.to(device)
    edge_index_d = edge_index.to(device)
    y = labels.to(device)
    train_mask_d = train_mask.to(device)
    valid_mask_d = valid_mask.to(device)
    test_mask_d = test_mask.to(device)

    # --- 4. train ------------------------------------------------------------
    model = GNNClassifier(in_channels=features.size(1), hidden_channels=args.hidden,
                          conv=args.conv).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)

    history = {"train_loss": [], "train_acc": [], "valid_acc": [], "epoch_seconds": []}
    best_valid = 0.0
    best_path = f"{args.out_dir}/checkpoints/{tag}_best.pt"

    start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        optim.zero_grad()
        logits = model(x, edge_index_d)
        loss = F.cross_entropy(logits[train_mask_d], y[train_mask_d])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index_d)
            train_acc = (logits[train_mask_d].argmax(-1) == y[train_mask_d]).float().mean().item()
            valid_acc = (logits[valid_mask_d].argmax(-1) == y[valid_mask_d]).float().mean().item()
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)
        history["epoch_seconds"].append(time.time() - t0)

        if valid_acc > best_valid:
            best_valid = valid_acc
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"[{tag}] epoch {epoch+1:03d} loss={loss.item():.4f} "
                  f"train_acc={train_acc:.4f} valid_acc={valid_acc:.4f}")

    # final test metrics with best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    with torch.no_grad():
        logits_full = model(x, edge_index_d)
        test_acc = (logits_full[test_mask_d].argmax(-1) == y[test_mask_d]).float().mean().item()
    total_train_sec = time.time() - start
    print(f"[{tag}] best_valid_acc={best_valid:.4f} test_acc={test_acc:.4f} "
          f"total_time={total_train_sec:.1f}s")

    # --- save artifacts (only if not loaded from cache) ---------------------
    if not os.path.exists(graph_path):
        torch.save(
            {
                "features": features.cpu(),
                "edge_index": edge_index.cpu(),
                "labels": labels.cpu(),
                "train_mask": train_mask.cpu(),
                "valid_mask": valid_mask.cpu(),
                "test_mask": test_mask.cpu(),
                "sizes": (n_tr, n_va, n_te),
            },
            graph_path,
        )

    summary = {
        "model": f"{args.conv.upper()} (2 layers, hidden={args.hidden})",
        "n_nodes": total, "n_edges": int(edge_index.shape[1]),
        "best_valid_acc": best_valid, "test_acc": test_acc,
        "history": history,
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
        "total_training_seconds": total_train_sec,
    }
    log_path = f"{args.out_dir}/logs/{tag}_train.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] wrote {log_path}")


if __name__ == "__main__":
    main()
