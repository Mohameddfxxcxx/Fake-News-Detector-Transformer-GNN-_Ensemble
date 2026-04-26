"""Train dynamic attention fusion of transformer + GCN outputs.

This replaces the static scalar alpha of the GETE ensemble with a per-example
attention weight over (text, graph) branches. Training workflow:

1. Load the best BERT text model and best GCN graph model.
2. Compute per-article (a) BERT CLS embeddings and logits, (b) GCN hidden
   embeddings and logits on all splits.
3. Train the fusion module on the VALIDATION split for a number of epochs,
   minimising cross-entropy against the labels. Two training modes:
     - 'mixed'  : supervise probs_mixed = a[0]*pT + a[1]*pG
     - 'head'   : supervise a separate classifier on the fused embedding
     - 'both'   : average of both losses (default)
4. Save checkpoint + cached features/logits for evaluation.
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

from models.fusion import DynamicFusion
from models.gnn_model import GNNClassifier
from models.transformer_model import TransformerClassifier
from utils.preprocess import LIARDataset


@torch.no_grad()
def bert_encode_and_logits(model, dataset, device, batch_size: int = 32):
    loader = DataLoader(dataset, batch_size=batch_size)
    cls_feats, logits_out = [], []
    model.eval()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits, cls = model(ids, mask, return_embedding=True)
        cls_feats.append(cls.cpu())
        logits_out.append(logits.cpu())
    return torch.cat(cls_feats, 0), torch.cat(logits_out, 0)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["mixed", "head", "both"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--text_tag", type=str, default="transformer")
    parser.add_argument("--graph_tag", type=str, default="gnn")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fusion] device={device} mode={args.mode} text={args.text_tag} graph={args.graph_tag}")

    # ----- text branch -------------------------------------------------------
    text_ckpt = f"{args.out_dir}/checkpoints/{args.text_tag}_best.pt"
    if not os.path.exists(text_ckpt):
        raise SystemExit(f"missing {text_ckpt}")
    # infer HF name from filename
    name_map = {
        "transformer": "bert-base-uncased",
        "distilroberta": "distilroberta-base",
        "roberta": "roberta-base",
    }
    model_name = name_map.get(args.text_tag, "bert-base-uncased")
    train_ds = LIARDataset(f"{args.data_dir}/train.tsv", tokenizer_name=model_name)
    valid_ds = LIARDataset(f"{args.data_dir}/valid.tsv", tokenizer_name=model_name)
    test_ds  = LIARDataset(f"{args.data_dir}/test.tsv",  tokenizer_name=model_name)

    print(f"[fusion] loading text model ({model_name})...")
    text_model = TransformerClassifier(pretrained_name=model_name).to(device)
    text_model.load_state_dict(torch.load(text_ckpt, map_location=device))

    print("[fusion] computing text CLS features + logits...")
    t0 = time.time()
    e_tr_t, l_tr_t = bert_encode_and_logits(text_model, train_ds, device)
    e_va_t, l_va_t = bert_encode_and_logits(text_model, valid_ds, device)
    e_te_t, l_te_t = bert_encode_and_logits(text_model, test_ds, device)
    del text_model
    torch.cuda.empty_cache()
    print(f"[fusion] text done in {time.time()-t0:.1f}s")

    # ----- graph branch ------------------------------------------------------
    graph = torch.load(f"{args.out_dir}/checkpoints/graph_data.pt",
                       map_location="cpu", weights_only=False)
    features = graph["features"]
    edge_index = graph["edge_index"]
    n_tr, n_va, n_te = graph["sizes"]

    # figure out conv type from graph_tag
    conv_map = {"gnn": "gcn", "gcn": "gcn", "sage": "sage", "gat": "gat"}
    conv = conv_map.get(args.graph_tag, "gcn")
    print(f"[fusion] loading graph model ({args.graph_tag}, conv={conv})...")
    graph_model = GNNClassifier(in_channels=features.size(1),
                                hidden_channels=128 if conv != "gat" else 64,
                                conv=conv).to(device)
    g_ckpt = f"{args.out_dir}/checkpoints/{args.graph_tag}_best.pt"
    graph_model.load_state_dict(torch.load(g_ckpt, map_location=device))
    graph_model.eval()

    with torch.no_grad():
        g_logits, g_emb = graph_model(features.to(device), edge_index.to(device),
                                       return_embedding=True)
        g_logits = g_logits.cpu()
        g_emb = g_emb.cpu()

    e_tr_g = g_emb[:n_tr]
    e_va_g = g_emb[n_tr:n_tr + n_va]
    e_te_g = g_emb[n_tr + n_va:]
    l_tr_g = g_logits[:n_tr]
    l_va_g = g_logits[n_tr:n_tr + n_va]
    l_te_g = g_logits[n_tr + n_va:]

    y_tr = torch.tensor(train_ds.labels, dtype=torch.long)
    y_va = torch.tensor(valid_ds.labels, dtype=torch.long)
    y_te = torch.tensor(test_ds.labels, dtype=torch.long)

    # ----- fusion ------------------------------------------------------------
    dim_t = e_tr_t.size(1)
    dim_g = e_tr_g.size(1)
    print(f"[fusion] dim_t={dim_t} dim_g={dim_g}")
    fusion = DynamicFusion(dim_t=dim_t, dim_g=dim_g, hidden=args.hidden).to(device)
    optim = torch.optim.Adam(fusion.parameters(), lr=args.lr, weight_decay=1e-4)

    history = {"valid_loss": [], "valid_acc": [], "alpha_mean": [],
               "alpha_min": [], "alpha_max": []}
    best_valid_acc = 0.0
    best_path = f"{args.out_dir}/checkpoints/fusion_best.pt"

    def step(split, train_mode: bool):
        e_t = (e_tr_t if split == "train" else e_va_t if split == "valid" else e_te_t).to(device)
        e_g = (e_tr_g if split == "train" else e_va_g if split == "valid" else e_te_g).to(device)
        l_t = (l_tr_t if split == "train" else l_va_t if split == "valid" else l_te_t).to(device)
        l_g = (l_tr_g if split == "train" else l_va_g if split == "valid" else l_te_g).to(device)
        y   = (y_tr   if split == "train" else y_va   if split == "valid" else y_te).to(device)
        fusion.train(train_mode)
        out = fusion(e_t, e_g, l_t, l_g)
        losses = []
        if args.mode in ("mixed", "both"):
            losses.append(F.nll_loss(torch.log(out["probs_mixed"] + 1e-12), y))
        if args.mode in ("head", "both"):
            losses.append(F.cross_entropy(out["logits_head"], y))
        loss = sum(losses) / len(losses)

        # predictive accuracy from the probs actually being produced
        if args.mode == "head":
            preds = out["logits_head"].argmax(-1)
        elif args.mode == "mixed":
            preds = out["probs_mixed"].argmax(-1)
        else:
            preds = (0.5 * F.softmax(out["logits_head"], dim=-1)
                     + 0.5 * out["probs_mixed"]).argmax(-1)
        acc = (preds == y).float().mean().item()
        return loss, acc, out

    start = time.time()
    for epoch in range(args.epochs):
        optim.zero_grad()
        loss_tr, acc_tr, _ = step("train", True)
        loss_tr.backward()
        optim.step()

        with torch.no_grad():
            loss_va, acc_va, out_va = step("valid", False)
        a = out_va["attention"].detach().cpu().numpy()
        alpha_t = a[:, 0]
        history["valid_loss"].append(float(loss_va.item()))
        history["valid_acc"].append(acc_va)
        history["alpha_mean"].append(float(alpha_t.mean()))
        history["alpha_min"].append(float(alpha_t.min()))
        history["alpha_max"].append(float(alpha_t.max()))

        if acc_va > best_valid_acc:
            best_valid_acc = acc_va
            torch.save(fusion.state_dict(), best_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[fusion] epoch {epoch+1:03d} "
                  f"loss_tr={loss_tr.item():.4f} acc_tr={acc_tr:.4f} "
                  f"loss_va={loss_va.item():.4f} acc_va={acc_va:.4f} "
                  f"alpha_mean={alpha_t.mean():.3f}")

    # final test with best weights
    fusion.load_state_dict(torch.load(best_path, map_location=device))
    with torch.no_grad():
        loss_te, acc_te, out_te = step("test", False)
    total_sec = time.time() - start
    print(f"[fusion] best_valid={best_valid_acc:.4f} test_acc={acc_te:.4f} "
          f"total_time={total_sec:.1f}s")

    # save logits cache with fusion included
    torch.save(
        {
            "dim_t": dim_t, "dim_g": dim_g,
            "text_tag": args.text_tag, "graph_tag": args.graph_tag, "mode": args.mode,
            "embeddings": {
                "text":  {"train": e_tr_t, "valid": e_va_t, "test": e_te_t},
                "graph": {"train": e_tr_g, "valid": e_va_g, "test": e_te_g},
            },
            "logits": {
                "text":  {"train": l_tr_t, "valid": l_va_t, "test": l_te_t},
                "graph": {"train": l_tr_g, "valid": l_va_g, "test": l_te_g},
            },
            "labels":   {"train": y_tr, "valid": y_va, "test": y_te},
            "sizes":    (n_tr, n_va, n_te),
        },
        f"{args.out_dir}/checkpoints/fusion_cache_{args.text_tag}_{args.graph_tag}.pt",
    )

    summary = {
        "text_tag": args.text_tag, "graph_tag": args.graph_tag,
        "dim_t": dim_t, "dim_g": dim_g, "hidden": args.hidden,
        "mode": args.mode, "best_valid_acc": best_valid_acc,
        "test_acc": acc_te, "total_training_seconds": total_sec,
        "history": history, "epochs": args.epochs, "lr": args.lr,
    }
    log_path = f"{args.out_dir}/logs/fusion_{args.text_tag}_{args.graph_tag}_train.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[fusion] wrote {log_path}")


if __name__ == "__main__":
    main()
