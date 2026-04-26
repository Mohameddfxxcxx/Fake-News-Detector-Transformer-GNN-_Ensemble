"""Train the BERT text classifier on the full LIAR dataset.

Matches paper hyperparameters: Adam/AdamW, lr=2e-5, batch=16 (reduced to 8
here to fit on the 4 GB GTX 1650 Ti), max_len=128, dropout=0.3, 3 epochs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import TransformerClassifier
from utils.preprocess import LIARDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_loop(model, loader, device) -> tuple[float, float]:
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(ids, mask)
                loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
    return correct / total, total_loss / total


def main():
    # make stdout handle non-ASCII on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="HF model id, e.g. distilroberta-base, roberta-base")
    parser.add_argument("--tag", type=str, default="transformer",
                        help="checkpoint/log prefix (default 'transformer' for "
                             "backward compat; use e.g. 'distilroberta').")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing best checkpoint")
    parser.add_argument("--start_epoch", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.tag}] device={device}  seed={args.seed}")

    os.makedirs(f"{args.out_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.out_dir}/logs", exist_ok=True)

    print(f"[{args.tag}] model={args.model_name}")
    print(f"[{args.tag}] loading datasets...")
    t0 = time.time()
    kw = dict(max_len=args.max_len, tokenizer_name=args.model_name)
    train_ds = LIARDataset(f"{args.data_dir}/train.tsv", **kw)
    valid_ds = LIARDataset(f"{args.data_dir}/valid.tsv", **kw)
    test_ds = LIARDataset(f"{args.data_dir}/test.tsv", **kw)
    print(f"[{args.tag}] train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} "
          f"(loaded in {time.time()-t0:.1f}s)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, num_workers=0)

    model = TransformerClassifier(pretrained_name=args.model_name).to(device)
    optim = AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": [], "epoch_seconds": []}
    best_valid_acc = 0.0
    best_path = f"{args.out_dir}/checkpoints/{args.tag}_best.pt"
    if args.resume and os.path.exists(best_path):
        print(f"[{args.tag}] resuming from {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))
        best_valid_acc, _ = eval_loop(model, valid_loader, device)
        print(f"[{args.tag}] resumed checkpoint valid_acc={best_valid_acc:.4f}")

    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_t0 = time.time()
        total, correct, total_loss = 0, 0, 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(ids, mask)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)

        train_acc, train_loss = correct / total, total_loss / total
        valid_acc, valid_loss = eval_loop(model, valid_loader, device)
        epoch_sec = time.time() - epoch_t0
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["valid_acc"].append(valid_acc)
        history["valid_loss"].append(valid_loss)
        history["epoch_seconds"].append(epoch_sec)

        print(f"[{args.tag}] epoch {epoch+1} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} ({epoch_sec:.1f}s)")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), best_path)
            print(f"[{args.tag}] saved best -> {best_path} (valid_acc={valid_acc:.4f})")

    total_train_sec = time.time() - start
    # final eval using best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, test_loss = eval_loop(model, test_loader, device)
    print(f"[{args.tag}] TEST acc={test_acc:.4f} loss={test_loss:.4f}")

    summary = {
        "model": args.model_name,
        "tag": args.tag,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_len": args.max_len,
        "best_valid_acc": best_valid_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "total_training_seconds": total_train_sec,
        "history": history,
    }
    log_path = f"{args.out_dir}/logs/{args.tag}_train.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.tag}] wrote {log_path}")


if __name__ == "__main__":
    main()
