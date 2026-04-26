"""Paper-accurate preprocessing for LIAR dataset.

Paper steps (§ Preprocessing):
- Lowercase, remove URLs / punctuation / stopwords
- BERT tokenizer, max_len=128
- 6-class → 2-class binarization (true/half-true/mostly-true → REAL,
  false/pants-fire/barely-true → FAKE)
- Metadata retained for graph construction
"""

from __future__ import annotations

import os
import re
import string
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizerFast


# LIAR columns (14): id, label, statement, subject, speaker, job, state, party,
# bt_counts, f_counts, ht_counts, mt_counts, pf_counts, context
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job", "state",
    "party", "bt_counts", "f_counts", "ht_counts", "mt_counts",
    "pf_counts", "context",
]

LABEL_MAP = {
    "pants-fire": 0, "false": 0, "barely-true": 0,
    "half-true": 1, "mostly-true": 1, "true": 1,
}
LABEL_NAMES = ["FAKE", "REAL"]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "i", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
    "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am",
    "as",
}


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in s.split() if w not in STOPWORDS]
    return MULTISPACE_RE.sub(" ", " ".join(tokens)).strip()


def load_liar_frame(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t", header=None, names=LIAR_COLUMNS, dtype=str)
    df = df[df["label"].isin(LABEL_MAP)].reset_index(drop=True)
    df["binary_label"] = df["label"].map(LABEL_MAP).astype(int)
    df["text_clean"] = df["statement"].fillna("").map(clean_text)
    df["speaker"] = df["speaker"].fillna("unknown")
    df["subject"] = df["subject"].fillna("none")
    df["party"] = df["party"].fillna("none")
    return df


# Shared tokenizer cache (fast) - paper uses BERT tokenizer with max_len=128
_tokenizers: dict = {}


def get_tokenizer(name: str = "bert-base-uncased"):
    if name not in _tokenizers:
        try:
            _tokenizers[name] = AutoTokenizer.from_pretrained(name, use_fast=True)
        except Exception:
            _tokenizers[name] = BertTokenizerFast.from_pretrained(name)
    return _tokenizers[name]


class LIARDataset(Dataset):
    """LIAR dataset with paper preprocessing (max_len=128 by default)."""

    def __init__(self, filepath: str, max_len: int = 128, clean: bool = True,
                 tokenizer_name: str = "bert-base-uncased"):
        df = load_liar_frame(filepath)
        self.df = df
        self.tokenizer_name = tokenizer_name
        self.texts = (df["text_clean"] if clean else df["statement"]).tolist()
        self.labels = df["binary_label"].tolist()
        tok = get_tokenizer(tokenizer_name)
        self.encodings = tok(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def dataset_splits(data_dir: str = "data", max_len: int = 128,
                   tokenizer_name: str = "bert-base-uncased"):
    """Return (train, valid, test) LIARDatasets with paper preprocessing."""
    kw = dict(max_len=max_len, tokenizer_name=tokenizer_name)
    train = LIARDataset(os.path.join(data_dir, "train.tsv"), **kw)
    valid = LIARDataset(os.path.join(data_dir, "valid.tsv"), **kw)
    test = LIARDataset(os.path.join(data_dir, "test.tsv"), **kw)
    return train, valid, test
