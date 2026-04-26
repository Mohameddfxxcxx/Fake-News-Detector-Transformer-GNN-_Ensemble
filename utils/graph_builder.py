"""Build a GETE-style relational graph over LIAR articles.

Paper §Graph Module constructs a heterogeneous graph of articles, users, and
sources. LIAR does not contain user-share traces, so we project onto a
single article-node graph and connect two articles when they share any of:
  - same speaker        (user/source proxy)
  - same primary subject (topic)
  - same political party (source/ideology)
  - high textual similarity on BERT-CLS embeddings
This captures the relational signals the paper exploits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def build_article_graph(
    df,
    features: torch.Tensor,
    sim_topk: int = 8,
    sim_threshold: float = 0.85,
    add_meta_edges: bool = True,
) -> torch.Tensor:
    """Return an edge_index tensor of shape [2, E] (undirected, self-loops via PyG).

    - df : pandas DataFrame with columns speaker/subject/party (per article)
    - features : float tensor [N, D] (BERT CLS embeddings)
    """
    n = features.size(0)
    feats = features.detach().cpu().numpy().astype(np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

    src, dst = [], []

    # --- similarity edges (top-k per node, filtered by threshold) -----------
    # chunk to avoid OOM on large N
    chunk = 2048
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = cosine_similarity(feats[start:end], feats)
        # zero-out self
        for local_i, i in enumerate(range(start, end)):
            sims[local_i, i] = -1.0
        top_idx = np.argpartition(-sims, sim_topk, axis=1)[:, :sim_topk]
        for local_i, i in enumerate(range(start, end)):
            for j in top_idx[local_i]:
                if sims[local_i, j] >= sim_threshold:
                    src.append(int(i))
                    dst.append(int(j))

    # --- metadata edges ------------------------------------------------------
    if add_meta_edges:
        for key in ("speaker", "subject", "party"):
            groups = {}
            for idx, val in enumerate(df[key].tolist()):
                primary = str(val).split(",")[0].strip().lower()
                if primary in ("", "none", "unknown", "nan"):
                    continue
                groups.setdefault(primary, []).append(idx)
            for members in groups.values():
                if len(members) < 2:
                    continue
                # link each article to up to 4 others sharing this attribute
                members_arr = np.array(members)
                for idx in members:
                    others = members_arr[members_arr != idx]
                    if len(others) > 4:
                        others = np.random.default_rng(0).choice(others, 4, replace=False)
                    for other in others:
                        src.append(int(idx))
                        dst.append(int(other))

    if not src:  # degenerate: ensure at least self-loops
        src = list(range(n))
        dst = list(range(n))

    # symmetrize
    edge_index = np.stack([src + dst, dst + src], axis=0)
    edge_index = np.unique(edge_index, axis=1)
    return torch.tensor(edge_index, dtype=torch.long)
