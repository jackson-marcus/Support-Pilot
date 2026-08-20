"""Similar-ticket retrieval: hybrid dense (fastembed) + BM25 with RRF fusion.

Surfacing "we solved this exact thing last month" is the single highest-value
assist for a support agent; each hit carries its resolution category so the
agent sees what worked."""

from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from supportpilot.settings import get_config, resolve_path


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@functools.lru_cache(maxsize=1)
def _index():
    df = pd.read_parquet(resolve_path(get_config()["data"]["processed_dir"]) / "tickets.parquet")
    texts = df["text"].tolist()
    from fastembed import TextEmbedding

    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    dense = np.array([np.asarray(v, dtype=np.float32) for v in model.embed(texts)])
    dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    return df, dense, bm25, model


def invalidate_index() -> None:
    _index.cache_clear()


def similar_tickets(text: str, top_k: int | None = None) -> list[dict]:
    cfg = get_config()["retrieval"]
    top_k = top_k or cfg["top_k"]
    df, dense, bm25, model = _index()

    q = np.asarray(next(iter(model.embed([text]))), dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    dense_rank = np.argsort(-(dense @ q))
    bm25_rank = np.argsort(-np.asarray(bm25.get_scores(_tokenize(text))))

    fused: dict[int, float] = {}
    for rank_list in (dense_rank[: top_k * 3], bm25_rank[: top_k * 3]):
        for rank, idx in enumerate(rank_list):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (cfg["rrf_k"] + rank + 1)
    best = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [
        {
            "ticket_id": int(df.iloc[i]["ticket_id"]),
            "text": df.iloc[i]["text"],
            "category": df.iloc[i]["category"],
            "score": round(fused[i], 5),
        }
        for i in best
    ]
