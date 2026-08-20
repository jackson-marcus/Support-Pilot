"""Fixtures: synthetic tickets + stub embedder (no model downloads)."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_synthetic import generate


class StubEmbedder:
    def embed(self, texts):
        for text in texts:
            vec = np.zeros(64, dtype=np.float32)
            for token in re.findall(r"[a-z0-9]+", str(text).lower()):
                vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % 64] += 1.0
            yield vec


@pytest.fixture(scope="session")
def tickets():
    return generate(1200, seed=9)


@pytest.fixture()
def indexed(tickets, tmp_path, monkeypatch):
    """Point processed_dir at a temp corpus and stub the embedder."""
    import fastembed

    import supportpilot.retrieval.similar as similar
    from supportpilot.settings import get_config

    cfg = get_config()
    original = cfg["data"]["processed_dir"]
    proc = tmp_path / "processed"
    proc.mkdir()
    tickets.to_parquet(proc / "tickets.parquet", index=False)
    cfg["data"]["processed_dir"] = str(proc)
    monkeypatch.setattr(fastembed, "TextEmbedding", lambda *a, **k: StubEmbedder())
    similar.invalidate_index()
    yield
    cfg["data"]["processed_dir"] = original
    similar.invalidate_index()
