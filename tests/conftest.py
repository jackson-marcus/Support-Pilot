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


@pytest.fixture()
def trained_pipeline(tickets):
    """A classifier fitted on the synthetic corpus - no artifact on disk needed."""
    from supportpilot.triage.classify import build_pipeline

    return build_pipeline().fit(tickets["text"], tickets["category"])


@pytest.fixture()
def auto_reply_policy():
    """Override the auto-reply gate so tests do not depend on model confidence."""
    from supportpilot.settings import get_config

    policy = get_config()["triage"]["auto_reply"]
    original = dict(policy)

    def apply(**changes):
        policy.update(changes)

    yield apply
    policy.clear()
    policy.update(original)


@pytest.fixture()
def stream_client(tickets, indexed, tmp_path, trained_pipeline):
    """TestClient with the triage artifact written where the routes expect it."""
    import pickle

    from fastapi.testclient import TestClient

    import supportpilot.api.routes as routes
    from supportpilot.api.main import create_app
    from supportpilot.settings import get_config

    cfg = get_config()
    original = cfg["data"]["artifacts_dir"]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    cfg["data"]["artifacts_dir"] = str(artifacts)
    with open(artifacts / "triage.pkl", "wb") as handle:
        pickle.dump(trained_pipeline, handle)
    routes._pipeline.cache_clear()
    try:
        yield TestClient(create_app())
    finally:
        cfg["data"]["artifacts_dir"] = original
        routes._pipeline.cache_clear()
