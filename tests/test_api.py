"""API contract tests with fixture-trained model and stub retrieval."""

import pickle

import pytest
from fastapi.testclient import TestClient

import supportpilot.api.routes as routes
from supportpilot.api.main import create_app
from supportpilot.settings import get_config
from supportpilot.triage.classify import build_pipeline


@pytest.fixture()
def client(tickets, indexed, tmp_path):
    cfg = get_config()
    original = cfg["data"]["artifacts_dir"]
    art = tmp_path / "artifacts"
    art.mkdir()
    cfg["data"]["artifacts_dir"] = str(art)
    pipeline = build_pipeline().fit(tickets["text"], tickets["category"])
    with open(art / "triage.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    routes._pipeline.cache_clear()
    yield TestClient(create_app())
    cfg["data"]["artifacts_dir"] = original
    routes._pipeline.cache_clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_triage_full_pass(client):
    r = client.post(
        "/triage",
        json={
            "text": "Production is down, dashboard crashes with error 500 when saving",
            "plan": "enterprise",
            "sentiment": -0.7,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["classification"]["category"] in ("bug", "performance")
    assert body["priority"]["band"] == "P1"
    assert body["similar_tickets"]


def test_triage_with_fake_draft(client):
    r = client.post(
        "/triage",
        json={
            "text": "I cannot log in and password reset email never arrives, please help",
            "plan": "basic",
            "sentiment": -0.2,
            "draft": True,
            "provider": "fake",
        },
    )
    assert r.status_code == 200
    assert "reply" in r.json()
    assert r.json()["reply"]["provider"] == "fake"


def test_triage_validates_text_length(client):
    assert client.post("/triage", json={"text": "short"}).status_code == 422


def test_similar_endpoint(client):
    r = client.post("/similar", json={"text": "invoice shows wrong amount, charged twice"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)
