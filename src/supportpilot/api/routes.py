"""API routes: /triage (the full agent pass), /similar, /health."""

from __future__ import annotations

import functools
import logging
import pickle

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from supportpilot.drafts.reply import draft_reply
from supportpilot.llm.factory import get_provider
from supportpilot.retrieval.similar import similar_tickets
from supportpilot.settings import get_config, get_settings, resolve_path
from supportpilot.triage.classify import classify, priority_band, priority_score

logger = logging.getLogger(__name__)
router = APIRouter()


class Ticket(BaseModel):
    text: str = Field(min_length=10, max_length=4000)
    plan: str = Field(default="pro", pattern="^(basic|pro|enterprise)$")
    sentiment: float = Field(default=0.0, ge=-1, le=1)
    draft: bool = Field(default=False, description="also draft a reply (LLM call)")
    provider: str | None = None


@functools.lru_cache(maxsize=1)
def _pipeline():
    path = resolve_path(get_config()["data"]["artifacts_dir"]) / "triage.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No triage model at {path}; run training first")
    with open(path, "rb") as f:
        return pickle.load(f)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": get_settings().llm_provider}


@router.post("/triage")
def triage(ticket: Ticket) -> dict:
    try:
        pipeline = _pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    classification = classify(pipeline, ticket.text)
    score = priority_score(ticket.text, ticket.plan, ticket.sentiment)
    band = priority_band(score)
    try:
        similar = similar_tickets(ticket.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = {
        "classification": classification,
        "priority": {"score": score, "band": band},
        "similar_tickets": similar,
    }
    if ticket.draft:
        try:
            provider = get_provider(ticket.provider)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["reply"] = draft_reply(
            ticket.text, classification["category"], band, similar, provider=provider
        )
    return result


@router.post("/similar")
def similar(ticket: Ticket) -> list[dict]:
    try:
        return similar_tickets(ticket.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
