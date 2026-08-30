"""API routes: /triage, /triage/stream, /triage/staged, /similar, /health."""

from __future__ import annotations

import functools
import logging
import pickle

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from supportpilot.drafts.reply import draft_reply
from supportpilot.gateway.sse_handler import SseHandler
from supportpilot.gateway.ws_handler import TicketRejectedError
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


def _stream_handler(ticket: Ticket) -> SseHandler:
    """A handler per request: the subscriber journals this ticket's stages."""
    return SseHandler(_pipeline(), get_provider(ticket.provider))


@router.post("/triage/stream")
def triage_stream(ticket: Ticket) -> StreamingResponse:
    """Run the pass as a server-sent event stream, one event per stage.

    The classification lands in the agent's hands while retrieval and the draft
    are still running - and if the routing policy escalates, the draft never
    runs at all.
    """
    try:
        handler = _stream_handler(ticket)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StreamingResponse(
        handler.as_wire_format(ticket.model_dump()),
        media_type="text/event-stream",
    )


@router.post("/triage/staged")
def triage_staged(ticket: Ticket) -> dict:
    """The same staged pass returned as one JSON body, with per-stage timings."""
    try:
        handler = _stream_handler(ticket)
        return handler.run(ticket.model_dump()).as_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TicketRejectedError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
