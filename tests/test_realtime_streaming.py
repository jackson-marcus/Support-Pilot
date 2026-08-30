"""Staged triage: fast stages first, and the expensive one only when it is warranted.

The point of streaming this pass is not decoration. Classification and the
priority scorecard are local milliseconds; the draft is a call out to a model.
So the tests care about two things: that the early stages genuinely reach the
caller before the draft is requested, and that the draft is skipped when the
routing policy says a human should take the ticket.
"""

from __future__ import annotations

import json

import pytest

from supportpilot.gateway.sse_handler import SseHandler
from supportpilot.gateway.ws_handler import TicketRejectedError, WebSocketHandler
from supportpilot.llm.base import FakeProvider
from supportpilot.pubsub.broker import CHANNEL, PubSubBroker
from supportpilot.pubsub.subscriber import STAGES, TriageSubscriber
from supportpilot.triage.routing import route

OUTAGE = {
    "text": "Production is down, the dashboard crashes with error 500 when saving",
    "plan": "enterprise",
    "sentiment": -0.7,
}


def test_route_never_auto_replies_to_the_configured_categories(auto_reply_policy):
    auto_reply_policy(min_confidence=0.0, never_categories=["billing"])
    decision = route({"category": "billing", "confidence": 0.99}, "P1")

    assert decision.escalated
    assert "billing" in decision.reason


def test_route_escalates_when_the_category_is_a_guess(auto_reply_policy):
    auto_reply_policy(min_confidence=0.80, never_categories=[])
    unsure = route({"category": "bug", "confidence": 0.41}, "P2")
    sure = route({"category": "bug", "confidence": 0.95}, "P2")

    assert unsure.escalated
    assert "0.41" in unsure.reason and "KB article" in unsure.reason
    assert sure.draft


def test_the_llm_is_not_called_before_the_first_stage_is_delivered(
    indexed, trained_pipeline, auto_reply_policy
):
    """Progressive delivery has to be real, not a finished pass replayed."""
    auto_reply_policy(min_confidence=0.0, never_categories=[])
    provider = FakeProvider()
    events = SseHandler(trained_pipeline, provider).stream(OUTAGE)

    first = next(events)
    assert first["event"] == "classify"
    assert provider.calls == [], "the model must not have been asked yet"

    assert next(events)["event"] == "priority"
    assert provider.calls == []

    remaining = list(events)
    assert provider.calls, "the draft stage does eventually call the model"
    assert remaining[-1]["event"] == "complete"


def test_every_stage_is_emitted_in_order_with_a_timing(
    indexed, trained_pipeline, auto_reply_policy
):
    auto_reply_policy(min_confidence=0.0, never_categories=[])
    events = list(SseHandler(trained_pipeline, FakeProvider()).stream(OUTAGE))

    assert [e["event"] for e in events] == [*STAGES, "complete"]
    for event in events[:-1]:
        assert event["data"]["elapsed_ms"] >= 0.0
        assert event["data"]["stage"] == event["event"]


def test_an_escalated_ticket_never_reaches_the_model(
    indexed, trained_pipeline, auto_reply_policy
):
    auto_reply_policy(min_confidence=0.999, never_categories=[])
    provider = FakeProvider()
    outcome = WebSocketHandler(trained_pipeline, provider).handle_frame(OUTAGE)

    assert outcome.routing.escalated
    assert not outcome.llm_called
    assert outcome.reply is None
    assert provider.calls == [], "the expensive stage was skipped, not just hidden"
    assert outcome.stage("draft_reply").skipped


def test_an_eligible_ticket_gets_a_draft(indexed, trained_pipeline, auto_reply_policy):
    auto_reply_policy(min_confidence=0.0, never_categories=[])
    provider = FakeProvider(canned="Sorry about the outage - try restarting the export.")
    outcome = WebSocketHandler(trained_pipeline, provider).handle_frame(OUTAGE)

    assert outcome.routing.draft
    assert outcome.llm_called
    assert outcome.reply["draft"] == "Sorry about the outage - try restarting the export."
    assert outcome.reply["provider"] == "fake"
    assert len(provider.calls) == 1


def test_the_earlier_stages_still_run_when_the_draft_is_skipped(
    indexed, trained_pipeline, auto_reply_policy
):
    auto_reply_policy(min_confidence=0.999, never_categories=[])
    outcome = WebSocketHandler(trained_pipeline, FakeProvider()).handle_frame(OUTAGE)

    assert outcome.classification["category"]
    assert outcome.band in {"P1", "P2", "P3"}
    assert outcome.stage("similar").payload["similar_tickets"], "precedent still helps a human"
    assert outcome.total_ms >= 0.0


def test_broker_delivers_tickets_to_the_subscriber(
    indexed, trained_pipeline, auto_reply_policy
):
    auto_reply_policy(min_confidence=0.0, never_categories=[])
    broker = PubSubBroker()
    subscriber = TriageSubscriber(broker, trained_pipeline, FakeProvider())

    broker.publish(CHANNEL, OUTAGE)
    assert len(subscriber.journal) == 1
    assert subscriber.rejected == []


def test_a_frame_without_text_is_rejected_without_killing_the_subscriber(
    indexed, trained_pipeline
):
    broker = PubSubBroker()
    subscriber = TriageSubscriber(broker, trained_pipeline, FakeProvider())

    broker.publish(CHANNEL, {"plan": "pro"})
    broker.publish(CHANNEL, {"text": "help", "sentiment": "not-a-number"})
    assert len(subscriber.rejected) == 2
    assert subscriber.journal == []

    broker.publish(CHANNEL, OUTAGE)
    assert len(subscriber.journal) == 1, "a good ticket after bad ones must still run"


def test_websocket_raises_on_a_frame_it_cannot_run(indexed, trained_pipeline):
    with pytest.raises(TicketRejectedError):
        WebSocketHandler(trained_pipeline, FakeProvider()).handle_frame({"plan": "pro"})


def test_sse_rejection_is_streamed_not_raised(indexed, trained_pipeline):
    events = list(SseHandler(trained_pipeline, FakeProvider()).stream({"plan": "pro"}))
    assert len(events) == 1 and events[0]["event"] == "rejected"


def test_sse_wire_format_is_parseable(indexed, trained_pipeline, auto_reply_policy):
    auto_reply_policy(min_confidence=0.0, never_categories=[])
    frames = list(SseHandler(trained_pipeline, FakeProvider()).as_wire_format(OUTAGE))

    assert frames[0].startswith("event: classify\ndata: {")
    assert frames[0].endswith("\n\n")
    payload = json.loads(frames[0].split("data: ", 1)[1].strip())
    assert payload["classification"]["category"]


def test_triage_stream_endpoint_emits_every_stage(stream_client, auto_reply_policy):
    auto_reply_policy(min_confidence=0.0, never_categories=[])
    response = stream_client.post("/triage/stream", json={**OUTAGE, "provider": "fake"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    names = [line[7:] for line in response.text.splitlines() if line.startswith("event: ")]
    assert names == [*STAGES, "complete"]


def test_triage_staged_endpoint_reports_timings_and_routing(
    stream_client, auto_reply_policy
):
    auto_reply_policy(min_confidence=0.0, never_categories=[])
    body = stream_client.post("/triage/staged", json={**OUTAGE, "provider": "fake"}).json()

    assert [s["stage"] for s in body["stages"]] == list(STAGES)
    assert body["routing"]["draft"] is True
    assert body["total_ms"] >= 0.0
    assert body["stages"][-1]["reply"]["draft"]


def test_triage_staged_endpoint_shows_the_escalation(stream_client, auto_reply_policy):
    auto_reply_policy(min_confidence=0.999, never_categories=[])
    body = stream_client.post("/triage/staged", json={**OUTAGE, "provider": "fake"}).json()

    assert body["routing"]["escalated"] is True
    assert body["stages"][-1]["skipped"] is True
    assert body["stages"][-1]["reply"] is None
    assert "below" in body["routing"]["reason"]
