"""Running the triage pass one stage at a time.

`/triage` runs the whole pass before it answers, so the agent waits on the
slowest part of it. The stages are wildly uneven: classification and the
priority scorecard are local milliseconds, retrieval is a lexical scan, and the
draft is a call out to a language model. Emitting each stage as it finishes puts
the category and the queue position in front of a human immediately, and lets
the expensive stage be skipped entirely when the routing policy says a person
should handle this ticket.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from supportpilot.drafts.reply import draft_reply
from supportpilot.llm.base import LLMProvider
from supportpilot.pubsub.broker import CHANNEL, PubSubBroker
from supportpilot.retrieval.similar import similar_tickets
from supportpilot.triage.classify import classify, priority_band, priority_score
from supportpilot.triage.routing import RoutingDecision, route

STAGES: tuple[str, ...] = ("classify", "priority", "similar", "draft_reply")


@dataclass(frozen=True, slots=True)
class StageResult:
    """One stage of the pass, with what it produced and what it cost."""

    name: str
    payload: dict[str, Any]
    elapsed_ms: float
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "skipped": self.skipped,
            "elapsed_ms": self.elapsed_ms,
            **self.payload,
        }


@dataclass(frozen=True, slots=True)
class TriageOutcome:
    """Everything one ticket produced, stage by stage."""

    stages: tuple[StageResult, ...]
    routing: RoutingDecision

    @classmethod
    def from_stages(cls, stages: list[StageResult]) -> TriageOutcome:
        routing = stages[1].payload["routing"]
        return cls(tuple(stages), RoutingDecision(routing["draft"], routing["reason"]))

    def stage(self, name: str) -> StageResult:
        for result in self.stages:
            if result.name == name:
                return result
        raise KeyError(f"stage {name!r} did not run")

    @property
    def classification(self) -> dict[str, Any]:
        return self.stage("classify").payload["classification"]

    @property
    def band(self) -> str:
        return str(self.stage("priority").payload["band"])

    @property
    def reply(self) -> dict[str, Any] | None:
        """The drafted reply bundle, or None when the ticket was escalated."""
        return self.stage("draft_reply").payload.get("reply")

    @property
    def llm_called(self) -> bool:
        return not self.stage("draft_reply").skipped

    @property
    def total_ms(self) -> float:
        return round(sum(s.elapsed_ms for s in self.stages), 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "routing": self.routing.as_dict(),
            "total_ms": self.total_ms,
            "stages": [s.as_dict() for s in self.stages],
        }


class TriageSubscriber:
    """Runs the staged triage pass for every ticket published to the channel."""

    def __init__(
        self,
        broker: PubSubBroker,
        pipeline: Any,
        provider: LLMProvider,
        channel: str = CHANNEL,
    ) -> None:
        self.channel = channel
        self.pipeline = pipeline
        self.provider = provider
        self.journal: list[TriageOutcome] = []
        self.rejected: list[dict[str, Any]] = []
        broker.subscribe(channel, self.on_message)

    def on_message(self, message: dict[str, Any]) -> None:
        try:
            outcome = self._run(message)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            self.rejected.append({"frame": dict(message), "error": str(exc)})
            return
        self.journal.append(outcome)

    def _run(self, message: dict[str, Any]) -> TriageOutcome:
        return TriageOutcome.from_stages(list(self.iter_stages(message)))

    def iter_stages(self, message: dict[str, Any]) -> Iterator[StageResult]:
        """Yield each stage the moment it finishes.

        A generator rather than a list so a streaming caller genuinely gets the
        classification before the draft has been requested - replaying a
        finished pass would make the progressive delivery a fiction.
        """
        text = str(message["text"])
        plan = str(message.get("plan", "pro"))
        sentiment = float(message.get("sentiment", 0.0))

        started = time.perf_counter()
        classification = classify(self.pipeline, text)
        yield StageResult("classify", {"classification": classification}, _ms_since(started))

        started = time.perf_counter()
        score = priority_score(text, plan, sentiment)
        band = priority_band(score)
        decision = route(classification, band)
        yield StageResult(
            "priority",
            {"score": score, "band": band, "routing": decision.as_dict()},
            _ms_since(started),
        )

        started = time.perf_counter()
        similar = similar_tickets(text)
        yield StageResult("similar", {"similar_tickets": similar}, _ms_since(started))

        started = time.perf_counter()
        if not decision.draft:
            yield StageResult(
                "draft_reply",
                {"reply": None, "reason": decision.reason},
                _ms_since(started),
                skipped=True,
            )
            return
        reply = draft_reply(text, classification["category"], band, similar, provider=self.provider)
        yield StageResult("draft_reply", {"reply": reply}, _ms_since(started))


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
