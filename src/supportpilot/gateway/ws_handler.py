"""WebSocket-shaped handler: one ticket frame in, one staged outcome out."""

from __future__ import annotations

from typing import Any

from supportpilot.llm.base import LLMProvider
from supportpilot.pubsub.broker import CHANNEL, PubSubBroker
from supportpilot.pubsub.subscriber import TriageOutcome, TriageSubscriber


class TicketRejectedError(ValueError):
    """The frame did not describe a ticket the pass could run on."""


class WebSocketHandler:
    """Publishes a ticket to the broker and returns the outcome it produced."""

    def __init__(
        self,
        pipeline: Any,
        provider: LLMProvider,
        broker: PubSubBroker | None = None,
    ) -> None:
        self.broker = broker or PubSubBroker()
        self.subscriber = TriageSubscriber(self.broker, pipeline, provider)

    def handle_frame(self, frame: dict[str, Any]) -> TriageOutcome:
        before = len(self.subscriber.journal)
        self.broker.publish(CHANNEL, frame)
        if len(self.subscriber.journal) == before:
            rejected = self.subscriber.rejected
            raise TicketRejectedError(rejected[-1]["error"] if rejected else "unknown")
        return self.subscriber.journal[-1]
