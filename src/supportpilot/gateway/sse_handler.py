"""SSE-shaped handler: emit each triage stage the moment it finishes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from supportpilot.gateway.ws_handler import WebSocketHandler
from supportpilot.llm.base import LLMProvider
from supportpilot.pubsub.broker import PubSubBroker
from supportpilot.pubsub.subscriber import StageResult, TriageOutcome


class SseHandler:
    """Streams the pass so an agent reads the category before the draft exists."""

    def __init__(
        self,
        pipeline: Any,
        provider: LLMProvider,
        broker: PubSubBroker | None = None,
    ) -> None:
        self.ws = WebSocketHandler(pipeline, provider, broker=broker)

    def stream(self, frame: dict[str, Any]) -> Iterator[dict[str, Any]]:
        stages: list[StageResult] = []
        try:
            for result in self.ws.subscriber.iter_stages(frame):
                stages.append(result)
                yield {"event": result.name, "data": result.as_dict()}
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            yield {"event": "rejected", "data": {"reason": str(exc)}}
            return
        yield {"event": "complete", "data": TriageOutcome.from_stages(stages).as_dict()}

    def as_wire_format(self, frame: dict[str, Any]) -> Iterator[str]:
        for item in self.stream(frame):
            payload = json.dumps(item["data"], default=str)
            yield f"event: {item['event']}\ndata: {payload}\n\n"

    def run(self, frame: dict[str, Any]) -> TriageOutcome:
        """The whole pass, for callers that want one answer rather than a stream."""
        return self.ws.handle_frame(frame)
