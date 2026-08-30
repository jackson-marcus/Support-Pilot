"""Who gets an automatic draft, and who gets a human.

Drafting is the only expensive stage in the pass: classification and the
priority scorecard are milliseconds of local computation, the draft is a call
out to a model. So it is worth deciding, deliberately, whether to make it.

The decision is not about cost. A drafted reply is grounded in KB articles
chosen by the predicted category, so a low-confidence classification produces a
confident-sounding reply grounded in the wrong article - worse than no reply at
all. And some categories should never be answered by a machine regardless of how
sure the classifier is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from supportpilot.settings import get_config


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Whether to draft a reply, and the reason - which is shown to the agent."""

    draft: bool
    reason: str

    @property
    def escalated(self) -> bool:
        return not self.draft

    def as_dict(self) -> dict[str, Any]:
        return {"draft": self.draft, "escalated": self.escalated, "reason": self.reason}


def route(classification: dict[str, Any], band: str) -> RoutingDecision:
    """Decide whether this ticket may be answered automatically."""
    policy = get_config()["triage"]["auto_reply"]
    category = str(classification["category"])
    confidence = float(classification["confidence"])

    if category in policy["never_categories"]:
        return RoutingDecision(False, f"category '{category}' is never auto-replied")

    threshold = float(policy["min_confidence"])
    if confidence < threshold:
        return RoutingDecision(
            False,
            f"category confidence {confidence:.2f} is below {threshold:.2f}, "
            "so the reply would cite the wrong KB article",
        )

    return RoutingDecision(True, f"'{category}' at {confidence:.2f} confidence, priority {band}")
