"""ASGI-Style Middleware Architecture - Ticket Context.

Mutable context object flowing into and out of the onion middleware stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TicketContext:
    """Carries state across all middleware layers during ticket triage and drafting."""

    # Raw input
    text: str
    plan: str = "basic"
    customer_id: str = ""

    # Enriched features & intermediate states
    sanitized_text: str = ""
    sentiment: float = 0.0
    category: str = ""
    category_confidence: float = 0.0
    priority_score: float = 0.0
    priority_band: str = "P3"
    retrieved_kb: list[dict[str, Any]] = field(default_factory=list)
    draft_reply: str = ""

    # Execution trace recording layer timing and transformations
    trace: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
