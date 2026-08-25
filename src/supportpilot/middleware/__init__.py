"""ASGI-Style Onion Middleware Pipeline Package."""

from supportpilot.middleware.base import Middleware, MiddlewareStack, NextHandler
from supportpilot.middleware.context import TicketContext
from supportpilot.middleware.layers import (
    CategoryClassifierMiddleware,
    InputSanitizationMiddleware,
    SentimentScoringMiddleware,
    SLAPriorityMiddleware,
)

__all__ = [
    "CategoryClassifierMiddleware",
    "InputSanitizationMiddleware",
    "Middleware",
    "MiddlewareStack",
    "NextHandler",
    "SLAPriorityMiddleware",
    "SentimentScoringMiddleware",
    "TicketContext",
]
