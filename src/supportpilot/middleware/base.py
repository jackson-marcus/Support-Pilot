"""ASGI-Style Middleware Architecture - Base Protocol and Stack Builder.

Implements the classic onion middleware model:
Request flows in through outer layers to inner terminal handler,
and response flows back out through outer layers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from supportpilot.middleware.context import TicketContext

NextHandler = Callable[[TicketContext], TicketContext]


class Middleware(Protocol):
    """Protocol for an onion middleware layer."""

    def __call__(self, ctx: TicketContext, call_next: NextHandler) -> TicketContext: ...


class MiddlewareStack:
    """Composes a sequence of Middleware layers around a terminal handler."""

    def __init__(self) -> None:
        self._middlewares: list[Middleware] = []

    def use(self, middleware: Middleware) -> MiddlewareStack:
        """Add a middleware layer to the outer stack."""
        self._middlewares.append(middleware)
        return self

    def build(self, terminal_handler: NextHandler) -> NextHandler:
        """Wrap the terminal handler in all registered middleware layers (onion model)."""
        handler = terminal_handler
        for middleware in reversed(self._middlewares):
            prev_handler = handler

            def make_layer(mw=middleware, nxt=prev_handler):
                return lambda ctx: mw(ctx, nxt)

            handler = make_layer()
        return handler
