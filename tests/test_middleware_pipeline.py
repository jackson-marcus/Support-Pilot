"""Unit tests for the ASGI-Style Onion Middleware Pipeline in SupportPilot."""

from supportpilot.middleware.base import MiddlewareStack
from supportpilot.middleware.context import TicketContext
from supportpilot.middleware.layers import (
    CategoryClassifierMiddleware,
    InputSanitizationMiddleware,
    SentimentScoringMiddleware,
    SLAPriorityMiddleware,
)


def test_middleware_onion_execution_order():
    """Verify that layers execute entering outer->inner and exiting inner->outer."""
    stack = MiddlewareStack()
    stack.use(InputSanitizationMiddleware())
    stack.use(SentimentScoringMiddleware())
    stack.use(SLAPriorityMiddleware())

    def terminal(ctx: TicketContext) -> TicketContext:
        ctx.trace.append("terminal:handler")
        return ctx

    pipeline = stack.build(terminal)
    ctx = TicketContext(text="   Production is completely broken down!   ", plan="enterprise")
    res = pipeline(ctx)

    assert res.sanitized_text == "Production is completely broken down!"
    assert res.priority_band == "P1"
    assert res.sentiment < 0.0

    # Verify strict onion trace ordering
    expected_order = [
        "enter:InputSanitization",
        "enter:SentimentScoring",
        "enter:SLAPriority",
        "terminal:handler",
        "exit:SLAPriority",
        "exit:SentimentScoring",
        "exit:InputSanitization",
    ]
    assert res.trace == expected_order


def test_middleware_short_circuit():
    """Verify that a middleware layer can short-circuit without calling next."""
    stack = MiddlewareStack()

    class AuthGuardMiddleware:
        def __call__(self, ctx: TicketContext, call_next):
            if not ctx.customer_id:
                ctx.metadata["blocked"] = True
                return ctx
            return call_next(ctx)

    stack.use(AuthGuardMiddleware())
    stack.use(InputSanitizationMiddleware())

    pipeline = stack.build(lambda c: c)

    # Missing customer_id -> blocks before InputSanitization
    ctx = TicketContext(text="Hello", customer_id="")
    res = pipeline(ctx)

    assert res.metadata.get("blocked") is True
    assert "enter:InputSanitization" not in res.trace


def test_category_classifier_middleware_heuristics():
    stack = MiddlewareStack()
    stack.use(CategoryClassifierMiddleware())
    pipeline = stack.build(lambda c: c)

    ctx_billing = pipeline(TicketContext(text="I need a refund for the invoice charge"))
    assert ctx_billing.category == "billing"

    ctx_tech = pipeline(TicketContext(text="The database server threw a 500 crash error"))
    assert ctx_tech.category == "technical"
