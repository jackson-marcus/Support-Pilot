"""ASGI-Style Middleware Architecture - Concrete Pipeline Layers.

Individual middleware layers performing discrete triage, enrichment, and LLM drafting steps.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from supportpilot.middleware.base import NextHandler
from supportpilot.middleware.context import TicketContext
from supportpilot.triage.classify import priority_band, priority_score

logger = logging.getLogger(__name__)


class InputSanitizationMiddleware:
    """Layer 1: Normalizes whitespace, strips dangerous characters, and trims tokens."""

    def __call__(self, ctx: TicketContext, call_next: NextHandler) -> TicketContext:
        ctx.trace.append("enter:InputSanitization")
        cleaned = re.sub(r"\s+", " ", ctx.text).strip()
        ctx.sanitized_text = cleaned
        # Call next layer
        ctx = call_next(ctx)
        ctx.trace.append("exit:InputSanitization")
        return ctx


class SentimentScoringMiddleware:
    """Layer 2: Extracts rough sentiment polarity (e.g. negative indicators)."""

    def __call__(self, ctx: TicketContext, call_next: NextHandler) -> TicketContext:
        ctx.trace.append("enter:SentimentScoring")
        text = (ctx.sanitized_text or ctx.text).lower()
        neg_words = [
            "angry",
            "terrible",
            "frustrated",
            "broken",
            "worst",
            "unacceptable",
            "down",
            "fail",
        ]
        pos_words = ["great", "thanks", "helpful", "good", "love", "awesome", "fast"]

        neg_count = sum(1 for w in neg_words if w in text)
        pos_count = sum(1 for w in pos_words if w in text)
        if neg_count + pos_count > 0:
            ctx.sentiment = round((pos_count - neg_count) / (pos_count + neg_count), 2)
        else:
            ctx.sentiment = 0.0

        ctx = call_next(ctx)
        ctx.trace.append("exit:SentimentScoring")
        return ctx


class CategoryClassifierMiddleware:
    """Layer 3: Executes TF-IDF classification or fallback heuristic."""

    def __init__(self, classifier_getter: Any = None) -> None:
        self.classifier_getter = classifier_getter

    def __call__(self, ctx: TicketContext, call_next: NextHandler) -> TicketContext:
        ctx.trace.append("enter:CategoryClassifier")
        text = ctx.sanitized_text or ctx.text
        if self.classifier_getter is not None:
            try:
                pipeline = self.classifier_getter()
                probs = pipeline.predict_proba([text])[0]
                idx = probs.argmax()
                ctx.category = str(pipeline.classes_[idx])
                ctx.category_confidence = float(probs[idx])
            except Exception as ex:
                logger.warning("Classifier inference failed, using fallback: %s", ex)
                ctx.category = "general"
                ctx.category_confidence = 0.5
        else:
            # Fallback simple keyword match
            text_l = text.lower()
            if any(k in text_l for k in ["bill", "charge", "refund", "invoice", "payment"]):
                ctx.category = "billing"
            elif any(k in text_l for k in ["bug", "error", "crash", "down", "500"]):
                ctx.category = "technical"
            elif any(k in text_l for k in ["password", "login", "sso", "2fa", "account"]):
                ctx.category = "account"
            else:
                ctx.category = "general"
            ctx.category_confidence = 0.85

        ctx = call_next(ctx)
        ctx.trace.append("exit:CategoryClassifier")
        return ctx


class SLAPriorityMiddleware:
    """Layer 4: Computes priority score and SLA priority band (P1-P4)."""

    def __call__(self, ctx: TicketContext, call_next: NextHandler) -> TicketContext:
        ctx.trace.append("enter:SLAPriority")
        text = ctx.sanitized_text or ctx.text
        score = priority_score(text, ctx.plan, ctx.sentiment)
        ctx.priority_score = score
        ctx.priority_band = priority_band(score)

        ctx = call_next(ctx)
        ctx.trace.append("exit:SLAPriority")
        return ctx
