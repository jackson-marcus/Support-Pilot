"""Synthetic support-ticket corpus with realistic language and noisy labels.

Tickets are assembled from per-category phrase banks with cross-category
bleed (a billing ticket can mention login trouble), typos, and plan/sentiment
metadata — so classification is honest work, not keyword lookup.

Usage:
    uv run python scripts/make_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportpilot.settings import get_config, resolve_path

PHRASES = {
    "billing": [
        "I was charged twice this month",
        "my invoice shows the wrong amount",
        "need a refund for the duplicate payment",
        "why did my subscription price increase",
        "the receipt does not match my card statement",
        "billing cycle seems off by a week",
    ],
    "bug": [
        "the export button crashes the app",
        "I get an error 500 when saving a report",
        "the dashboard shows blank charts since the update",
        "file upload fails with a timeout",
        "search returns no results even for exact names",
        "the app freezes when I open settings",
    ],
    "how_to": [
        "how do I add a new team member",
        "where can I download my data",
        "how to set up weekly email reports",
        "can you explain how permissions work",
        "what is the difference between projects and workspaces",
        "how do I change my notification settings",
    ],
    "account_access": [
        "I cannot log in to my account",
        "password reset email never arrives",
        "my account seems locked after too many attempts",
        "two factor code is not accepted",
        "sso login redirects me in a loop",
        "I lost access after changing my email",
    ],
    "feature_request": [
        "please add a dark mode",
        "would love an integration with our CRM",
        "can you support exporting to excel",
        "a bulk edit option would save us hours",
        "we need role based access controls",
        "API webhooks for status changes would help",
    ],
    "performance": [
        "the dashboard takes forever to load",
        "queries have been very slow since yesterday",
        "page load times are terrible on large projects",
        "the app is laggy during peak hours",
        "reports time out on big date ranges",
        "sync takes hours instead of minutes",
    ],
}
FILLERS = [
    "this is really frustrating",
    "we depend on this for our daily work",
    "please help as soon as possible",
    "let me know if you need more details",
    "our whole team is affected",
    "thanks in advance",
]
OUTAGE_WORDS = ["down", "outage", "cannot work", "completely broken", "urgent", "production"]
PLANS = ["basic", "pro", "enterprise"]


def _typo(text: str, rng: np.random.Generator) -> str:
    if rng.random() > 0.25 or len(text) < 12:
        return text
    i = int(rng.integers(1, len(text) - 2))
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = list(PHRASES)
    rows = []
    for i in range(n):
        cat = categories[int(rng.integers(0, len(categories)))]
        parts = [str(rng.choice(PHRASES[cat]))]
        # Cross-category bleed makes classification non-trivial.
        if rng.random() < 0.3:
            other = categories[int(rng.integers(0, len(categories)))]
            parts.append(str(rng.choice(PHRASES[other])))
        if rng.random() < 0.6:
            parts.append(str(rng.choice(FILLERS)))
        urgent = rng.random() < 0.15
        if urgent:
            parts.insert(0, str(rng.choice(OUTAGE_WORDS)))
        text = _typo(". ".join(parts).capitalize() + ".", rng)

        plan = str(rng.choice(PLANS, p=[0.5, 0.35, 0.15]))
        sentiment = float(np.clip(rng.normal(-0.2 if urgent else 0.1, 0.4), -1, 1))
        rows.append(
            {
                "ticket_id": i + 1,
                "text": text,
                "category": cat,
                "plan": plan,
                "sentiment": round(sentiment, 3),
                "urgent_language": int(urgent),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = get_config()["data"]
    df = generate(cfg["n_tickets"], cfg["seed"])
    out = resolve_path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "tickets.parquet", index=False)
    print(f"Wrote {len(df):,} tickets -> {out / 'tickets.parquet'}")
    print(df["category"].value_counts().to_dict())


if __name__ == "__main__":
    main()
