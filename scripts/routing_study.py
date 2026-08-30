"""How many tickets does the auto-reply gate keep away from the model?

Drafting is the only stage in the triage pass that leaves the machine. The gate
in `supportpilot.triage.routing` refuses it when the predicted category is a
guess - a confident reply grounded in the wrong KB article is worse than no
reply - or when the category is one a human must always answer.

This measures the gate over the synthetic corpus: how often it fires, and why.
Classification and the priority scorecard are the only work done here, so it
runs offline and needs no retrieval index or model.

Usage:
    uv run python scripts/routing_study.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic import generate

from supportpilot.settings import get_config
from supportpilot.triage.classify import build_pipeline, classify, priority_band, priority_score
from supportpilot.triage.routing import route


def study(n_tickets: int, seed: int) -> dict:
    tickets = generate(n_tickets, seed=seed)
    split = int(len(tickets) * 0.7)
    train, holdout = tickets.iloc[:split], tickets.iloc[split:]
    pipeline = build_pipeline().fit(train["text"], train["category"])

    escalations: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    drafted = 0
    confidences = []

    for row in holdout.itertuples(index=False):
        classification = classify(pipeline, row.text)
        band = priority_band(priority_score(row.text, "pro", 0.0))
        decision = route(classification, band)
        confidences.append(classification["confidence"])
        if decision.draft:
            drafted += 1
        else:
            kind = "never_category" if "never auto-replied" in decision.reason else "low_confidence"
            escalations[kind] += 1
            by_category[classification["category"]] += 1

    total = len(holdout)
    return {
        "n_tickets": total,
        "drafted": drafted,
        "escalated": total - drafted,
        "escalated_pct": round((total - drafted) / total, 4) if total else 0.0,
        "reasons": dict(escalations),
        "escalated_by_category": dict(by_category),
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    policy = get_config()["triage"]["auto_reply"]
    result = study(args.tickets, args.seed)

    print(f"policy: min_confidence={policy['min_confidence']}, "
          f"never_categories={policy['never_categories']}")
    print(f"holdout: {result['n_tickets']} tickets, "
          f"mean category confidence {result['mean_confidence']:.3f}\n")
    print(f"  drafted automatically : {result['drafted']:>5}")
    print(f"  escalated to a human  : {result['escalated']:>5}  "
          f"({result['escalated_pct']:.1%} of model calls avoided)")
    print()
    print("  escalation reasons:")
    for reason, count in sorted(result["reasons"].items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<16} {count:>5}")
    print()
    print("  escalated by predicted category:")
    for category, count in sorted(result["escalated_by_category"].items(), key=lambda kv: -kv[1]):
        print(f"    {category:<16} {count:>5}")


if __name__ == "__main__":
    main()
