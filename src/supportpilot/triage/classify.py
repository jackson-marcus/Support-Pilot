"""Ticket triage: TF-IDF + linear classifier with calibrated probabilities,
plus a transparent priority score.

Priority = weighted blend of negative sentiment, enterprise plan, and outage
language — deliberately a scorecard, not a black box, because support leads
need to defend queue order to customers.
"""

from __future__ import annotations

import logging
import pickle

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from supportpilot.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)

OUTAGE_WORDS = ("down", "outage", "cannot work", "completely broken", "urgent", "production")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "clf",
                CalibratedClassifierCV(
                    LogisticRegression(max_iter=2000, C=4.0), method="sigmoid", cv=3
                ),
            ),
        ]
    )


def priority_score(text: str, plan: str, sentiment: float) -> float:
    w = get_config()["triage"]["priority_weights"]
    outage = any(word in text.lower() for word in OUTAGE_WORDS)
    score = (
        w["sentiment_negative"] * max(-sentiment, 0.0)
        + w["plan_enterprise"] * (1.0 if plan == "enterprise" else 0.35 if plan == "pro" else 0.0)
        + w["outage_keywords"] * (1.0 if outage else 0.0)
    )
    return round(float(min(score, 1.0)), 4)


def priority_band(score: float) -> str:
    if score >= 0.55:
        return "P1"
    if score >= 0.3:
        return "P2"
    return "P3"


def train() -> dict:
    cfg = get_config()
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])

    df = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "tickets.parquet")
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=cfg["triage"]["test_frac"],
        random_state=42,
        stratify=df["category"],
    )
    pipeline = build_pipeline()
    with mlflow.start_run(run_name="triage-tfidf-logreg"):
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
        }
        mlflow.log_params({"n_tickets": len(df), "n_categories": df["category"].nunique()})
        mlflow.log_metrics(metrics)
        logger.info("triage: %s", {k: round(v, 4) for k, v in metrics.items()})

    artifacts = resolve_path(cfg["data"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "triage.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    return metrics


def classify(pipeline: Pipeline, text: str) -> dict:
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    order = np.argsort(-probs)
    return {
        "category": str(classes[order[0]]),
        "confidence": round(float(probs[order[0]]), 4),
        "alternatives": [
            {"category": str(classes[i]), "prob": round(float(probs[i]), 4)} for i in order[1:3]
        ],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
