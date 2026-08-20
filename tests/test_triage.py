"""Classification + priority scoring."""

from sklearn.model_selection import train_test_split

from supportpilot.triage.classify import (
    build_pipeline,
    classify,
    priority_band,
    priority_score,
)


def test_classifier_beats_chance_comfortably(tickets):
    x_train, x_test, y_train, y_test = train_test_split(
        tickets["text"],
        tickets["category"],
        test_size=0.3,
        random_state=0,
        stratify=tickets["category"],
    )
    pipeline = build_pipeline().fit(x_train, y_train)
    accuracy = (pipeline.predict(x_test) == y_test).mean()
    assert accuracy > 0.6, f"6-class accuracy {accuracy:.2f} too low"
    assert accuracy < 1.0, "perfect accuracy suggests leakage or trivial data"


def test_classify_returns_alternatives(tickets):
    pipeline = build_pipeline().fit(tickets["text"], tickets["category"])
    result = classify(pipeline, "I cannot log in and the password reset never arrives")
    assert result["category"] == "account_access"
    assert len(result["alternatives"]) == 2
    assert 0 < result["confidence"] <= 1


def test_priority_orders_sensibly():
    calm_basic = priority_score("how do I export data", "basic", 0.3)
    angry_enterprise = priority_score("production is down urgent", "enterprise", -0.8)
    assert angry_enterprise > calm_basic
    assert priority_band(angry_enterprise) == "P1"
    assert priority_band(calm_basic) == "P3"


def test_priority_bounded():
    assert 0 <= priority_score("down outage urgent production", "enterprise", -1.0) <= 1
