"""Unit tests for metric calculation engine."""

import pytest
from src.eval.metrics import (
    compute_intent_metrics,
    compute_retrieval_mrr_and_precision,
    compute_escalation_safety_metrics,
    compute_judge_alignment,
)


def test_intent_metrics():
    y_true = ["INT-BATTERY", "INT-BATTERY", "INT-IOS", "INT-ICLOUD"]
    y_pred = ["INT-BATTERY", "INT-IOS", "INT-IOS", "INT-ICLOUD"]
    labels = ["INT-BATTERY", "INT-IOS", "INT-ICLOUD"]

    macro_f1, per_class_f1, cm, detailed = compute_intent_metrics(y_true, y_pred, labels=labels)
    assert macro_f1 > 0.0
    assert per_class_f1["INT-ICLOUD"] == 1.0
    assert cm["INT-ICLOUD"]["INT-ICLOUD"] == 1


def test_retrieval_metrics():
    retrieved = [
        [101, 102, 103, 104, 105],  # Rank 1 hit -> RR = 1.0
        [205, 201, 202, 203, 204],  # Rank 2 hit -> RR = 0.5
        [301, 302, 303, 304, 305],  # No hit -> RR = 0.0
    ]
    relevant = [
        {101},
        {201},
        {999}
    ]

    mrr, prec = compute_retrieval_mrr_and_precision(retrieved, relevant, k=5)
    # MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert pytest.approx(mrr, 0.01) == 0.50


def test_escalation_safety_metrics():
    y_true = ["escalate", "escalate", "auto-handle", "auto-handle"]
    y_pred = ["escalate", "auto-handle", "auto-handle", "escalate"]

    metrics = compute_escalation_safety_metrics(y_true, y_pred)
    # Total true escalations = 2
    # Correctly escalated = 1
    # Unsafe auto-handled = 1
    # False-auto-handle rate = 1 / 2 = 0.50
    # Precision = 1 / 2 = 0.50
    assert metrics["false_auto_handle_rate"] == 0.50
    assert metrics["escalation_precision"] == 0.50
    assert metrics["escalation_recall"] == 0.50


def test_judge_alignment():
    human = [4, 5, 3, 2, 5]
    judge = [4, 4, 3, 1, 5]  # All within 1 point of human

    agreement_rate, kappa = compute_judge_alignment(human, judge)
    assert agreement_rate == 1.0
    assert kappa > 0.40


def test_threshold_sweep():
    from src.eval.metrics import compute_threshold_sweep

    risk_scores = [0.10, 0.30, 0.45, 0.70, 0.90]
    y_true = ["auto-handle", "auto-handle", "escalate", "escalate", "escalate"]
    thresholds = [0.20, 0.40, 0.60, 0.80]

    sweep = compute_threshold_sweep(risk_scores, y_true, thresholds=thresholds)
    assert len(sweep) == 4

    # At tau = 0.40:
    # preds = [auto, auto, esc, esc, esc] -> matches y_true perfectly
    s_04 = [s for s in sweep if s["threshold"] == 0.40][0]
    assert s_04["false_auto_handle_rate"] == 0.0
    assert s_04["escalation_precision"] == 1.0
    assert s_04["escalation_recall"] == 1.0
    assert s_04["auto_handle_coverage"] == 0.40

    # At tau = 0.80:
    # preds = [auto, auto, auto, auto, esc] -> 2 false auto-handles
    s_08 = [s for s in sweep if s["threshold"] == 0.80][0]
    assert s_08["false_auto_handle_rate"] == pytest.approx(2 / 3, 0.01)
    assert s_08["auto_handle_coverage"] == 0.80


def test_confidence_calibration():
    from src.eval.metrics import compute_confidence_calibration

    confidences = [0.95, 0.90, 0.85, 0.60, 0.40]
    y_true = ["INT-IOS", "INT-IOS", "INT-BATTERY", "INT-CONN", "INT-STORE"]
    y_pred = ["INT-IOS", "INT-IOS", "INT-BATTERY", "INT-IOS", "INT-CONN"]
    buckets = [(0.0, 0.50), (0.50, 0.80), (0.80, 1.01)]

    calib = compute_confidence_calibration(confidences, y_true, y_pred, buckets=buckets)
    assert len(calib) == 3

    # High confidence bucket [0.80, 1.01]: 3 samples, 3 correct (accuracy = 1.0)
    high_bucket = calib[2]
    assert high_bucket["sample_count"] == 3
    assert high_bucket["accuracy"] == 1.0
    assert high_bucket["mean_confidence"] == pytest.approx(0.90, 0.01)

