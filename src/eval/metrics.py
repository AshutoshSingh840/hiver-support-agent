"""Comprehensive metric calculators for classification, retrieval, safety, and judge alignment."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, cohen_kappa_score


def compute_intent_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, int]], Dict[str, Any]]:
    """
    Computes intent classification metrics: macro F1, per-class F1/P/R, and confusion matrix.
    """
    if not labels:
        labels = sorted(list(set(y_true + y_pred)))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    macro_f1 = float(np.mean(f1))
    per_class_f1 = {lbl: float(f) for lbl, f in zip(labels, f1)}
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {}
    for i, actual in enumerate(labels):
        cm_dict[actual] = {pred: int(cm[i, j]) for j, pred in enumerate(labels)}

    detailed = {
        "per_class_precision": {lbl: float(val) for lbl, val in zip(labels, p)},
        "per_class_recall": {lbl: float(val) for lbl, val in zip(labels, r)},
        "support": {lbl: int(val) for lbl, val in zip(labels, support)}
    }
    return macro_f1, per_class_f1, cm_dict, detailed


def compute_retrieval_mrr_and_precision(
    retrieved_items_per_query: List[List[int]],
    relevant_ids_per_query: List[Set[int]],
    k: int = 5
) -> Tuple[float, float]:
    """
    Computes Mean Reciprocal Rank (MRR@k) and Precision@k across queries.
    """
    rr_scores = []
    p_scores = []

    for retrieved, relevant in zip(retrieved_items_per_query, relevant_ids_per_query):
        top_k = retrieved[:k]
        if not relevant:
            continue

        # MRR
        rr = 0.0
        for rank, doc_id in enumerate(top_k, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

        # Precision@k
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        p_scores.append(hits / min(k, len(relevant)) if relevant else 0.0)

    mrr = float(np.mean(rr_scores)) if rr_scores else 0.0
    prec = float(np.mean(p_scores)) if p_scores else 0.0
    return mrr, prec


def compute_escalation_safety_metrics(
    y_true_routing: List[str],  # "auto-handle" or "escalate"
    y_pred_routing: List[str]
) -> Dict[str, float]:
    """
    Computes escalation precision, recall, and the primary safety metric: false-auto-handle rate.
    
    False-Auto-Handle Rate = (True Escalate & Pred Auto-Handle) / (Total True Escalate)
    """
    total_true_escalate = sum(1 for y in y_true_routing if y == "escalate")
    total_pred_escalate = sum(1 for y in y_pred_routing if y == "escalate")

    correct_escalate = sum(1 for yt, yp in zip(y_true_routing, y_pred_routing) if yt == "escalate" and yp == "escalate")
    unsafe_auto_handle = sum(1 for yt, yp in zip(y_true_routing, y_pred_routing) if yt == "escalate" and yp == "auto-handle")

    precision = (correct_escalate / total_pred_escalate) if total_pred_escalate > 0 else 0.0
    recall = (correct_escalate / total_true_escalate) if total_true_escalate > 0 else 0.0
    false_auto_handle_rate = (unsafe_auto_handle / total_true_escalate) if total_true_escalate > 0 else 0.0

    return {
        "escalation_precision": float(precision),
        "escalation_recall": float(recall),
        "false_auto_handle_rate": float(false_auto_handle_rate),
        "total_true_escalations": total_true_escalate,
        "unsafe_auto_handle_count": unsafe_auto_handle
    }


def compute_judge_alignment(
    human_ratings: List[int],  # 1 to 5
    judge_ratings: List[int]
) -> Tuple[float, float]:
    """
    Computes percentage agreement (within 1 point) and Cohen's Kappa on 1–5 scale.
    """
    if not human_ratings or not judge_ratings or len(human_ratings) != len(judge_ratings):
        return 0.0, 0.0

    # Agreement within 1 point on 1-5 scale
    agreements = sum(1 for h, j in zip(human_ratings, judge_ratings) if abs(h - j) <= 1)
    agreement_rate = agreements / len(human_ratings)

    # Cohen's kappa (nominal/ordinal agreement)
    try:
        kappa = float(cohen_kappa_score(human_ratings, judge_ratings))
        if np.isnan(kappa):
            kappa = 0.0
    except Exception:
        kappa = 0.0

    return float(agreement_rate), float(kappa)


def compute_threshold_sweep(
    risk_scores: List[float],
    y_true_routing: List[str],
    thresholds: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Computes routing safety and coverage metrics across a continuous sweep of routing thresholds.
    Enables empirical selection of the safety-first operating point on the development set.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]

    sweep_results = []
    total_samples = len(y_true_routing)
    total_true_esc = sum(1 for y in y_true_routing if y == "escalate")

    for tau in thresholds:
        preds = ["escalate" if s >= tau else "auto-handle" for s in risk_scores]
        total_pred_esc = sum(1 for p in preds if p == "escalate")
        total_pred_auto = total_samples - total_pred_esc

        tp = sum(1 for yt, yp in zip(y_true_routing, preds) if yt == "escalate" and yp == "escalate")
        fn = sum(1 for yt, yp in zip(y_true_routing, preds) if yt == "escalate" and yp == "auto-handle")

        precision = (tp / total_pred_esc) if total_pred_esc > 0 else 0.0
        recall = (tp / total_true_esc) if total_true_esc > 0 else 0.0
        false_auto_handle = (fn / total_true_esc) if total_true_esc > 0 else 0.0
        coverage = total_pred_auto / total_samples if total_samples > 0 else 0.0
        escalation_rate = total_pred_esc / total_samples if total_samples > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        sweep_results.append({
            "threshold": round(float(tau), 2),
            "false_auto_handle_rate": round(float(false_auto_handle), 4),
            "escalation_precision": round(float(precision), 4),
            "escalation_recall": round(float(recall), 4),
            "auto_handle_coverage": round(float(coverage), 4),
            "escalation_rate": round(float(escalation_rate), 4),
            "escalation_f1": round(float(f1), 4),
            "predicted_escalations": int(total_pred_esc),
            "predicted_auto_handles": int(total_pred_auto),
        })

    return sweep_results


def compute_confidence_calibration(
    confidences: List[float],
    y_true: List[str],
    y_pred: List[str],
    buckets: Optional[List[Tuple[float, float]]] = None
) -> List[Dict[str, Any]]:
    """
    Computes prediction accuracy across confidence score buckets to evaluate calibration.
    """
    if buckets is None:
        buckets = [(0.0, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 1.01)]

    calibration = []
    for low, high in buckets:
        indices = [
            i for i, c in enumerate(confidences)
            if (low <= c < high) or (high >= 1.0 and c >= low and c <= high)
        ]
        count = len(indices)
        if count == 0:
            calibration.append({
                "bucket": f"[{low:.2f}, {min(1.0, high):.2f})",
                "sample_count": 0,
                "accuracy": 0.0,
                "mean_confidence": 0.0
            })
            continue

        correct = sum(1 for i in indices if y_true[i] == y_pred[i])
        acc = correct / count
        mean_conf = sum(confidences[i] for i in indices) / count

        calibration.append({
            "bucket": f"[{low:.2f}, {min(1.0, high):.2f})",
            "sample_count": count,
            "accuracy": round(float(acc), 4),
            "mean_confidence": round(float(mean_conf), 4)
        })

    return calibration
