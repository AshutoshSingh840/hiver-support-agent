"""Integration and regression tests for evaluation harness."""

from pathlib import Path
import pytest
from src.eval.harness import EvaluationHarness
from src.models import (
    GoldenExample,
    IntentCode,
    RoutingDecision,
    IntentPrediction,
    RetrievedEvidence,
    EvidenceItem
)


def test_harness_handles_escalated_trace_with_none_draft_reply(tmp_path):
    # Setup minimal golden set fixture
    golden_file = tmp_path / "test_golden.jsonl"
    manifest_file = tmp_path / "test_manifest.json"

    ge = GoldenExample(
        golden_id="GOLDEN-TEST-001",
        conversation_id=9991,
        root_text="Someone stole my phone and made fraudulent charges!",
        true_intent=IntentCode.INT_STORE,
        true_escalation=RoutingDecision.ESCALATE,
        annotator_id="rule_assisted_curated"
    )
    with open(golden_file, "w", encoding="utf-8") as f:
        f.write(ge.model_dump_json() + "\n")

    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write('{"retrieval_conversation_ids": [101, 102]}')

    harness = EvaluationHarness(
        golden_set_path=golden_file,
        split_manifest_path=manifest_file
    )

    # Mock pipeline returning an escalated trace with draft_reply = None
    def mock_intent_clf(txt):
        return IntentPrediction(conversation_id=9991, label=IntentCode.INT_STORE, confidence=0.90, supporting_excerpt="fraudulent")

    def mock_retriever(txt, intent):
        return RetrievedEvidence(query_conversation_id=9991, items=[], top_score=0.0)

    def mock_pipeline(txt):
        return {
            "conversation_id": 9991,
            "input_text": txt,
            "intent": {"label": "INT-STORE", "confidence": 0.90, "supporting_excerpt": "fraudulent"},
            "retrieval": {"top_k": 0, "top_score": 0.0, "evidence": []},
            "retrieval_items": [],
            "escalation": {
                "routing": "escalate",
                "triggers": ["sensitive_safety_trigger"],
                "rationale": "Safety keyword matched"
            },
            "draft_reply": None,  # Explicitly None for escalated tickets
            "latency_ms": 10.0
        }

    # Execute evaluation - should complete without AttributeError
    report = harness.run_evaluation(
        intent_classifier_fn=mock_intent_clf,
        retriever_fn=mock_retriever,
        pipeline_fn=mock_pipeline
    )

    assert report.golden_set_size == 1
    assert report.headline_metrics["false_auto_handle_rate"].score == 0.0
    assert report.headline_metrics["escalation_precision"].score == 1.0
    assert len(report.failure_mode_analysis) >= 0
