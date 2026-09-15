"""Contract tests verifying Pydantic models against pipeline JSON schemas."""

import json
from datetime import datetime
from pathlib import Path
import pytest
from src.models import (
    IntentCode,
    IntentPrediction,
    EvidenceItem,
    RetrievedEvidence,
    RoutingDecision,
    EscalationDecision,
    DraftReply,
)


@pytest.fixture
def schema_definitions():
    schema_path = Path("docs/contracts/PIPELINE_SCHEMAS.json")
    assert schema_path.exists(), "PIPELINE_SCHEMAS.json must exist"
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["definitions"]


def test_intent_prediction_schema_compliance(schema_definitions):
    pred = IntentPrediction(
        conversation_id=115854,
        label=IntentCode.INT_BATTERY,
        confidence=0.92,
        supporting_excerpt="battery drains rapidly"
    )
    dumped = json.loads(pred.model_dump_json())
    
    schema = schema_definitions["IntentPrediction"]
    for req in schema["required"]:
        assert req in dumped
    assert dumped["label"] in schema["properties"]["label"]["enum"]
    assert 0.0 <= dumped["confidence"] <= 1.0


def test_evidence_item_schema_compliance(schema_definitions):
    item = EvidenceItem(
        source_conversation_id=115854,
        relevance_score=0.88,
        intent_label=IntentCode.INT_BATTERY,
        customer_query="battery life low",
        agent_resolution="Check battery health settings",
        is_dm_escalation=False,
        temporal_date=datetime(2017, 10, 15, 12, 0, 0)
    )
    dumped = json.loads(item.model_dump_json())
    schema = schema_definitions["EvidenceItem"]
    for req in schema["required"]:
        assert req in dumped
    assert 0.0 <= dumped["relevance_score"] <= 1.0


def test_escalation_decision_schema_compliance(schema_definitions):
    decision = EscalationDecision(
        conversation_id=115854,
        routing=RoutingDecision.ESCALATE,
        triggers=["sensitive_keyword_trigger"],
        intent_confidence=0.45,
        evidence_top_score=0.20,
        rationale="Low confidence and keyword trigger"
    )
    dumped = json.loads(decision.model_dump_json())
    schema = schema_definitions["EscalationDecision"]
    for req in schema["required"]:
        assert req in dumped
    assert dumped["routing"] in schema["properties"]["routing"]["enum"]


def test_draft_reply_schema_compliance(schema_definitions):
    reply = DraftReply(
        conversation_id=115854,
        draft_text="Please check your battery settings.",
        cited_evidence_ids=[115854],
        intent_label=IntentCode.INT_BATTERY,
        escalation_routing=RoutingDecision.AUTO_HANDLE,
        escalation_rationale="Grounded in precedent"
    )
    dumped = json.loads(reply.model_dump_json())
    schema = schema_definitions["DraftReply"]
    for req in schema["required"]:
        assert req in dumped
    assert dumped["escalation_routing"] in schema["properties"]["escalation_routing"]["enum"]


def test_golden_example_provenance_contract():
    from src.models import GoldenExample

    # Must accept valid rule-assisted curation label
    ge = GoldenExample(
        golden_id="GOLDEN-001",
        conversation_id=101,
        root_text="Battery issue",
        true_intent=IntentCode.INT_BATTERY,
        true_escalation=RoutingDecision.AUTO_HANDLE,
        annotator_id="rule_assisted_curated"
    )
    assert ge.annotator_id == "rule_assisted_curated"

    # Must reject misleading human annotation claims
    with pytest.raises(ValueError, match="Misleading annotator_id"):
        GoldenExample(
            golden_id="GOLDEN-002",
            conversation_id=102,
            root_text="Battery issue",
            true_intent=IntentCode.INT_BATTERY,
            true_escalation=RoutingDecision.AUTO_HANDLE,
            annotator_id="human_annotator_curated"
        )

