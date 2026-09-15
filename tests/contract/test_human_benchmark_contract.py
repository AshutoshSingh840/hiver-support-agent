"""Contract tests for human benchmark schemas and validation rules."""

import pytest
from pydantic import ValidationError
from src.models import HumanAnnotationRecord, HumanBenchmarkManifest, IntentCode


def test_human_annotation_record_defaults():
    rec = HumanAnnotationRecord(
        conversation_id=12345,
        customer_text="Help with my iPad",
    )
    assert rec.conversation_id == 12345
    assert rec.intent is None
    assert rec.escalation_required is None
    assert rec.annotator_id is None
    assert rec.annotation_version is None


def test_human_annotation_record_rejects_automated_annotators():
    prohibited_names = ["rule_assisted", "rule_assisted_curated", "automated", "bot", "gemini", "gpt", "mock"]
    for name in prohibited_names:
        with pytest.raises(ValidationError, match="Automated/heuristic agents cannot populate human benchmark"):
            HumanAnnotationRecord(
                conversation_id=12345,
                customer_text="Help with my iPad",
                intent=IntentCode.INT_IOS,
                escalation_required=False,
                annotator_id=name,
                annotation_version="v1.0",
            )


def test_human_annotation_record_accepts_valid_human_annotator():
    rec = HumanAnnotationRecord(
        conversation_id=12345,
        customer_text="Help with my iPad",
        intent=IntentCode.INT_IOS,
        escalation_required=False,
        annotator_id="ashutosh_annotator_01",
        annotation_version="v1.0-human",
    )
    assert rec.annotator_id == "ashutosh_annotator_01"
    assert rec.intent == IntentCode.INT_IOS
    assert rec.escalation_required is False


def test_human_annotation_record_accepts_numeric_annotator_and_version():
    rec = HumanAnnotationRecord(
        conversation_id=12345,
        customer_text="Help with my iPad",
        intent=IntentCode.INT_IOS,
        escalation_required=False,
        annotator_id=1,
        annotation_version=1.0,
    )
    assert rec.annotator_id == 1
    assert rec.annotation_version == 1.0

    # Model validation from raw dictionary
    raw_dict = {
        "conversation_id": 12345,
        "customer_text": "Help with my iPad",
        "intent": "INT-IOS",
        "escalation_required": False,
        "ambiguity": False,
        "annotator_id": 42,
        "annotation_version": 2.5,
        "annotation_notes": "Valid note"
    }
    loaded = HumanAnnotationRecord.model_validate(raw_dict)
    assert loaded.annotator_id == 42
    assert loaded.annotation_version == 2.5


def test_human_benchmark_manifest_contract():
    manifest = HumanBenchmarkManifest(
        frozen_at="2026-09-12T00:00:00Z",
        benchmark_sha256="abc1234567890",
        record_count=200,
        conversation_ids_sha256="def0987654321",
        source_twcs_sha256="1234567890abcdef",
        annotation_version="v1.0-human",
        annotator_ids=["annotator_1", "annotator_2"],
    )
    assert manifest.record_count == 200
    assert len(manifest.annotator_ids) == 2
    assert manifest.temporal_split_date == "2017-11-01"
