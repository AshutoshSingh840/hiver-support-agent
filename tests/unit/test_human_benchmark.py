"""Unit tests for independent human-validated evaluation benchmark."""

import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
from src.data.models import Conversation, DialogueTurn, SpeakerRole
from src.models import HumanAnnotationRecord, IntentCode, HumanBenchmarkManifest
from src.data.human_benchmark import (
    sample_human_benchmark_candidates,
    validate_human_benchmark,
    freeze_human_benchmark,
    load_human_annotation_records,
    save_human_annotation_template,
)


@pytest.fixture
def sample_conversations():
    convs = []
    for i in range(1, 20):
        t1 = DialogueTurn(
            tweet_id=i * 10,
            speaker=SpeakerRole.CUSTOMER,
            author_id=f"cust_{i}",
            text=f"Sample inquiry text for customer {i} with battery issue.",
            created_at=datetime(2017, 11, 5, 12, 0, tzinfo=timezone.utc),
        )
        t2 = DialogueTurn(
            tweet_id=i * 10 + 1,
            speaker=SpeakerRole.AGENT,
            author_id="AppleSupport",
            text="Let us help with that.",
            created_at=datetime(2017, 11, 5, 12, 10, tzinfo=timezone.utc),
        )
        convs.append(
            Conversation(
                conversation_id=i * 100,
                brand="AppleSupport",
                created_at=datetime(2017, 11, 5, 12, 0, tzinfo=timezone.utc),
                turns=[t1, t2],
                root_text=t1.text,
                turn_count=2,
                customer_id=f"cust_{i}",
                is_multi_turn=False,
            )
        )
    return convs


def test_deterministic_sampling(sample_conversations):
    retrieval_ids = {100, 200}
    golden_ids = {300, 400}

    # Run 1
    sample1 = sample_human_benchmark_candidates(
        conversations=sample_conversations,
        retrieval_ids=retrieval_ids,
        golden_ids=golden_ids,
        sample_size=5,
        seed=42,
    )
    # Run 2
    sample2 = sample_human_benchmark_candidates(
        conversations=sample_conversations,
        retrieval_ids=retrieval_ids,
        golden_ids=golden_ids,
        sample_size=5,
        seed=42,
    )

    ids1 = [r.conversation_id for r in sample1]
    ids2 = [r.conversation_id for r in sample2]
    assert ids1 == ids2, "Sampling with same seed must be identical"
    assert len(ids1) == 5

    # Check zero leakage
    for cid in ids1:
        assert cid not in retrieval_ids
        assert cid not in golden_ids

    # Check that ground truth fields are explicitly unpopulated
    for rec in sample1:
        assert rec.intent is None
        assert rec.escalation_required is None
        assert rec.annotator_id is None
        assert rec.annotation_version is None


def test_sampling_seed_variation(sample_conversations):
    retrieval_ids = {100}
    golden_ids = {200}

    sample_seed_42 = sample_human_benchmark_candidates(
        conversations=sample_conversations,
        retrieval_ids=retrieval_ids,
        golden_ids=golden_ids,
        sample_size=5,
        seed=42,
    )
    sample_seed_99 = sample_human_benchmark_candidates(
        conversations=sample_conversations,
        retrieval_ids=retrieval_ids,
        golden_ids=golden_ids,
        sample_size=5,
        seed=99,
    )

    ids_42 = [r.conversation_id for r in sample_seed_42]
    ids_99 = [r.conversation_id for r in sample_seed_99]
    assert ids_42 != ids_99, "Different seeds must produce different sample orderings/draws"


def test_validate_unannotated_fails():
    unannotated = [
        HumanAnnotationRecord(
            conversation_id=101,
            customer_text="iPhone screen broke",
            created_at="2017-11-02T00:00:00Z",
            intent=None,
            escalation_required=None,
            annotator_id=None,
            annotation_version=None,
        )
    ]
    passed, errors = validate_human_benchmark(unannotated, retrieval_ids=set(), golden_ids=set(), expected_count=1)
    assert not passed
    assert any("Missing required 'intent'" in e for e in errors)
    assert any("Missing required 'escalation_required'" in e for e in errors)
    assert any("Missing required 'annotator_id'" in e for e in errors)


def test_validate_prohibited_automated_annotator():
    # Pydantic model itself rejects prohibited names
    with pytest.raises(Exception, match="Automated/heuristic agents cannot populate human benchmark"):
        HumanAnnotationRecord(
            conversation_id=101,
            customer_text="iPhone screen broke",
            created_at="2017-11-02T00:00:00Z",
            intent=IntentCode.INT_HARDWARE,
            escalation_required=False,
            annotator_id="rule_assisted_curated",
            annotation_version="v1.0",
        )


def test_validate_duplicate_ids():
    records = [
        HumanAnnotationRecord(
            conversation_id=101,
            customer_text="iPhone screen broke",
            created_at="2017-11-02T00:00:00Z",
            intent=IntentCode.INT_HARDWARE,
            escalation_required=False,
            ambiguity=False,
            annotator_id="expert_reviewer_1",
            annotation_version="v1.0",
            annotation_notes="Verified",
        ),
        HumanAnnotationRecord(
            conversation_id=101,
            customer_text="Duplicate id test",
            created_at="2017-11-02T00:00:00Z",
            intent=IntentCode.INT_HARDWARE,
            escalation_required=False,
            ambiguity=False,
            annotator_id="expert_reviewer_1",
            annotation_version="v1.0",
            annotation_notes="Verified",
        ),
    ]
    passed, errors = validate_human_benchmark(records, retrieval_ids=set(), golden_ids=set(), expected_count=2)
    assert not passed
    assert any("duplicate conversation ids" in e.lower() for e in errors)


def test_validate_leakage_detected():
    records = [
        HumanAnnotationRecord(
            conversation_id=101,
            customer_text="iPhone screen broke",
            created_at="2017-11-02T00:00:00Z",
            intent=IntentCode.INT_HARDWARE,
            escalation_required=False,
            ambiguity=False,
            annotator_id="expert_reviewer_1",
            annotation_version="v1.0",
            annotation_notes="Verified",
        )
    ]
    passed, errors = validate_human_benchmark(records, retrieval_ids={101}, golden_ids=set(), expected_count=1)
    assert not passed
    assert any("Leakage detected" in e for e in errors)


def test_validate_numeric_annotator_and_version():
    records = [
        HumanAnnotationRecord(
            conversation_id=101,
            customer_text="iPhone screen broke",
            created_at="2017-11-02T00:00:00Z",
            intent=IntentCode.INT_HARDWARE,
            escalation_required=False,
            ambiguity=False,
            annotator_id=1,
            annotation_version=1.0,
            annotation_notes="Numeric annotator and version verified",
        )
    ]
    passed, errors = validate_human_benchmark(records, retrieval_ids=set(), golden_ids=set(), expected_count=1)
    assert passed
    assert len(errors) == 0
