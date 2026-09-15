"""Integration tests for human benchmark lifecycle workflow."""

import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

from src.data.models import Conversation, DialogueTurn, SpeakerRole
from src.models import HumanAnnotationRecord, IntentCode, RoutingDecision
from src.data.human_benchmark import (
    sample_human_benchmark_candidates,
    save_human_annotation_template,
    load_human_annotation_records,
    validate_human_benchmark,
    freeze_human_benchmark,
)
from src.eval.harness import EvaluationHarness


@pytest.fixture
def mock_corpus(tmp_path):
    # Create mock conversations
    convs = []
    for i in range(1, 15):
        t1 = DialogueTurn(
            tweet_id=i * 10,
            speaker=SpeakerRole.CUSTOMER,
            author_id=f"cust_{i}",
            text=f"Battery issue number {i} on iOS 11 update",
            created_at=datetime(2017, 11, 5, 12, 0, tzinfo=timezone.utc),
        )
        t2 = DialogueTurn(
            tweet_id=i * 10 + 1,
            speaker=SpeakerRole.AGENT,
            author_id="AppleSupport",
            text="Let us help with your battery.",
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


def test_full_human_benchmark_lifecycle(mock_corpus, tmp_path):
    retrieval_ids = {100, 200}
    golden_ids = {300, 400}

    # 1. Sample unannotated candidates
    candidates = sample_human_benchmark_candidates(
        conversations=mock_corpus,
        retrieval_ids=retrieval_ids,
        golden_ids=golden_ids,
        sample_size=5,
        seed=42,
    )
    template_file = tmp_path / "unannotated.jsonl"
    save_human_annotation_template(candidates, template_file)

    # 2. Verify validation fails on unannotated fixture
    unannotated_records = load_human_annotation_records(template_file)
    passed, errors = validate_human_benchmark(unannotated_records, retrieval_ids, golden_ids, expected_count=5)
    assert not passed
    assert len(errors) >= 5

    # 3. Simulate independent human manual annotation
    annotated_records = []
    for rec in unannotated_records:
        rec.intent = IntentCode.INT_BATTERY
        rec.escalation_required = False
        rec.ambiguity = False
        rec.annotator_id = "human_evaluator_alpha"
        rec.annotation_version = "v1.0-manual"
        rec.annotation_notes = "Verified authentic inquiry"
        annotated_records.append(rec)

    annotated_file = tmp_path / "human_benchmark_annotated.jsonl"
    save_human_annotation_template(annotated_records, annotated_file)

    # 4. Verify validation passes on completed human benchmark
    reloaded = load_human_annotation_records(annotated_file)
    passed, errors = validate_human_benchmark(reloaded, retrieval_ids, golden_ids, expected_count=5)
    assert passed
    assert len(errors) == 0

    # 5. Freeze human benchmark into manifest
    raw_dummy = tmp_path / "raw_twcs.csv"
    raw_dummy.write_text("dummy raw csv content", encoding="utf-8")
    manifest_file = tmp_path / "human_manifest.json"

    frozen_manifest = freeze_human_benchmark(
        benchmark_path=annotated_file,
        raw_twcs_path=raw_dummy,
        retrieval_ids=retrieval_ids,
        golden_ids=golden_ids,
        output_manifest_path=manifest_file,
        annotation_version="v1.0-manual",
        expected_count=5,
    )
    assert frozen_manifest.record_count == 5
    assert frozen_manifest.annotation_version == "v1.0-manual"
    assert manifest_file.exists()

    # 6. Verify harness evaluates against completed human benchmark
    split_manifest_dummy = tmp_path / "split_manifest.json"
    split_manifest_dummy.write_text(json.dumps({"retrieval_conversation_ids": list(retrieval_ids)}), encoding="utf-8")

    harness = EvaluationHarness(
        golden_set_path=annotated_file,
        split_manifest_path=split_manifest_dummy,
    )
    harness.load_data()
    assert len(harness.golden_examples) == 5

    # 7. Verify harness rejects unannotated template
    harness_unannotated = EvaluationHarness(
        golden_set_path=template_file,
        split_manifest_path=split_manifest_dummy,
    )
    with pytest.raises(ValueError, match="Cannot evaluate against unpopulated human benchmark"):
        harness_unannotated.load_data()
