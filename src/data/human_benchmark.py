"""Independent Human-Validated Evaluation Benchmark management.

Provides deterministic candidate sampling, unpopulated annotation fixture creation,
rigorous validation, leakage prevention, and cryptographic freezing.
"""

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.data.models import Conversation
from src.models import HumanAnnotationRecord, HumanBenchmarkManifest, IntentCode


PROHIBITED_AUTOMATED_ANNOTATORS = {
    "rule_assisted",
    "rule_assisted_curated",
    "automated",
    "auto",
    "bot",
    "gemini",
    "gpt",
    "llm",
    "mock",
    "synthetic",
    "classifier",
    "system",
    "unassigned",
}


def sample_human_benchmark_candidates(
    conversations: List[Conversation],
    retrieval_ids: Set[int],
    golden_ids: Set[int],
    sample_size: int = 200,
    seed: int = 42,
    target_brand: str = "AppleSupport",
) -> List[HumanAnnotationRecord]:
    """
    Deterministically samples authentic conversations for human annotation.
    
    Guarantees:
    1. Zero overlap with retrieval corpus (retrieval_ids).
    2. Zero overlap with curated development benchmark (golden_ids).
    3. Deterministic sampling using fixed random seed.
    4. Explicitly leaves all ground-truth fields unfilled (None).
    """
    # Filter candidates to post-split AppleSupport conversations excluding retrieval & golden IDs
    eligible: List[Conversation] = []
    for conv in conversations:
        if conv.brand != target_brand:
            continue
        cid = conv.conversation_id
        if cid in retrieval_ids or cid in golden_ids:
            continue
        eligible.append(conv)

    if len(eligible) < sample_size:
        raise ValueError(
            f"Insufficient eligible conversations ({len(eligible)}) for requested sample size {sample_size}."
        )

    # Sort deterministically by conversation_id before sampling
    eligible.sort(key=lambda c: c.conversation_id)

    rng = random.Random(seed)
    sampled_convs = rng.sample(eligible, sample_size)
    sampled_convs.sort(key=lambda c: c.conversation_id)

    records: List[HumanAnnotationRecord] = []
    for conv in sampled_convs:
        # Convert turns to JSON-serializable dicts
        turn_dicts = [
            {
                "tweet_id": t.tweet_id,
                "speaker": t.speaker.value,
                "author_id": t.author_id,
                "text": t.text,
                "created_at": t.created_at.isoformat() if isinstance(t.created_at, datetime) else str(t.created_at),
            }
            for t in conv.turns
        ]

        # Ground truth fields are strictly None (unpopulated)
        record = HumanAnnotationRecord(
            conversation_id=conv.conversation_id,
            customer_text=conv.root_text,
            created_at=conv.created_at.isoformat() if isinstance(conv.created_at, datetime) else str(conv.created_at),
            turns=turn_dicts,
            intent=None,
            escalation_required=None,
            ambiguity=None,
            annotator_id=None,
            annotation_version=None,
            annotation_notes=None,
        )
        records.append(record)

    return records


def save_human_annotation_template(records: List[HumanAnnotationRecord], output_path: Path) -> Path:
    """Saves unpopulated candidate records to a JSONL fixture for manual human annotation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
    return output_path


def load_human_annotation_records(file_path: Path) -> List[HumanAnnotationRecord]:
    """Loads HumanAnnotationRecord objects from a JSONL file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Human benchmark file not found: {file_path}")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            records.append(HumanAnnotationRecord.model_validate(data))
    return records


def validate_human_benchmark(
    records: List[HumanAnnotationRecord],
    retrieval_ids: Set[int],
    golden_ids: Set[int],
    expected_count: int = 200,
) -> Tuple[bool, List[str]]:
    """
    Validates that a human benchmark dataset is complete, leak-free, and authentically annotated.
    
    Rejection rules:
    1. Count mismatch (expected_count).
    2. Missing/null intent label.
    3. Missing/null escalation_required flag.
    4. Missing/null annotator_id.
    5. Prohibited automated annotator_id.
    6. Missing/null annotation_version.
    7. Duplicate conversation IDs.
    8. Overlap with retrieval corpus.
    9. Overlap with curated development benchmark (golden set).
    """
    errors: List[str] = []

    if len(records) < expected_count:
        errors.append(f"Insufficient record count: found {len(records)}, expected at least {expected_count}.")

    seen_ids: Set[int] = set()
    duplicate_ids: Set[int] = set()

    for idx, rec in enumerate(records):
        cid = rec.conversation_id
        if cid in seen_ids:
            duplicate_ids.add(cid)
        seen_ids.add(cid)

        # 1. Check intent label
        if rec.intent is None:
            errors.append(f"Record #{idx+1} (conv_id={cid}): Missing required 'intent' label.")

        # 2. Check escalation requirement
        if rec.escalation_required is None or not isinstance(rec.escalation_required, bool):
            errors.append(f"Record #{idx+1} (conv_id={cid}): Missing required 'escalation_required' decision.")

        # 3. Check ambiguity
        if rec.ambiguity is None or not isinstance(rec.ambiguity, bool):
            errors.append(f"Record #{idx+1} (conv_id={cid}): Missing required 'ambiguity' decision.")

        # 4. Check annotation notes
        if rec.annotation_notes is None or not str(rec.annotation_notes).strip():
            errors.append(f"Record #{idx+1} (conv_id={cid}): Missing required 'annotation_notes'.")

        # 5. Check annotator identity
        if rec.annotator_id is None or not str(rec.annotator_id).strip():
            errors.append(f"Record #{idx+1} (conv_id={cid}): Missing required 'annotator_id'.")
        elif str(rec.annotator_id).lower().strip() in PROHIBITED_AUTOMATED_ANNOTATORS:
            errors.append(
                f"Record #{idx+1} (conv_id={cid}): Prohibited automated annotator '{rec.annotator_id}'. "
                f"Human benchmark must be annotated by independent human annotator."
            )

        # 6. Check annotation version
        if rec.annotation_version is None or not str(rec.annotation_version).strip():
            errors.append(f"Record #{idx+1} (conv_id={cid}): Missing required 'annotation_version'.")

    # Check duplicates
    if duplicate_ids:
        errors.append(f"Found {len(duplicate_ids)} duplicate conversation IDs: {sorted(list(duplicate_ids))[:10]}")

    # Check leakage against retrieval corpus
    retrieval_overlap = seen_ids.intersection(retrieval_ids)
    if retrieval_overlap:
        errors.append(
            f"Leakage detected! {len(retrieval_overlap)} conversation IDs overlap with retrieval corpus: "
            f"{sorted(list(retrieval_overlap))[:10]}"
        )

    # Check leakage against curated development benchmark
    golden_overlap = seen_ids.intersection(golden_ids)
    if golden_overlap:
        errors.append(
            f"Leakage detected! {len(golden_overlap)} conversation IDs overlap with curated golden set: "
            f"{sorted(list(golden_overlap))[:10]}"
        )

    passed = len(errors) == 0
    return passed, errors


def freeze_human_benchmark(
    benchmark_path: Path,
    raw_twcs_path: Path,
    retrieval_ids: Set[int],
    golden_ids: Set[int],
    output_manifest_path: Path,
    annotation_version: str = "v1.0-human",
    temporal_split_date: str = "2017-11-01",
    expected_count: int = 200,
) -> HumanBenchmarkManifest:
    """
    Validates and freezes the completed human benchmark into an immutable cryptographic manifest.
    """
    records = load_human_annotation_records(benchmark_path)
    passed, errors = validate_human_benchmark(records, retrieval_ids, golden_ids, expected_count=expected_count)
    if not passed:
        raise ValueError(f"Cannot freeze invalid human benchmark:\n" + "\n".join(errors))

    # Compute SHA-256 of the benchmark file
    with open(benchmark_path, "rb") as f:
        bench_hash = hashlib.sha256(f.read()).hexdigest()

    # Compute SHA-256 of sorted conversation IDs
    sorted_ids = sorted([r.conversation_id for r in records])
    ids_hash = hashlib.sha256(",".join(map(str, sorted_ids)).encode("utf-8")).hexdigest()

    # Compute SHA-256 of raw source dataset
    if raw_twcs_path.exists():
        with open(raw_twcs_path, "rb") as f:
            source_hash = hashlib.sha256(f.read()).hexdigest()
    else:
        source_hash = "UNKNOWN_SOURCE"

    annotators = sorted(list(set(str(r.annotator_id) for r in records if r.annotator_id is not None)))

    manifest = HumanBenchmarkManifest(
        frozen_at=datetime.now(timezone.utc).isoformat(),
        benchmark_sha256=bench_hash,
        record_count=len(records),
        conversation_ids_sha256=ids_hash,
        source_twcs_sha256=source_hash,
        temporal_split_date=temporal_split_date,
        annotation_version=annotation_version,
        annotator_ids=annotators,
    )

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    return manifest
