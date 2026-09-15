"""Temporal partitioner and split manifest generator."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Set
from src.data.models import Conversation


def parse_split_date(date_str: str) -> datetime:
    """Parses date string 'YYYY-MM-DD' into UTC datetime."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def partition_conversations(
    conversations: List[Conversation],
    split_date_str: str = "2017-11-01"
) -> Tuple[List[Conversation], List[Conversation]]:
    """
    Partitions conversations into (retrieval_corpus, eval_candidate_pool).
    
    Cutoff condition:
      - retrieval_corpus: conversation.created_at < split_date
      - eval_candidate_pool: conversation.created_at >= split_date
    """
    split_dt = parse_split_date(split_date_str)
    retrieval_corpus: List[Conversation] = []
    eval_candidates: List[Conversation] = []

    for conv in conversations:
        conv_dt = conv.created_at
        if conv_dt.tzinfo is None:
            conv_dt = conv_dt.replace(tzinfo=timezone.utc)

        if conv_dt < split_dt:
            retrieval_corpus.append(conv)
        else:
            eval_candidates.append(conv)

    return retrieval_corpus, eval_candidates


def verify_split_leakage(
    retrieval_ids: Set[int],
    eval_ids: Set[int]
) -> Tuple[bool, Set[int]]:
    """
    Asserts zero conversation-level ID overlap.
    Returns (passed, overlapping_ids).
    """
    overlap = retrieval_ids.intersection(eval_ids)
    return len(overlap) == 0, overlap


def save_partition_jsonl(conversations: List[Conversation], output_path: Path):
    """Saves a list of conversations to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(conv.model_dump_json() + "\n")


def generate_split_manifest(
    retrieval_corpus: List[Conversation],
    eval_candidates: List[Conversation],
    raw_hash: str,
    output_path: Path,
    split_date_str: str = "2017-11-01"
) -> Dict:
    """Generates and writes a cryptographic and statistical split manifest."""
    retrieval_ids = [c.conversation_id for c in retrieval_corpus]
    eval_ids = [c.conversation_id for c in eval_candidates]
    
    passed, overlap = verify_split_leakage(set(retrieval_ids), set(eval_ids))
    if not passed:
        raise ValueError(f"Leakage detected! {len(overlap)} overlapping conversation IDs.")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_twcs_sha256": raw_hash,
        "split_date": split_date_str,
        "retrieval_corpus_count": len(retrieval_corpus),
        "eval_candidate_pool_count": len(eval_candidates),
        "total_conversations": len(retrieval_corpus) + len(eval_candidates),
        "leakage_check_passed": passed,
        "retrieval_conversation_ids": retrieval_ids,
        "eval_conversation_ids": eval_ids
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
