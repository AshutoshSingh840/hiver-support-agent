"""Integration tests for temporal splitting and leakage verification."""

from datetime import datetime, timezone
import pytest
from src.data.models import Conversation, DialogueTurn, SpeakerRole
from src.data.splitter import partition_conversations, verify_split_leakage, generate_split_manifest


def make_conv(cid: int, dt: datetime) -> Conversation:
    t = DialogueTurn(
        tweet_id=cid,
        speaker=SpeakerRole.CUSTOMER,
        author_id="115854",
        text=f"Customer message {cid}",
        created_at=dt
    )
    return Conversation(
        conversation_id=cid,
        created_at=dt,
        turns=[t],
        root_text=t.text,
        turn_count=1,
        customer_id="115854"
    )


def test_temporal_partitioning():
    c1 = make_conv(1, datetime(2017, 10, 15, 12, 0, 0, tzinfo=timezone.utc))
    c2 = make_conv(2, datetime(2017, 10, 31, 23, 59, 59, tzinfo=timezone.utc))
    c3 = make_conv(3, datetime(2017, 11, 1, 0, 0, 0, tzinfo=timezone.utc))
    c4 = make_conv(4, datetime(2017, 11, 15, 12, 0, 0, tzinfo=timezone.utc))

    retrieval, eval_pool = partition_conversations([c1, c2, c3, c4], split_date_str="2017-11-01")
    assert len(retrieval) == 2
    assert [c.conversation_id for c in retrieval] == [1, 2]
    assert len(eval_pool) == 2
    assert [c.conversation_id for c in eval_pool] == [3, 4]


def test_leakage_detection():
    retrieval_ids = {1, 2, 3, 4, 5}
    clean_eval_ids = {6, 7, 8, 9}
    leaked_eval_ids = {5, 6, 7, 8}

    passed, overlap = verify_split_leakage(retrieval_ids, clean_eval_ids)
    assert passed is True
    assert len(overlap) == 0

    passed_leaked, overlap_leaked = verify_split_leakage(retrieval_ids, leaked_eval_ids)
    assert passed_leaked is False
    assert overlap_leaked == {5}
