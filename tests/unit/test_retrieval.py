"""Unit tests for BM25 indexing and retrieval engine."""

from datetime import datetime, timezone
import pytest
from src.retrieval.bm25 import BM25Index, tokenize
from src.retrieval.engine import RetrievalEngine
from src.data.models import Conversation, DialogueTurn, SpeakerRole
from src.models import IntentCode


def test_bm25_tokenization():
    text = "iPhone 8 battery draining fast after iOS 11.1 update!"
    tokens = tokenize(text)
    assert "iphone" in tokens
    assert "battery" in tokens
    assert "draining" in tokens


def test_bm25_index_search():
    index = BM25Index()
    docs = [
        (101, "iPhone battery replacement and draining troubleshooting"),
        (102, "iCloud sync error and photo backup failed"),
        (103, "Wi-Fi connection dropped on Mac"),
    ]
    index.fit(docs)

    results = index.search("battery dying quickly", top_k=2)
    assert len(results) >= 1
    assert results[0][0] == 101  # Document 101 is top match


def test_retrieval_engine_with_conversations():
    dt = datetime(2017, 10, 1, tzinfo=timezone.utc)
    t1 = DialogueTurn(tweet_id=1, speaker=SpeakerRole.CUSTOMER, author_id="u1", text="Battery drains fast", created_at=dt)
    t2 = DialogueTurn(tweet_id=2, speaker=SpeakerRole.AGENT, author_id="AppleSupport", text="Check Battery Health in Settings.", created_at=dt)
    conv1 = Conversation(conversation_id=101, created_at=dt, turns=[t1, t2], root_text=t1.text, turn_count=2, customer_id="u1")

    engine = RetrievalEngine(min_relevance_score=0.20)
    engine.build_from_conversations([conv1])

    evidence = engine.retrieve(999, "my battery is low", query_intent=IntentCode.INT_BATTERY, top_k=3)
    assert evidence.is_empty is False
    assert len(evidence.items) == 1
    assert evidence.items[0].source_conversation_id == 101
    assert "Battery Health" in evidence.items[0].agent_resolution


def test_retrieval_recovers_relevant_doc_even_if_intent_mispredicted():
    dt = datetime(2017, 10, 1, tzinfo=timezone.utc)
    t1 = DialogueTurn(tweet_id=1, speaker=SpeakerRole.CUSTOMER, author_id="u1", text="Battery drains fast", created_at=dt)
    t2 = DialogueTurn(tweet_id=2, speaker=SpeakerRole.AGENT, author_id="AppleSupport", text="Check Battery Health in Settings.", created_at=dt)
    conv1 = Conversation(conversation_id=101, created_at=dt, turns=[t1, t2], root_text=t1.text, turn_count=2, customer_id="u1")

    engine = RetrievalEngine(min_relevance_score=0.20)
    engine.build_from_conversations([conv1])

    # Query clearly about battery, but classifier mispredicted INT_IOS: soft intent preference should still retrieve conv1
    evidence = engine.retrieve(999, "my battery drains fast", query_intent=IntentCode.INT_IOS, top_k=3)
    assert evidence.is_empty is False
    assert len(evidence.items) == 1
    assert evidence.items[0].source_conversation_id == 101

