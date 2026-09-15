"""Integration tests for the complete end-to-end support agent pipeline."""

from datetime import datetime, timezone
import pytest
from src.pipeline import SupportAgentPipeline
from src.retrieval.engine import RetrievalEngine
from src.data.models import Conversation, DialogueTurn, SpeakerRole
from src.models import RoutingDecision, IntentCode


@pytest.fixture
def populated_pipeline():
    dt = datetime(2017, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
    convs = [
        Conversation(
            conversation_id=101,
            created_at=dt,
            turns=[
                DialogueTurn(tweet_id=1, speaker=SpeakerRole.CUSTOMER, author_id="u1", text="Battery life is terrible", created_at=dt),
                DialogueTurn(tweet_id=2, speaker=SpeakerRole.AGENT, author_id="AppleSupport", text="Check Battery Health in Settings > Battery.", created_at=dt)
            ],
            root_text="Battery life is terrible",
            turn_count=2,
            customer_id="u1"
        ),
        Conversation(
            conversation_id=102,
            created_at=dt,
            turns=[
                DialogueTurn(tweet_id=3, speaker=SpeakerRole.CUSTOMER, author_id="u2", text="iCloud sync error photos", created_at=dt),
                DialogueTurn(tweet_id=4, speaker=SpeakerRole.AGENT, author_id="AppleSupport", text="Toggle iCloud Photos in Settings > Photos.", created_at=dt)
            ],
            root_text="iCloud sync error photos",
            turn_count=2,
            customer_id="u2"
        )
    ]
    retriever = RetrievalEngine(min_relevance_score=0.10)
    retriever.build_from_conversations(convs)
    return SupportAgentPipeline(retrieval_engine=retriever)


def test_pipeline_battery_inquiry_flow(populated_pipeline):
    res = populated_pipeline.process_ticket(
        conversation_id=901,
        text="My iPhone battery drains completely in 3 hours. How do I fix it?"
    )

    assert res["conversation_id"] == 901
    assert res["intent"]["label"] == "INT-BATTERY"
    assert res["intent"]["confidence"] >= 0.70
    assert res["escalation"]["routing"] == "auto-handle"
    assert res["draft_reply"] is not None
    assert 101 in res["draft_reply"]["cited_evidence_ids"]
    assert "Settings" in res["draft_reply"]["draft_text"]


def test_pipeline_sensitive_escalation_flow(populated_pipeline):
    res = populated_pipeline.process_ticket(
        conversation_id=902,
        text="Someone hacked into my Apple ID and charged $500. This is fraud!"
    )

    assert res["conversation_id"] == 902
    assert res["escalation"]["routing"] == "escalate"
    assert res["draft_reply"] is None
    assert any("sensitive_safety_trigger" in t for t in res["escalation"]["triggers"])
