"""Contract and functional tests for reply generation agent."""

from datetime import datetime, timezone
import pytest
from src.generator.reply_agent import ReplyAgent
from src.generator.prompts import format_generation_prompt
from src.models import IntentCode, RoutingDecision, RetrievedEvidence, EvidenceItem


def test_prompt_formatting():
    evidence_blocks = [
        {"id": 101, "query": "Battery dies", "resolution": "Check Battery Health in Settings."}
    ]
    prompt = format_generation_prompt(
        customer_query="My phone is dying fast",
        intent_label="INT-BATTERY",
        evidence_blocks=evidence_blocks
    )
    assert "INT-BATTERY" in prompt
    assert "Evidence Item 1" in prompt
    assert "101" in prompt


def test_reply_agent_generation_with_evidence():
    dt = datetime(2017, 10, 1, tzinfo=timezone.utc)
    ev_item = EvidenceItem(
        source_conversation_id=101,
        relevance_score=0.90,
        intent_label=IntentCode.INT_BATTERY,
        customer_query="Battery issues",
        agent_resolution="Go to Settings > Battery > Battery Health to check capacity.",
        temporal_date=dt
    )
    evidence = RetrievedEvidence.from_items(999, [ev_item])

    agent = ReplyAgent()
    reply = agent.generate_draft(
        conversation_id=999,
        customer_query="My iPhone battery drains fast",
        intent_label=IntentCode.INT_BATTERY,
        retrieved_evidence=evidence
    )

    assert reply.conversation_id == 999
    assert reply.intent_label == IntentCode.INT_BATTERY
    assert reply.escalation_routing == RoutingDecision.AUTO_HANDLE
    assert 101 in reply.cited_evidence_ids
    assert len(reply.draft_text) > 0
