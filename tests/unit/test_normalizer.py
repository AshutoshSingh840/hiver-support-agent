"""Unit tests for text normalization and entity validation."""

from datetime import datetime
import pytest
from src.data.normalizer import normalize_tweet_text, extract_customer_tokens
from src.data.models import RawTweetRecord, DialogueTurn, Conversation, SpeakerRole


def test_normalize_html_entities():
    raw = "&lt;3 @115854 I need help &amp; support with iOS &gt; 11 &quot;bug&quot;"
    expected = '<3 @115854 I need help & support with iOS > 11 "bug"'
    assert normalize_tweet_text(raw) == expected


def test_normalize_whitespace():
    raw = "My phone   is   broken. \n\n  Please help!  "
    expected = "My phone is broken. Please help!"
    assert normalize_tweet_text(raw) == expected


def test_extract_customer_tokens():
    raw = "@115854 @115855 Thanks @AppleSupport for the help!"
    tokens = extract_customer_tokens(raw)
    assert tokens == ["@115854", "@115855"]


def test_raw_tweet_record_validation():
    rec = RawTweetRecord(
        tweet_id=1,
        author_id="115854",
        inbound=True,
        created_at=datetime(2017, 10, 1, 12, 0, 0),
        text="iPhone battery issue"
    )
    assert rec.tweet_id == 1
    assert rec.inbound is True

    with pytest.raises(ValueError):
        RawTweetRecord(
            tweet_id=-1,
            author_id="115854",
            inbound=True,
            created_at=datetime(2017, 10, 1, 12, 0, 0),
            text="invalid id"
        )


def test_conversation_properties():
    t1 = DialogueTurn(
        tweet_id=1,
        speaker=SpeakerRole.CUSTOMER,
        author_id="115854",
        text="My battery dies quickly",
        created_at=datetime(2017, 10, 1, 12, 0, 0)
    )
    t2 = DialogueTurn(
        tweet_id=2,
        speaker=SpeakerRole.AGENT,
        author_id="AppleSupport",
        text="Please send us a DM with your iOS version.",
        created_at=datetime(2017, 10, 1, 12, 5, 0)
    )
    conv = Conversation(
        conversation_id=1,
        created_at=datetime(2017, 10, 1, 12, 0, 0),
        turns=[t1, t2],
        root_text=t1.text,
        turn_count=2,
        customer_id="115854",
        is_multi_turn=False
    )
    assert conv.resolution_turn == t2
    assert conv.agent_resolution_text == t2.text
    assert conv.is_dm_escalation is True
