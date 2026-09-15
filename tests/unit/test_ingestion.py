"""Unit tests for dataset ingestion and BFS conversation reconstruction."""

from datetime import datetime, timezone
import pytest
from src.data.ingestion import TweetIndex
from src.data.models import SpeakerRole


def test_bfs_conversation_reconstruction():
    index = TweetIndex()
    # Mock tweets:
    # 101: Customer root (inbound=True, parent=None, response=102,103)
    # 102: AppleSupport reply (inbound=False, parent=101, response=104)
    # 103: OtherUser reply (inbound=True, parent=101)
    # 104: Customer follow-up (inbound=True, parent=102, response=105)
    # 105: AppleSupport resolution (inbound=False, parent=104)

    dt1 = datetime(2017, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2017, 10, 1, 10, 5, 0, tzinfo=timezone.utc)
    dt3 = datetime(2017, 10, 1, 10, 6, 0, tzinfo=timezone.utc)
    dt4 = datetime(2017, 10, 1, 10, 10, 0, tzinfo=timezone.utc)
    dt5 = datetime(2017, 10, 1, 10, 15, 0, tzinfo=timezone.utc)

    index.tweet_author = {101: "115854", 102: "AppleSupport", 103: "115855", 104: "115854", 105: "AppleSupport"}
    index.tweet_inbound = {101: True, 102: False, 103: True, 104: True, 105: False}
    index.tweet_created = {101: dt1, 102: dt2, 103: dt3, 104: dt4, 105: dt5}
    index.tweet_text = {
        101: "My iPhone 7 will not charge",
        102: "We can help. Have you tried a different lightning cable?",
        103: "Same thing happened to me!",
        104: "Yes, tried 3 cables.",
        105: "Please send us a DM to set up a repair."
    }
    index.tweet_parent = {101: None, 102: 101, 103: 101, 104: 102, 105: 104}
    index.forward_graph = {101: [102, 103], 102: [104], 103: [], 104: [105], 105: []}
    index.clean_customer_roots = [101]

    conv = index.reconstruct_conversation(101)
    assert conv is not None
    assert conv.conversation_id == 101
    assert conv.brand == "AppleSupport"
    assert conv.customer_id == "115854"
    assert conv.turn_count == 5
    assert conv.is_multi_turn is True
    assert conv.is_dm_escalation is True
    assert conv.turns[0].speaker == SpeakerRole.CUSTOMER
    assert conv.turns[0].text == "My iPhone 7 will not charge"
    assert conv.resolution_turn.text == "Please send us a DM to set up a repair."


def test_reconstruct_non_brand_root():
    index = TweetIndex()
    # 201: Customer root without any brand response
    index.tweet_author = {201: "115854"}
    index.tweet_inbound = {201: True}
    index.tweet_created = {201: datetime.now(timezone.utc)}
    index.tweet_text = {201: "Random question to the void"}
    index.tweet_parent = {201: None}
    index.forward_graph = {201: []}
    index.clean_customer_roots = [201]

    conv = index.reconstruct_conversation(201)
    assert conv is None
