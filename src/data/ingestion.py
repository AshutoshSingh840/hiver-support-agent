"""Streaming CSV reader and deterministic BFS conversation tree reconstruction."""

import csv
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Generator
from src.data.models import RawTweetRecord, DialogueTurn, Conversation, SpeakerRole
from src.data.normalizer import normalize_tweet_text

# Increase CSV field size limit for large fields
csv.field_size_limit(sys.maxsize)


def parse_datetime(dt_str: str) -> datetime:
    """Parses Twitter datetime format: 'Tue Oct 31 22:10:45 +0000 2017'."""
    try:
        return datetime.strptime(dt_str.strip(), "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return datetime.utcnow()


class TweetIndex:
    """In-memory indexing structure for fast conversation reconstruction."""

    def __init__(self):
        self.tweet_author: Dict[int, str] = {}
        self.tweet_inbound: Dict[int, bool] = {}
        self.tweet_created: Dict[int, datetime] = {}
        self.tweet_text: Dict[int, str] = {}
        self.tweet_parent: Dict[int, Optional[int]] = {}
        self.forward_graph: Dict[int, List[int]] = {}
        self.clean_customer_roots: List[int] = []

    def load_from_csv(self, csv_path: Path, max_rows: Optional[int] = None) -> int:
        """Loads tweet graph from raw CSV file."""
        row_count = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = int(row["tweet_id"])
                auth = row["author_id"]
                inb = row["inbound"].strip().lower() == "true"
                created = parse_datetime(row["created_at"])
                text = row["text"]
                parent = int(row["in_response_to_tweet_id"]) if row.get("in_response_to_tweet_id") and row["in_response_to_tweet_id"].strip() else None
                
                resp_str = row.get("response_tweet_id", "") or ""
                responses = []
                if resp_str.strip():
                    for r in resp_str.split(","):
                        r_clean = r.strip()
                        if r_clean.isdigit():
                            responses.append(int(r_clean))

                self.tweet_author[tid] = auth
                self.tweet_inbound[tid] = inb
                self.tweet_created[tid] = created
                self.tweet_text[tid] = text
                self.tweet_parent[tid] = parent
                self.forward_graph[tid] = responses

                if inb and parent is None:
                    self.clean_customer_roots.append(tid)

                row_count += 1
                if max_rows and row_count >= max_rows:
                    break
        return row_count

    def reconstruct_conversation(self, root_id: int) -> Optional[Conversation]:
        """Reconstructs the conversation subtree rooted at root_id using BFS."""
        if root_id not in self.tweet_author:
            return None

        direct_children = self.forward_graph.get(root_id, [])
        first_brand = None
        for child_id in direct_children:
            if child_id in self.tweet_author and not self.tweet_inbound[child_id]:
                first_brand = self.tweet_author[child_id]
                break

        if not first_brand:
            return None

        # BFS collection of turns
        queue = deque([root_id])
        visited: Set[int] = set([root_id])
        turns: List[DialogueTurn] = []

        while queue:
            curr_id = queue.popleft()
            if curr_id not in self.tweet_author:
                continue

            auth = self.tweet_author[curr_id]
            inb = self.tweet_inbound[curr_id]
            speaker = SpeakerRole.CUSTOMER if inb else SpeakerRole.AGENT
            norm_text = normalize_tweet_text(self.tweet_text[curr_id])
            created = self.tweet_created[curr_id]

            turns.append(
                DialogueTurn(
                    tweet_id=curr_id,
                    speaker=speaker,
                    author_id=auth,
                    text=norm_text,
                    created_at=created
                )
            )

            for child_id in self.forward_graph.get(curr_id, []):
                if child_id not in visited and child_id in self.tweet_author:
                    visited.add(child_id)
                    queue.append(child_id)

        # Sort turns chronologically
        turns.sort(key=lambda t: t.created_at)

        root_created = self.tweet_created[root_id]
        root_text = turns[0].text if turns else ""
        cust_id = self.tweet_author[root_id]

        return Conversation(
            conversation_id=root_id,
            brand=first_brand,
            created_at=root_created,
            turns=turns,
            root_text=root_text,
            turn_count=len(turns),
            customer_id=cust_id,
            is_multi_turn=len(turns) > 2
        )


def reconstruct_conversations_for_brand(
    csv_path: Path,
    target_brand: str = "AppleSupport",
    max_rows: Optional[int] = None
) -> Generator[Conversation, None, None]:
    """Generates all reconstructed conversations for the target brand."""
    index = TweetIndex()
    index.load_from_csv(csv_path, max_rows=max_rows)
    for root_id in index.clean_customer_roots:
        conv = index.reconstruct_conversation(root_id)
        if conv and conv.brand == target_brand:
            yield conv
