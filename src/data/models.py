"""Core data models for raw tweets and reconstructed conversations."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SpeakerRole(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"


class RawTweetRecord(BaseModel):
    tweet_id: int
    author_id: str
    inbound: bool
    created_at: datetime
    text: str
    response_tweet_id: Optional[str] = None
    in_response_to_tweet_id: Optional[int] = None

    model_config = {"frozen": True}

    @field_validator("tweet_id")
    @classmethod
    def validate_tweet_id(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tweet_id must be non-negative")
        return v


class DialogueTurn(BaseModel):
    tweet_id: int
    speaker: SpeakerRole
    author_id: str
    text: str
    created_at: datetime

    model_config = {"frozen": True}


class Conversation(BaseModel):
    conversation_id: int  # Equals root tweet_id
    brand: str = "AppleSupport"
    created_at: datetime  # Root customer tweet timestamp
    turns: List[DialogueTurn] = Field(default_factory=list)
    root_text: str  # First customer query text
    turn_count: int
    customer_id: str
    is_multi_turn: bool = False
    has_broken_link: bool = False

    @property
    def resolution_turn(self) -> Optional[DialogueTurn]:
        """Returns the final agent response turn if present."""
        agent_turns = [t for t in self.turns if t.speaker == SpeakerRole.AGENT]
        return agent_turns[-1] if agent_turns else None

    @property
    def agent_resolution_text(self) -> str:
        res = self.resolution_turn
        return res.text if res else ""

    @property
    def is_dm_escalation(self) -> bool:
        """Checks if historical agent resolution directed to DM / private messaging."""
        res_text = self.agent_resolution_text.lower()
        return "dm" in res_text or "direct message" in res_text or "private message" in res_text
