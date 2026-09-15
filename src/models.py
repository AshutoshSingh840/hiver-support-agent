"""Pydantic models for AI predictions, retrieval evidence, decisions, and eval."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IntentCode(str, Enum):
    INT_IOS = "INT-IOS"
    INT_STORE = "INT-STORE"
    INT_ICLOUD = "INT-ICLOUD"
    INT_HARDWARE = "INT-HARDWARE"
    INT_CONN = "INT-CONN"
    INT_BATTERY = "INT-BATTERY"
    INT_WATCH_MAC = "INT-WATCH-MAC"
    INT_OUT_OF_SCOPE = "INT-OUT-OF-SCOPE"


class RoutingDecision(str, Enum):
    AUTO_HANDLE = "auto-handle"
    ESCALATE = "escalate"


class IntentPrediction(BaseModel):
    conversation_id: int
    label: IntentCode
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_excerpt: str
    model_version: str = "1.0.0"
    created_at: datetime = Field(default_factory=get_utc_now)


class EvidenceItem(BaseModel):
    source_conversation_id: int
    relevance_score: float = Field(ge=0.0, le=1.0)
    intent_label: IntentCode
    customer_query: str
    agent_resolution: str
    is_dm_escalation: bool = False
    temporal_date: datetime


class RetrievedEvidence(BaseModel):
    query_conversation_id: int
    items: List[EvidenceItem] = Field(default_factory=list)
    top_score: float = 0.0
    retrieval_latency_ms: float = 0.0
    is_empty: bool = True

    @classmethod
    def from_items(cls, query_conversation_id: int, items: List[EvidenceItem], latency_ms: float = 0.0) -> "RetrievedEvidence":
        top = max([it.relevance_score for it in items]) if items else 0.0
        return cls(
            query_conversation_id=query_conversation_id,
            items=items,
            top_score=top,
            retrieval_latency_ms=latency_ms,
            is_empty=len(items) == 0
        )


class EscalationDecision(BaseModel):
    conversation_id: int
    routing: RoutingDecision
    triggers: List[str] = Field(default_factory=list)
    intent_confidence: float = Field(ge=0.0, le=1.0)
    evidence_top_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    decision_timestamp: datetime = Field(default_factory=get_utc_now)


class DraftReply(BaseModel):
    conversation_id: int
    draft_text: str
    cited_evidence_ids: List[int] = Field(default_factory=list)
    intent_label: IntentCode
    escalation_routing: RoutingDecision
    escalation_rationale: str = ""
    generation_latency_ms: float = 0.0
    generated_at: datetime = Field(default_factory=get_utc_now)


class GoldenExample(BaseModel):
    golden_id: str
    conversation_id: int
    root_text: str
    true_intent: IntentCode
    true_escalation: RoutingDecision
    reference_reply: Optional[str] = None
    annotator_id: str = "rule_assisted_curated"
    label_notes: Optional[str] = None

    @field_validator("annotator_id")
    @classmethod
    def validate_provenance(cls, v: str) -> str:
        prohibited = ["human_curator", "human_annotator_curated", "human_panel", "hand_labelled"]
        if v in prohibited:
            raise ValueError(f"Misleading annotator_id '{v}'; must reflect rule-assisted curation.")
        return v


class MetricResult(BaseModel):
    metric_name: str
    score: float
    baseline_score: float
    sample_count: int
    passed_target: bool


class EvaluationReport(BaseModel):
    run_id: str
    run_timestamp: datetime = Field(default_factory=get_utc_now)
    benchmark_name: str = "human_benchmark"
    execution_duration_sec: float
    golden_set_size: int
    leakage_check_passed: bool
    headline_metrics: Dict[str, MetricResult]
    per_class_f1: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    failure_mode_analysis: List[Dict[str, Any]] = Field(default_factory=list)
    llm_judge_agreement_rate: float = 0.0


class HumanAnnotationRecord(BaseModel):
    """Schema for independently human-validated evaluation benchmark records."""
    conversation_id: int
    customer_text: str
    created_at: Optional[str] = None
    turns: List[Dict[str, Any]] = Field(default_factory=list)
    intent: Optional[IntentCode] = None
    escalation_required: Optional[bool] = None
    ambiguity: Optional[bool] = None
    annotator_id: Optional[Union[int, str]] = None
    annotation_version: Optional[Union[float, int, str]] = None
    annotation_notes: Optional[str] = None

    @field_validator("annotator_id")
    @classmethod
    def validate_human_annotator(cls, v: Optional[Union[int, str]]) -> Optional[Union[int, str]]:
        if v is None:
            return v
        prohibited_automated = [
            "rule_assisted", "rule_assisted_curated", "automated", "auto",
            "bot", "gemini", "gpt", "llm", "mock", "synthetic", "classifier"
        ]
        if str(v).lower().strip() in prohibited_automated:
            raise ValueError(f"Invalid human annotator_id '{v}': Automated/heuristic agents cannot populate human benchmark.")
        return v


class HumanBenchmarkManifest(BaseModel):
    """Cryptographic and metadata manifest for frozen human-validated benchmark."""
    frozen_at: str
    benchmark_sha256: str
    record_count: int
    conversation_ids_sha256: str
    source_twcs_sha256: str
    temporal_split_date: str = "2017-11-01"
    annotation_version: str
    annotator_ids: List[str] = Field(default_factory=list)
