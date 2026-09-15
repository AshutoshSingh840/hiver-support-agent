"""End-to-end customer support ticket handling pipeline."""

import time
from typing import Dict, Any, Optional
from src.config import AppConfig, default_config
from src.intent.classifier import IntentClassifier
from src.retrieval.engine import RetrievalEngine
from src.escalation.router import EscalationRouter
from src.generator.reply_agent import ReplyAgent
from src.models import (
    IntentPrediction,
    RetrievedEvidence,
    EscalationDecision,
    DraftReply,
    RoutingDecision,
)


class SupportAgentPipeline:
    """End-to-end orchestration pipeline for the Hiver Support Agent."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        retrieval_engine: Optional[RetrievalEngine] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        escalation_router: Optional[EscalationRouter] = None,
        reply_agent: Optional[ReplyAgent] = None
    ):
        self.config = config or default_config
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.retrieval_engine = retrieval_engine or RetrievalEngine(
            min_relevance_score=self.config.retrieval.min_relevance_score
        )
        self.escalation_router = escalation_router or EscalationRouter(
            config=self.config.escalation
        )
        self.reply_agent = reply_agent or ReplyAgent()

    def process_ticket(self, conversation_id: int, text: str) -> Dict[str, Any]:
        """
        Processes a single customer inquiry through all stages.
        Returns a complete structured decision trace.
        """
        start_time = time.time()

        # 1. Intent Classification
        intent_pred: IntentPrediction = self.intent_classifier.predict(conversation_id, text)

        # 2. Evidence Retrieval
        evidence: RetrievedEvidence = self.retrieval_engine.retrieve(
            conversation_id=conversation_id,
            query_text=text,
            query_intent=intent_pred.label,
            top_k=self.config.retrieval.top_k
        )

        # 3. Escalation Routing
        decision: EscalationDecision = self.escalation_router.evaluate(
            conversation_id=conversation_id,
            query_text=text,
            intent_prediction=intent_pred,
            retrieved_evidence=evidence
        )

        # 4. Draft Reply Generation (if auto-handled or for preview)
        draft_reply: Optional[DraftReply] = None
        if decision.routing == RoutingDecision.AUTO_HANDLE:
            draft_reply = self.reply_agent.generate_draft(
                conversation_id=conversation_id,
                customer_query=text,
                intent_label=intent_pred.label,
                retrieved_evidence=evidence,
                escalation_routing=decision.routing,
                escalation_rationale=decision.rationale
            )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "conversation_id": conversation_id,
            "input_text": text,
            "intent": {
                "label": intent_pred.label.value,
                "confidence": intent_pred.confidence,
                "supporting_excerpt": intent_pred.supporting_excerpt
            },
            "retrieval": {
                "top_k": len(evidence.items),
                "top_score": evidence.top_score,
                "evidence": [
                    {
                        "source_conversation_id": it.source_conversation_id,
                        "relevance_score": it.relevance_score,
                        "resolution_text": it.agent_resolution,
                        "is_dm": it.is_dm_escalation
                    }
                    for it in evidence.items
                ]
            },
            "retrieval_items": evidence.items,
            "escalation": {
                "routing": decision.routing.value,
                "risk_score": decision.risk_score,
                "routing_threshold": decision.routing_threshold,
                "triggers": decision.triggers,
                "rationale": decision.rationale
            },
            "draft_reply": {
                "draft_text": draft_reply.draft_text,
                "cited_evidence_ids": draft_reply.cited_evidence_ids
            } if draft_reply else None,
            "latency_ms": elapsed_ms
        }
