"""Draft reply agent assembling structured metadata and evidence citations."""

import time
from typing import List, Optional
from src.models import IntentCode, RoutingDecision, DraftReply, RetrievedEvidence
from src.generator.prompts import format_generation_prompt
from src.generator.llm_client import LLMClient


class ReplyAgent:
    """Generates grounded customer support draft replies."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate_draft(
        self,
        conversation_id: int,
        customer_query: str,
        intent_label: IntentCode,
        retrieved_evidence: RetrievedEvidence,
        escalation_routing: RoutingDecision = RoutingDecision.AUTO_HANDLE,
        escalation_rationale: str = ""
    ) -> DraftReply:
        """Generates a DraftReply based on customer query and retrieved evidence."""
        start_time = time.time()
        
        evidence_blocks = [
            {
                "id": it.source_conversation_id,
                "query": it.customer_query,
                "resolution": it.agent_resolution
            }
            for it in retrieved_evidence.items
        ]

        prompt = format_generation_prompt(
            customer_query=customer_query,
            intent_label=intent_label.value,
            evidence_blocks=evidence_blocks
        )

        gen_result = self.llm_client.generate(prompt, evidence_blocks)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return DraftReply(
            conversation_id=conversation_id,
            draft_text=gen_result.get("draft_text", ""),
            cited_evidence_ids=gen_result.get("cited_evidence_ids", []),
            intent_label=intent_label,
            escalation_routing=escalation_routing,
            escalation_rationale=gen_result.get("grounding_rationale", escalation_rationale),
            generation_latency_ms=elapsed_ms
        )
