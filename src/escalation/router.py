import re
from typing import List, Optional
from src.models import (
    IntentPrediction,
    RetrievedEvidence,
    RoutingDecision,
    EscalationDecision,
    IntentCode,
)
from src.config import EscalationConfig, default_config


def _is_non_english(text: str) -> bool:
    """Detects whether text requires non-English language support routing."""
    if not text:
        return False
    # Check for CJK (Japanese/Chinese/Korean), Cyrillic, Arabic scripts
    cjk_or_foreign = re.findall(
        r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u0400-\u04FF\u0600-\u06FF]',
        text
    )
    if len(cjk_or_foreign) >= 3:
        return True
    non_ascii = re.findall(r'[^\x00-\x7F]', text)
    return len(non_ascii) / max(len(text), 1) > 0.25


SEVERE_FAILURE_PATTERNS = [
    # Critical data & photo loss
    (r"(photos?|contacts?|data|pictures?|files?|messages?|music) (disappeared|lost|missing|gone|deleted|vanished|wiped|stopped syncing)", "critical_data_or_media_loss"),
    (r"(lost|missing|deleted|wiped) (all|most) (of my |my )?(photos?|contacts?|data|pictures?|files?|messages?|music)", "critical_data_or_media_loss"),
    (r"photos? (are )?not loading from icloud", "critical_data_or_media_loss"),
    # Device bricked / restore / boot loop failure
    (r"\b(bricked|bricking)\b", "device_bricked_or_restore_failure"),
    (r"boot ?loop", "device_bricked_or_restore_failure"),
    (r"stuck on (apple logo|black screen|loading screen|spinning wheel)", "device_bricked_or_restore_failure"),
    (r"unable to restore|restore (failed|error|keeps saying)", "device_bricked_or_restore_failure"),
    # Failed support attempts / channel bouncebacks / warranty rejections
    (r"(went to|visited|called) (apple store|support|genius bar).*?(can'?t do anything|couldn'?t help|refused|declined|no help|standard)", "prior_support_channel_exhausted"),
    (r"support (ticket|request|email) (bounced|bounceback|full|rejected)", "support_channel_bounceback_failure"),
    (r"support request .*? (bounceback|bounced)", "support_channel_bounceback_failure"),
    (r"warranty.*?(expired|denied)", "warranty_denial_unresolved_case")
]


class EscalationRouter:
    """Evaluates case confidence, safety triggers, and evidence quality to route tickets."""

    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config or default_config.escalation

    def evaluate(
        self,
        conversation_id: int,
        query_text: str,
        intent_prediction: IntentPrediction,
        retrieved_evidence: RetrievedEvidence
    ) -> EscalationDecision:
        """Evaluates whether to auto-handle or escalate."""
        triggers: List[str] = []
        lower_query = query_text.lower()

        # 1. Out-of-scope / Unclear intent trigger (calibrated: only escalate if lacking domain evidence or non-support)
        if intent_prediction.label == IntentCode.INT_OUT_OF_SCOPE:
            if retrieved_evidence.is_empty or retrieved_evidence.top_score < self.config.retrieval_min_score:
                triggers.append("out_of_scope_or_unclear_intent")
            elif intent_prediction.supporting_excerpt == "No specific support domain keyword matched" and intent_prediction.confidence >= 0.80:
                triggers.append("out_of_scope_or_unclear_intent")

        # 2. Non-English language support trigger
        if _is_non_english(query_text):
            triggers.append("non_english_language_support_needed")

        # 3. Low intent classifier confidence trigger
        if intent_prediction.confidence < self.config.intent_confidence_threshold:
            triggers.append(
                f"low_intent_confidence ({intent_prediction.confidence:.2f} < {self.config.intent_confidence_threshold:.2f})"
            )

        # 4. Insufficient retrieval evidence trigger
        if retrieved_evidence.is_empty or retrieved_evidence.top_score < self.config.retrieval_min_score:
            triggers.append(
                f"insufficient_retrieval_evidence (top_score={retrieved_evidence.top_score:.2f} < {self.config.retrieval_min_score:.2f})"
            )

        # 5. Sensitive domain / safety triggers
        for kw in self.config.sensitive_keywords:
            if " " in kw:
                if kw in lower_query:
                    triggers.append(f"sensitive_safety_trigger ('{kw}')")
                    break
            else:
                if re.search(rf"\b{re.escape(kw)}", lower_query):
                    triggers.append(f"sensitive_safety_trigger ('{kw}')")
                    break

        # 6. Explicit severe failure / data loss / bricking / bounceback patterns
        for pattern, reason in SEVERE_FAILURE_PATTERNS:
            if re.search(pattern, lower_query):
                triggers.append(f"severe_unresolved_failure_trigger ('{reason}')")
                break

        # 7. DM-only precedent trigger (calibrated for ambiguity)
        if retrieved_evidence.items and all(it.is_dm_escalation for it in retrieved_evidence.items):
            if intent_prediction.confidence < 0.85:
                triggers.append("historical_precedent_mandates_dm_escalation")

        # Decision routing
        is_escalate = len(triggers) > 0
        routing = RoutingDecision.ESCALATE if is_escalate else RoutingDecision.AUTO_HANDLE
        rationale = "; ".join(triggers) if triggers else "Passed all auto-handling confidence and safety criteria."

        return EscalationDecision(
            conversation_id=conversation_id,
            routing=routing,
            triggers=triggers,
            intent_confidence=intent_prediction.confidence,
            evidence_top_score=retrieved_evidence.top_score,
            rationale=rationale
        )


