import re
from typing import List, Optional, Tuple
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


FRUSTRATION_AND_RETRY_PATTERNS = [
    (r"(already|have) (tried|done) (that|everything|all of that|restarting|resetting)", "exhausted_self_troubleshooting"),
    (r"(still|keeps?) (not working|failing|crashing|freezing|dropping)", "persistent_unresolved_issue"),
    (r"(third|fourth|3rd|4th|5th|multiple) time", "repeated_failure_cycle"),
    (r"(tired|fed up|sick) of (this|apple)", "customer_escalation_sentiment"),
    (r"(fix this|pls fix|please fix|fix it now|when y'?all gonna|address the fact|damn glitch|notimpressed|not impressed|annoyed|blowing mine)", "customer_demand_or_rant"),
    (r"(why do i always|why is it impossible|got people out here)", "rhetorical_complaint_or_rant"),
]


class EscalationRouter:
    """Evaluates case confidence, safety triggers, and evidence quality using continuous risk scoring."""

    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config or default_config.escalation

    def compute_risk_score(
        self,
        query_text: str,
        intent_prediction: IntentPrediction,
        retrieved_evidence: RetrievedEvidence
    ) -> Tuple[float, List[str]]:
        """
        Computes a continuous probabilistic risk score in [0.0, 1.0] across all risk channels.
        Uses a noisy-OR formulation: Risk = 1 - product(1 - r_i).

        Channel weight design (two tiers):
          Hard/Safety triggers  → r in [0.72, 0.92]  (sensitive keywords, data-loss, non-English, out-of-scope)
          Soft/Gradient signals → r in [0.02, 0.35]  (confidence deficit, retrieval quality, DM precedent, frustration)

        Expected score ranges:
          Clean query (high conf, good retrieval)      → risk ~ 0.05–0.15
          One soft signal (moderate confidence drop)   → risk ~ 0.10–0.30
          Multiple soft signals                        → risk ~ 0.30–0.55
          Single hard trigger (sensitive keyword)      → risk ~ 0.72–0.92
          Hard + soft combination                      → risk ~ 0.85–0.97
        """
        triggers: List[str] = []
        risk_components: List[float] = []
        lower_query = query_text.lower() if query_text else ""

        # ── TIER 1: Hard / Safety Triggers ───────────────────────────────────────

        # 1. Critical safety / sensitive keyword triggers (r = 0.92)
        for kw in self.config.sensitive_keywords:
            if " " in kw:
                if kw in lower_query:
                    triggers.append(f"sensitive_safety_trigger ('{kw}')")
                    risk_components.append(0.92)
                    break
            else:
                if re.search(rf"\b{re.escape(kw)}", lower_query):
                    triggers.append(f"sensitive_safety_trigger ('{kw}')")
                    risk_components.append(0.92)
                    break

        # 2. Severe failure / data loss / bricking patterns (r = 0.88)
        for pattern, reason in SEVERE_FAILURE_PATTERNS:
            if re.search(pattern, lower_query):
                triggers.append(f"severe_unresolved_failure_trigger ('{reason}')")
                risk_components.append(0.88)
                break

        # 3. Non-English language detection (r = 0.92)
        if _is_non_english(query_text):
            triggers.append("non_english_language_support_needed")
            risk_components.append(0.92)

        # 4. Out-of-scope intent (r in [0.72, 0.82])
        if intent_prediction.label == IntentCode.INT_OUT_OF_SCOPE:
            if retrieved_evidence.is_empty or retrieved_evidence.top_score < self.config.retrieval_min_score:
                triggers.append("out_of_scope_or_unclear_intent")
                risk_components.append(0.82)
            elif intent_prediction.supporting_excerpt == "No specific support domain keyword matched":
                triggers.append("out_of_scope_or_unclear_intent")
                risk_components.append(0.72)

        # ── TIER 2: Soft / Gradient Signals ──────────────────────────────────────

        # 5. Intent Classifier Confidence Gradient (continuous risk in [0.02, 0.35])
        #    The keyword classifier legitimately operates in [0.40, 0.80]; moderate sub-threshold
        #    confidence contributes modest risk, not catastrophic risk.
        #    deficit fraction mapped linearly into [0.02, 0.35].
        conf = intent_prediction.confidence
        if conf < self.config.intent_confidence_threshold:
            deficit = (self.config.intent_confidence_threshold - conf) / max(0.01, self.config.intent_confidence_threshold)
            intent_risk = round(0.02 + 0.33 * deficit, 4)
            triggers.append(f"low_intent_confidence ({conf:.2f} < {self.config.intent_confidence_threshold:.2f})")
            risk_components.append(intent_risk)
        elif conf < 1.0:
            # Very slight residual risk for non-maximal confidence
            residual = max(0.0, 0.05 * (1.0 - conf))
            risk_components.append(residual)

        # 6. Retrieval Evidence Quality Gradient (continuous risk in [0.02, 0.35])
        #    BM25 scores for a 29k corpus typically range [0.05, 0.50]; min_score=0.35 is the quality bar.
        #    Missing/low retrieval is a soft signal.
        #    deficit fraction mapped linearly into [0.02, 0.35].
        if retrieved_evidence.is_empty:
            triggers.append("insufficient_retrieval_evidence (empty index result)")
            risk_components.append(0.35)
        else:
            top_score = retrieved_evidence.top_score
            min_score = self.config.retrieval_min_score
            if top_score < min_score:
                deficit_ratio = (min_score - top_score) / max(0.01, min_score)
                evidence_risk = round(0.02 + 0.33 * deficit_ratio, 4)
                triggers.append(f"insufficient_retrieval_evidence (top_score={top_score:.2f} < {min_score:.2f})")
                risk_components.append(evidence_risk)
            else:
                # Slight positive risk proportional to closeness to the quality floor
                quality_margin = (top_score - min_score) / max(0.01, 1.0 - min_score)
                residual = max(0.0, 0.06 * (1.0 - min(1.0, quality_margin)))
                risk_components.append(residual)

        # 7. DM Precedent Risk (r = 0.28 — soft signal: all precedents were DM escalations)
        if retrieved_evidence.items and all(it.is_dm_escalation for it in retrieved_evidence.items):
            if intent_prediction.confidence < 0.85:
                triggers.append("historical_precedent_mandates_dm_escalation")
                risk_components.append(0.28)

        # 8. Frustration & repeated failure patterns (r = 0.22)
        for pattern, reason in FRUSTRATION_AND_RETRY_PATTERNS:
            if re.search(pattern, lower_query):
                triggers.append(f"customer_frustration_or_retry ('{reason}')")
                risk_components.append(0.22)
                break

        # ── Probabilistic Aggregation (Noisy-OR) ─────────────────────────────────
        if not risk_components:
            risk_score = 0.05  # baseline floor: no signals detected
        else:
            prob_safe = 1.0
            for r in risk_components:
                prob_safe *= (1.0 - min(0.99, max(0.0, r)))
            risk_score = min(1.0, max(0.0, 1.0 - prob_safe))

        return float(round(risk_score, 4)), triggers

    def evaluate(
        self,
        conversation_id: int,
        query_text: str,
        intent_prediction: IntentPrediction,
        retrieved_evidence: RetrievedEvidence
    ) -> EscalationDecision:
        """Evaluates whether to auto-handle or escalate based solely on continuous risk score >= threshold.

        The risk score already encodes all trigger severity via Noisy-OR (sensitive/severe triggers
        contribute 0.90–0.95 to the score), so a separate hard-override bypass is unnecessary and
        would make the threshold sweep non-informative.
        """
        risk_score, triggers = self.compute_risk_score(query_text, intent_prediction, retrieved_evidence)
        threshold = self.config.routing_threshold

        is_escalate = risk_score >= threshold
        routing = RoutingDecision.ESCALATE if is_escalate else RoutingDecision.AUTO_HANDLE

        if triggers:
            rationale = (f"Risk Score {risk_score:.4f} >= Threshold {threshold:.2f} -> ESCALATE (Triggers: {'; '.join(triggers)})" if is_escalate
                         else f"Risk Score {risk_score:.4f} < Threshold {threshold:.2f} -> AUTO-HANDLE (Notes: {'; '.join(triggers)})")
        else:
            rationale = f"Risk Score {risk_score:.4f} < Threshold {threshold:.2f} -> AUTO-HANDLE. No risk signals detected."

        return EscalationDecision(
            conversation_id=conversation_id,
            routing=routing,
            risk_score=risk_score,
            routing_threshold=threshold,
            triggers=triggers,
            intent_confidence=intent_prediction.confidence,
            evidence_top_score=retrieved_evidence.top_score,
            rationale=rationale
        )


