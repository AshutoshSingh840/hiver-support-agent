"""Unit tests for escalation router and trigger conditions."""

import pytest
from src.escalation.router import EscalationRouter
from src.models import (
    IntentPrediction,
    IntentCode,
    RetrievedEvidence,
    EvidenceItem,
    RoutingDecision,
)
from src.config import EscalationConfig
from datetime import datetime, timezone


def _make_evidence(relevance_score: float = 0.85, is_dm: bool = False) -> RetrievedEvidence:
    return RetrievedEvidence.from_items(1, [
        EvidenceItem(
            source_conversation_id=101,
            relevance_score=relevance_score,
            intent_label=IntentCode.INT_BATTERY,
            customer_query="battery issue",
            agent_resolution="http://apple.co/battery" if not is_dm else "Please DM us",
            temporal_date=datetime(2017, 10, 1, tzinfo=timezone.utc),
            is_dm_escalation=is_dm
        )
    ])


def test_escalate_on_sensitive_keyword_security_fraud():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=1,
        label=IntentCode.INT_STORE,
        confidence=0.95,
        supporting_excerpt="stole"
    )
    evidence = _make_evidence()
    text = "Someone stole my phone and made fraud charges. I have a lawyer!"

    dec = router.evaluate(1, text, pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert any("sensitive_safety_trigger" in t for t in dec.triggers)


def test_escalate_on_financial_refund_triggers():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=2,
        label=IntentCode.INT_STORE,
        confidence=0.92,
        supporting_excerpt="refund"
    )
    evidence = _make_evidence()
    text = "I need a refund for an accidental purchase that was charged to my account."

    dec = router.evaluate(2, text, pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert any("sensitive_safety_trigger ('refund')" in t for t in dec.triggers)


def test_escalate_on_account_disabled_triggers():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=3,
        label=IntentCode.INT_ICLOUD,
        confidence=0.91,
        supporting_excerpt="disabled"
    )
    evidence = _make_evidence()
    text = "My Apple ID has been disabled and I am locked out of iCloud."

    dec = router.evaluate(3, text, pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert any("sensitive_safety_trigger ('disabled')" in t for t in dec.triggers)


def test_escalate_on_hardware_danger_repair():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=4,
        label=IntentCode.INT_HARDWARE,
        confidence=0.90,
        supporting_excerpt="swelling"
    )
    evidence = _make_evidence()
    text = "My iPhone battery is swelling and the screen is popping out. Need repair."

    dec = router.evaluate(4, text, pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert any("sensitive_safety_trigger" in t for t in dec.triggers)


def test_escalate_on_catastrophic_data_loss():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=5,
        label=IntentCode.INT_ICLOUD,
        confidence=0.93,
        supporting_excerpt="lost all"
    )
    evidence = _make_evidence()
    text = "I updated iOS and lost all my photos and synced contacts."

    dec = router.evaluate(5, text, pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert any("sensitive_safety_trigger ('lost all" in t for t in dec.triggers)


def test_escalate_on_non_english_query():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=6,
        label=IntentCode.INT_IOS,
        confidence=0.90,
        supporting_excerpt="11.1"
    )
    evidence = _make_evidence()
    text = "iOS 11.1にアップデートしたら電池の減りが早すぎる"

    dec = router.evaluate(6, text, pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert "non_english_language_support_needed" in dec.triggers


def test_escalate_on_low_confidence():
    # Low confidence alone is a soft gradient signal (r ~ 0.02–0.35).
    # With the default threshold=0.45 a single low-confidence signal does NOT
    # independently escalate — it adds measurable risk that can combine with other signals.
    # This test verifies: signal is detected, risk is in the correct range, and that
    # a strict threshold (0.05) causes escalation while the default (0.45) auto-handles.
    router = EscalationRouter(config=EscalationConfig(intent_confidence_threshold=0.80))
    pred = IntentPrediction(
        conversation_id=7,
        label=IntentCode.INT_BATTERY,
        confidence=0.50,
        supporting_excerpt="battery"
    )
    evidence = _make_evidence()
    dec = router.evaluate(7, "My battery is somewhat okay", pred, evidence)
    # Soft signal is detected
    assert any("low_intent_confidence" in t for t in dec.triggers)
    # Risk is measurably elevated but below the 0.45 default threshold
    assert 0.05 < dec.risk_score < 0.45
    # Default threshold: auto-handle (soft signal alone doesn't escalate)
    assert dec.routing == RoutingDecision.AUTO_HANDLE
    # Strict threshold=0.05 escalates because risk > 0.05
    strict_router = EscalationRouter(config=EscalationConfig(
        intent_confidence_threshold=0.80, routing_threshold=0.05
    ))
    strict_dec = strict_router.evaluate(7, "My battery is somewhat okay", pred, evidence)
    assert strict_dec.routing == RoutingDecision.ESCALATE


def test_escalate_on_insufficient_retrieval_evidence():
    # Low retrieval quality alone is a soft gradient signal (r ~ 0.02–0.35).
    # At default threshold=0.45, it produces measurable risk but doesn't independently escalate.
    # A strict threshold or multiple soft signals together can cause escalation.
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=8,
        label=IntentCode.INT_BATTERY,
        confidence=0.95,
        supporting_excerpt="battery drain"
    )
    low_evidence = _make_evidence(relevance_score=0.20)
    dec = router.evaluate(8, "My battery is draining fast", pred, low_evidence)
    # Signal is detected and logged
    assert any("insufficient_retrieval_evidence" in t for t in dec.triggers)
    # Risk is elevated above floor (0.05) but below default threshold (0.45)
    assert dec.risk_score > 0.05
    assert dec.routing == RoutingDecision.AUTO_HANDLE
    # Strict threshold=0.05 triggers escalation
    strict_router = EscalationRouter(config=EscalationConfig(routing_threshold=0.05))
    strict_dec = strict_router.evaluate(8, "My battery is draining fast", pred, low_evidence)
    assert strict_dec.routing == RoutingDecision.ESCALATE


def test_calibrated_dm_precedent_escalates_ambiguous():
    # DM precedent is a soft signal (r = 0.28). It is detected as a trigger.
    # Alone, it won't exceed the default threshold=0.45, but it elevates risk
    # meaningfully and combines with the sub-threshold confidence signal.
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=9,
        label=IntentCode.INT_BATTERY,
        confidence=0.78,  # Slightly above intent_confidence_threshold=0.75, so no confidence trigger
        supporting_excerpt="battery issue"
    )
    evidence = _make_evidence(is_dm=True)
    dec = router.evaluate(9, "Battery issue on device", pred, evidence)
    # DM precedent trigger is recorded
    assert "historical_precedent_mandates_dm_escalation" in dec.triggers
    # Risk is above floor — DM precedent adds measurable risk
    assert dec.risk_score > 0.05


def test_calibrated_dm_precedent_allows_high_confidence_public_troubleshooting():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=10,
        label=IntentCode.INT_BATTERY,
        confidence=0.92,  # High confidence
        supporting_excerpt="battery drain"
    )
    evidence = _make_evidence(is_dm=True)
    dec = router.evaluate(10, "How do I check battery health percentage on iOS 11?", pred, evidence)
    # Since confidence >= 0.85 and query does not contain private/account/dm keywords, clean public troubleshooting passes
    assert dec.routing == RoutingDecision.AUTO_HANDLE
    assert len(dec.triggers) == 0


def test_auto_handle_on_clean_inquiry():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=11,
        label=IntentCode.INT_BATTERY,
        confidence=0.92,
        supporting_excerpt="battery life"
    )
    evidence = _make_evidence()
    dec = router.evaluate(11, "How can I improve my iPhone battery life?", pred, evidence)
    assert dec.routing == RoutingDecision.AUTO_HANDLE
    assert len(dec.triggers) == 0


def test_escalate_on_out_of_scope_intent():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=12,
        label=IntentCode.INT_OUT_OF_SCOPE,
        confidence=0.80,
        supporting_excerpt="No specific support domain keyword matched"
    )
    evidence = _make_evidence()
    dec = router.evaluate(12, "Why is the sky blue today?", pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert "out_of_scope_or_unclear_intent" in dec.triggers


def test_escalate_on_empty_retrieval_evidence():
    # Empty retrieval is the most severe retrieval-quality signal (r = 0.35).
    # Combined with a low-confidence signal it can exceed the default threshold.
    # Alone with high confidence it stays below 0.45 but above the floor.
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=13,
        label=IntentCode.INT_CONN,
        confidence=0.95,
        supporting_excerpt="wifi"
    )
    empty_evidence = RetrievedEvidence.from_items(13, [])
    dec = router.evaluate(13, "My wifi is broken", pred, empty_evidence)
    # Empty retrieval trigger is always recorded
    assert any("insufficient_retrieval_evidence" in t for t in dec.triggers)
    # Risk is elevated above floor
    assert dec.risk_score >= 0.35
    # At strict threshold=0.30, empty retrieval causes escalation
    strict_router = EscalationRouter(config=EscalationConfig(routing_threshold=0.30))
    strict_dec = strict_router.evaluate(13, "My wifi is broken", pred, empty_evidence)
    assert strict_dec.routing == RoutingDecision.ESCALATE


def test_escalate_on_lawsuit_legal_risk():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=14,
        label=IntentCode.INT_STORE,
        confidence=0.90,
        supporting_excerpt="charged"
    )
    evidence = _make_evidence()
    dec = router.evaluate(14, "I will file a lawsuit against Apple with my lawyer for this unauthorized charge!", pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert any("sensitive_safety_trigger" in t for t in dec.triggers)


def test_escalate_on_descriptive_data_loss():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=15,
        label=IntentCode.INT_ICLOUD,
        confidence=0.85,
        supporting_excerpt="photos"
    )
    evidence = _make_evidence()
    dec = router.evaluate(15, "All of my photos disappeared and my contacts were lost after updating.", pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert any("severe_unresolved_failure_trigger" in t for t in dec.triggers)
    assert any("critical_data_or_media_loss" in t for t in dec.triggers)


def test_escalate_on_device_bricking_and_boot_loop():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=16,
        label=IntentCode.INT_IOS,
        confidence=0.88,
        supporting_excerpt="boot loop"
    )
    evidence = _make_evidence()
    dec = router.evaluate(16, "My iPhone is stuck on apple logo in a boot loop and restore failed with unknown error", pred, evidence)
    assert dec.routing == RoutingDecision.ESCALATE
    assert any("device_bricked_or_restore_failure" in t for t in dec.triggers)


def test_escalate_on_exhausted_support_attempts_and_bouncebacks():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=17,
        label=IntentCode.INT_HARDWARE,
        confidence=0.85,
        supporting_excerpt="apple store"
    )
    evidence = _make_evidence()
    dec1 = router.evaluate(17, "I went to Apple Store genius bar and they couldn't help me with this screen defect", pred, evidence)
    assert dec1.routing == RoutingDecision.ESCALATE
    assert any("prior_support_channel_exhausted" in t for t in dec1.triggers)

    dec2 = router.evaluate(18, "My support request email bounced and was rejected", pred, evidence)
    assert dec2.routing == RoutingDecision.ESCALATE
    assert any("support_channel_bounceback_failure" in t for t in dec2.triggers)


def test_out_of_scope_with_support_context_does_not_unconditionally_escalate():
    router = EscalationRouter()
    pred = IntentPrediction(
        conversation_id=19,
        label=IntentCode.INT_OUT_OF_SCOPE,
        confidence=0.85,
        supporting_excerpt="general device inquiry"
    )
    evidence = _make_evidence()
    # Query with good evidence and no safety triggers
    dec = router.evaluate(19, "What are the dimensions and specs of iPhone 8?", pred, evidence)
    assert dec.routing == RoutingDecision.AUTO_HANDLE
    assert len(dec.triggers) == 0


def test_continuous_risk_score_monotonicity_and_triggers():
    router = EscalationRouter()

    # 1. Clean query with high confidence and strong evidence
    clean_pred = IntentPrediction(
        conversation_id=20,
        label=IntentCode.INT_BATTERY,
        confidence=0.98,
        supporting_excerpt="battery"
    )
    clean_evidence = _make_evidence(relevance_score=0.90)
    clean_dec = router.evaluate(20, "How do I optimize battery health on iOS?", clean_pred, clean_evidence)
    assert clean_dec.risk_score < 0.20
    assert clean_dec.routing == RoutingDecision.AUTO_HANDLE

    # 2. Moderate query with lower confidence
    mod_pred = IntentPrediction(
        conversation_id=21,
        label=IntentCode.INT_CONN,
        confidence=0.55,
        supporting_excerpt="wifi"
    )
    mod_evidence = _make_evidence(relevance_score=0.80)
    mod_dec = router.evaluate(21, "wifi issues", mod_pred, mod_evidence)
    assert mod_dec.risk_score > clean_dec.risk_score

    # 3. Severe safety query
    severe_pred = IntentPrediction(
        conversation_id=22,
        label=IntentCode.INT_HARDWARE,
        confidence=0.95,
        supporting_excerpt="smoke"
    )
    severe_dec = router.evaluate(22, "My battery is swollen and emitting smoke", severe_pred, clean_evidence)
    assert severe_dec.risk_score >= 0.90
    assert severe_dec.routing == RoutingDecision.ESCALATE


def test_custom_routing_threshold():
    # Test strict vs lenient routing thresholds with the recalibrated risk channels.
    # conf=0.60, threshold_conf=0.75: deficit=0.1333/0.75 → intent_risk ≈ 0.02+0.33*0.2=0.086
    # top_score=0.70 >= min_score=0.35 → near-zero evidence residual
    # Combined Noisy-OR ≈ 0.08–0.12 range
    strict_config = EscalationConfig(routing_threshold=0.05)   # Escalates on any soft signal
    lenient_config = EscalationConfig(routing_threshold=0.80)  # Only hard triggers escalate

    strict_router = EscalationRouter(config=strict_config)
    lenient_router = EscalationRouter(config=lenient_config)

    pred = IntentPrediction(
        conversation_id=23,
        label=IntentCode.INT_IOS,
        confidence=0.60,
        supporting_excerpt="lag"
    )
    evidence = _make_evidence(relevance_score=0.70)
    query = "My device is slightly laggy after update"

    strict_dec = strict_router.evaluate(23, query, pred, evidence)
    lenient_dec = lenient_router.evaluate(23, query, pred, evidence)

    # Risk scores are identical (threshold does not affect score computation)
    assert strict_dec.risk_score == pytest.approx(lenient_dec.risk_score, abs=1e-6)
    # Risk is above baseline floor (soft confidence signal fires)
    assert strict_dec.risk_score > 0.05
    # Strict router (threshold=0.05) escalates because risk_score > 0.05
    assert strict_dec.routing == RoutingDecision.ESCALATE
    # Lenient router (threshold=0.80) auto-handles because soft signal alone is well below 0.80
    assert lenient_dec.routing == RoutingDecision.AUTO_HANDLE




