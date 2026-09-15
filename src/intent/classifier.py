"""Intent classification engine with calibrated confidence scoring and excerpt extraction."""

import re
from typing import Dict, List, Tuple, Optional
from src.models import IntentCode, IntentPrediction
from src.intent.taxonomy import INTENT_DEFINITIONS


class IntentClassifier:
    """Classifies customer inquiries into the 8-class taxonomy with calibrated confidence."""

    def __init__(
        self,
        ambiguity_threshold: float = 0.50,
        primary_weight: float = 2.5,
        secondary_weight: float = 0.8
    ):
        self.ambiguity_threshold = ambiguity_threshold
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
        self.patterns = {}
        for code, defs in INTENT_DEFINITIONS.items():
            prim_list = defs.get("primary_keywords", defs.get("keywords", []))
            sec_list = defs.get("secondary_keywords", [])
            self.patterns[code] = {
                "primary": [re.compile(p, re.IGNORECASE) for p in prim_list],
                "secondary": [re.compile(p, re.IGNORECASE) for p in sec_list],
            }

    def predict(self, conversation_id: int, text: str) -> IntentPrediction:
        """Predicts intent label, calibrated confidence, and supporting excerpt."""
        if not text or not text.strip():
            return IntentPrediction(
                conversation_id=conversation_id,
                label=IntentCode.INT_OUT_OF_SCOPE,
                confidence=1.0,
                supporting_excerpt="Empty query"
            )

        lower_text = text.lower()
        scores: Dict[IntentCode, float] = {code: 0.0 for code in IntentCode}
        matched_excerpts: Dict[IntentCode, List[str]] = {code: [] for code in IntentCode}

        for code, pattern_group in self.patterns.items():
            for regex in pattern_group["primary"]:
                matches = regex.findall(lower_text)
                if matches:
                    scores[code] += self.primary_weight * len(matches)
                    m = regex.search(text)
                    if m:
                        start = max(0, m.start() - 15)
                        end = min(len(text), m.end() + 15)
                        matched_excerpts[code].append(text[start:end].strip())

            for regex in pattern_group["secondary"]:
                matches = regex.findall(lower_text)
                if matches:
                    scores[code] += self.secondary_weight * len(matches)
                    if not matched_excerpts[code]:
                        m = regex.search(text)
                        if m:
                            start = max(0, m.start() - 15)
                            end = min(len(text), m.end() + 15)
                            matched_excerpts[code].append(text[start:end].strip())

        # Find best candidate
        best_code = IntentCode.INT_OUT_OF_SCOPE
        best_score = 0.0
        second_best_score = 0.0

        for code, score in scores.items():
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_code = code
            elif score > second_best_score:
                second_best_score = score

        if best_score == 0.0:
            return IntentPrediction(
                conversation_id=conversation_id,
                label=IntentCode.INT_OUT_OF_SCOPE,
                confidence=0.80,
                supporting_excerpt="No specific support domain keyword matched"
            )

        # Calibrated confidence calculation based on score margin and total hits
        total_hits = sum(scores.values())
        dominance = best_score / total_hits if total_hits > 0 else 1.0

        # Base confidence from 0.70 to 0.98
        confidence = min(0.98, 0.65 + 0.20 * dominance + 0.05 * min(3, int(best_score)))
        if second_best_score > 0 and (best_score - second_best_score) < 0.5:
            confidence = 0.50  # Ambiguous query flag

        excerpt = matched_excerpts[best_code][0] if matched_excerpts[best_code] else text[:40]

        return IntentPrediction(
            conversation_id=conversation_id,
            label=best_code,
            confidence=round(confidence, 2),
            supporting_excerpt=excerpt
        )

