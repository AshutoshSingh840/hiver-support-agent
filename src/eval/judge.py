"""LLM-as-a-judge scoring module for reply groundedness and human alignment."""

import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from src.models import DraftReply, EvidenceItem


class LLMJudge:
    """Scores draft replies for factual groundedness (1-5 scale) against retrieved evidence."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(model_name)
            except Exception:
                self.client = None

    def evaluate_groundedness(
        self,
        customer_query: str,
        draft_reply: str,
        evidence_items: List[EvidenceItem]
    ) -> Tuple[int, str]:
        """
        Scores groundedness from 1 to 5.
        1: Completely fabricated/hallucinated
        2: Major unsupported claims
        3: Acceptable; no contradictions with evidence
        4: Highly grounded; accurately reflects precedent
        5: Perfectly grounded and directly supported by evidence
        """
        if not draft_reply or not draft_reply.strip():
            return 1, "Empty draft reply"

        if not evidence_items:
            # If no evidence was provided, any affirmative advice is ungrounded
            if "dm" in draft_reply.lower() or "escalat" in draft_reply.lower():
                return 4, "Appropriately suggested DM when no evidence available"
            return 2, "Advice generated without supporting evidence"

        evidence_text = "\n".join([
            f"- Evidence [{item.source_conversation_id}]: Query: '{item.customer_query}' -> Resolution: '{item.agent_resolution}'"
            for item in evidence_items
        ])

        if self.client:
            prompt = f"""You are an objective evaluation judge scoring an AI customer support response for factual groundedness.

Customer Inquiry: "{customer_query}"

Retrieved Historical Support Evidence:
{evidence_text}

AI Generated Draft Reply:
"{draft_reply}"

Scoring Rubric:
1 - Completely hallucinated, contradicts evidence or makes false promises.
2 - Minor relevance, but introduces substantial unsubstantiated claims.
3 - Reasonable support; no direct contradictions with retrieved precedent.
4 - Well-grounded; actions and troubleshooting steps closely match evidence.
5 - Exceptionally grounded; faithfully adheres strictly to historical resolution.

Return ONLY a JSON object with two fields:
{{"score": <integer from 1 to 5>, "rationale": "<brief 1-sentence reason>"}}
"""
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={"temperature": 0.0}
                )
                txt = response.text.strip()
                # Parse JSON
                match = re.search(r"\{.*\}", txt, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    score = int(data.get("score", 3))
                    rationale = data.get("rationale", "Scored by Gemini Judge")
                    return max(1, min(5, score)), rationale
            except Exception as e:
                pass

        # Deterministic Heuristic Fallback (Offline / No API Key)
        # Evaluates token overlap between draft reply and evidence resolutions
        evidence_words = set(re.findall(r"\w+", " ".join([it.agent_resolution.lower() for it in evidence_items])))
        reply_words = set(re.findall(r"\w+", draft_reply.lower()))
        common = evidence_words.intersection(reply_words)
        overlap_ratio = len(common) / max(1, len(reply_words))

        if overlap_ratio > 0.45:
            score = 5
            rationale = f"High lexical grounding ({overlap_ratio:.1%} token overlap with precedent)"
        elif overlap_ratio > 0.30:
            score = 4
            rationale = f"Solid groundedness ({overlap_ratio:.1%} token overlap with precedent)"
        elif overlap_ratio > 0.15:
            score = 3
            rationale = f"Moderate groundedness ({overlap_ratio:.1%} token overlap)"
        else:
            score = 2
            rationale = f"Low lexical alignment ({overlap_ratio:.1%} token overlap with precedent)"

        return score, rationale
