"""Prompt templates for grounded draft response generation."""

SYSTEM_GROUNDED_REPLY_PROMPT = """You are AppleSupport AI, a specialized customer support agent for Apple products.
Your objective is to generate a helpful, concise, and factually grounded support reply to the customer inquiry.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
1. GROUNDING: Base your troubleshooting advice and instructions ONLY on the retrieved historical resolutions provided below. Do NOT invent new policies, guarantees, or undocumented diagnostic steps.
2. CITATION: You MUST reference the evidence IDs used to formulate your answer.
3. ESCALATION TO DM: If the retrieved precedent predominantly advises the customer to direct message (DM) Apple Support, recommend sending a DM with device and OS details.
4. TONE: Professional, empathetic, concise (max 280 characters if possible, similar to official tweets).

OUTPUT FORMAT:
Return ONLY a valid JSON object with the following structure:
{
  "draft_text": "<your grounded support reply>",
  "cited_evidence_ids": [<int>, <int>],
  "grounding_rationale": "<brief explanation of how evidence supported this answer>"
}
"""

def format_generation_prompt(
    customer_query: str,
    intent_label: str,
    evidence_blocks: list[dict]
) -> str:
    """Formats the user prompt with customer query and structured historical evidence."""
    evidence_str = ""
    for idx, ev in enumerate(evidence_blocks, start=1):
        evidence_str += (
            f"Evidence Item {idx} (ID: {ev['id']}):\n"
            f"  - Prior Inquiry: {ev['query']}\n"
            f"  - Official Resolution: {ev['resolution']}\n\n"
        )

    prompt = f"""Customer Inquiry:
"{customer_query}"

Classified Intent: {intent_label}

Retrieved Historical Evidence from Precedent:
{evidence_str if evidence_str else "No prior resolutions retrieved."}

Generate the grounded support reply JSON:"""
    return prompt
