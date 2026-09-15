"""Text normalization utilities for customer tweets and dialogues."""

import html
import re


def normalize_tweet_text(text: str) -> str:
    """
    Decodes HTML entities, strips excess whitespace, and preserves @anonymized user handles.
    
    Examples:
        "&lt;3 @115854 I need help &amp; support" -> "<3 @115854 I need help & support"
    """
    if not text:
        return ""
    
    # 1. Unescape HTML entities (&amp;, &lt;, &gt;, &quot;, &#39;, etc.)
    unescaped = html.unescape(text)
    
    # 2. Normalize repeated whitespace/newlines
    normalized = re.sub(r"\s+", " ", unescaped).strip()
    
    return normalized


def extract_customer_tokens(text: str) -> list[str]:
    """Extracts anonymized customer handles (e.g. '@115854')."""
    return re.findall(r"@\d+", text)
