"""Retrieval engine with metadata lookup and intent-aware filtering."""

import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.models import IntentCode, EvidenceItem, RetrievedEvidence
from src.retrieval.bm25 import BM25Index
from src.data.models import Conversation


class RetrievalEngine:
    """Retrieves top-k historically resolved conversations grounded as evidence."""

    def __init__(self, min_relevance_score: float = 0.35):
        self.min_relevance_score = min_relevance_score
        self.index: Optional[BM25Index] = None
        self.conversation_lookup: Dict[int, Conversation] = {}
        self.conversation_intents: Dict[int, IntentCode] = {}

    def build_from_conversations(
        self,
        conversations: List[Conversation],
        intent_classifier_fn = None
    ):
        """Builds in-memory index from a list of historical conversations."""
        self.conversation_lookup = {c.conversation_id: c for c in conversations}
        docs = []
        for conv in conversations:
            text = f"{conv.root_text} {conv.agent_resolution_text}"
            docs.append((conv.conversation_id, text))
            if intent_classifier_fn:
                pred = intent_classifier_fn(conv.conversation_id, conv.root_text)
                self.conversation_intents[conv.conversation_id] = pred.label

        self.index = BM25Index()
        self.index.fit(docs)

    def retrieve(
        self,
        conversation_id: int,
        query_text: str,
        query_intent: IntentCode = IntentCode.INT_OUT_OF_SCOPE,
        top_k: int = 5
    ) -> RetrievedEvidence:
        """Retrieves top-k relevant historically resolved conversations with soft intent preference."""
        start_time = time.time()
        if not self.index or not query_text.strip():
            return RetrievedEvidence(query_conversation_id=conversation_id, items=[], top_score=0.0)

        # Search BM25 over wider candidate pool (top_k * 6) for soft intent re-ranking
        raw_results = self.index.search(query_text, top_k=max(30, top_k * 6))
        if not raw_results:
            return RetrievedEvidence(query_conversation_id=conversation_id, items=[], top_score=0.0)

        max_bm25 = raw_results[0][1]
        candidates: List[EvidenceItem] = []

        for doc_id, raw_score in raw_results:
            conv = self.conversation_lookup.get(doc_id)
            if not conv or not conv.agent_resolution_text:
                continue

            # Normalized BM25 score
            norm_score = min(1.0, raw_score / max(1.0, max_bm25 * 1.1))
            
            # Soft Intent boost preference (boosts matching intent without excluding other intents)
            doc_intent = self.conversation_intents.get(doc_id, IntentCode.INT_OUT_OF_SCOPE)
            if query_intent != IntentCode.INT_OUT_OF_SCOPE and doc_intent == query_intent:
                norm_score = min(1.0, norm_score * 1.25)

            if norm_score >= self.min_relevance_score:
                candidates.append(
                    EvidenceItem(
                        source_conversation_id=doc_id,
                        relevance_score=round(norm_score, 3),
                        intent_label=doc_intent,
                        customer_query=conv.root_text,
                        agent_resolution=conv.agent_resolution_text,
                        is_dm_escalation=conv.is_dm_escalation,
                        temporal_date=conv.created_at
                    )
                )

        # Rank candidates deterministically by final relevance score descending
        candidates.sort(key=lambda it: it.relevance_score, reverse=True)
        items = candidates[:top_k]

        latency = round((time.time() - start_time) * 1000, 2)
        return RetrievedEvidence.from_items(conversation_id, items, latency_ms=latency)

