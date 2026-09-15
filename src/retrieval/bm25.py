"""In-memory Okapi BM25 indexing and ranking engine."""

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric lowercase tokenizer."""
    return re.findall(r"\b[a-z0-9_]+\b", text.lower())


class BM25Index:
    """Pure Python, zero-dependency in-memory Okapi BM25 search index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: Dict[int, int] = {}
        self.doc_ids: List[int] = []
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[Tuple[int, str]]):
        """Indexes a list of (doc_id, text) tuples."""
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        total_len = 0
        doc_freqs: Counter = Counter()

        for doc_id, text in documents:
            tokens = tokenize(text)
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            self.doc_ids.append(doc_id)
            total_len += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term][doc_id] = count
                doc_freqs[term] += 1

        self.avg_doc_len = total_len / max(1, self.corpus_size)

        # Compute IDF: log((N - n + 0.5) / (n + 0.5) + 1)
        for term, df in doc_freqs.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Searches index and returns top_k (doc_id, score) tuples."""
        query_tokens = tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf_val = self.idf[token]
            postings = self.inverted_index[token]

            for doc_id, tf in postings.items():
                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_id] += idf_val * (numerator / denominator)

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return ranked

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)
