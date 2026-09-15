"""Baseline benchmark models for calibration against AI agent components."""

import random
from typing import List, Dict, Set
from collections import Counter
from src.models import IntentCode, RoutingDecision


class MajorityClassIntentBaseline:
    """Predicts the most frequent intent label from training/retrieval corpus."""

    def __init__(self, majority_label: IntentCode = IntentCode.INT_IOS):
        self.majority_label = majority_label

    def fit(self, training_labels: List[IntentCode]):
        if training_labels:
            counts = Counter(training_labels)
            self.majority_label = counts.most_common(1)[0][0]

    def predict(self, texts: List[str]) -> List[IntentCode]:
        return [self.majority_label for _ in texts]


class RandomRetrievalBaseline:
    """Retrieves k random conversation IDs from the candidate index."""

    def __init__(self, corpus_ids: List[int], random_seed: int = 42):
        self.corpus_ids = corpus_ids
        self.rng = random.Random(random_seed)

    def retrieve(self, query_count: int, k: int = 5) -> List[List[int]]:
        results = []
        for _ in range(query_count):
            if len(self.corpus_ids) >= k:
                results.append(self.rng.sample(self.corpus_ids, k))
            else:
                results.append(list(self.corpus_ids))
        return results


class TrivialRoutingBaseline:
    """Trivial baseline that always predicts auto-handle or always escalate."""

    def __init__(self, default_decision: RoutingDecision = RoutingDecision.AUTO_HANDLE):
        self.default_decision = default_decision

    def predict(self, count: int) -> List[str]:
        return [self.default_decision.value for _ in range(count)]
