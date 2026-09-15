"""Baseline benchmark models for calibration against AI agent components."""

import random
from typing import List, Dict, Set, Any, Optional
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


class TfIdfLogisticRegressionIntentBaseline:
    """
    Standard simple ML baseline using TF-IDF word/sub-word n-grams and Logistic Regression.
    Trained strictly on pre-split historical partition to prevent evaluation leakage.
    """

    def __init__(
        self,
        ngram_range=(1, 2),
        max_features=5000,
        random_state: int = 42,
        c_param: float = 1.0,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.random_state = random_state
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b"
            )),
            ("clf", LogisticRegression(
                C=c_param,
                max_iter=500,
                random_state=random_state
            ))
        ])
        self.is_fitted = False
        self.classes_ = []

    def fit(self, texts: List[str], labels: List[Any]):
        """Fits the TF-IDF vectorizer and logistic regression classifier."""
        if not texts or not labels:
            return self
        str_labels = [l.value if isinstance(l, IntentCode) else str(l) for l in labels]
        self.pipeline.fit(texts, str_labels)
        self.classes_ = list(self.pipeline.classes_)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[IntentCode]:
        """Predicts IntentCode labels for input texts."""
        if not self.is_fitted or not texts:
            return [IntentCode.INT_OUT_OF_SCOPE for _ in texts]
        raw_preds = self.pipeline.predict(texts)
        results = []
        for p in raw_preds:
            try:
                results.append(IntentCode(p))
            except Exception:
                results.append(IntentCode.INT_OUT_OF_SCOPE)
        return results

    def predict_proba(self, texts: List[str]):
        """Predicts intent class probability distribution."""
        if not self.is_fitted or not texts:
            return None
        return self.pipeline.predict_proba(texts)
