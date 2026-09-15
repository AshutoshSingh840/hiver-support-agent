"""Unit tests for baseline models (trivial and simple ML)."""

import pytest
from src.models import IntentCode, RoutingDecision
from src.eval.baselines import (
    MajorityClassIntentBaseline,
    RandomRetrievalBaseline,
    TrivialRoutingBaseline,
    TfIdfLogisticRegressionIntentBaseline,
)


def test_majority_class_baseline():
    baseline = MajorityClassIntentBaseline()
    labels = [IntentCode.INT_IOS, IntentCode.INT_BATTERY, IntentCode.INT_IOS, IntentCode.INT_CONN]
    baseline.fit(labels)
    preds = baseline.predict(["my battery is dead", "wifi dropped"])
    assert len(preds) == 2
    assert preds[0] == IntentCode.INT_IOS
    assert preds[1] == IntentCode.INT_IOS


def test_random_retrieval_baseline():
    corpus_ids = [101, 102, 103, 104, 105, 106, 107]
    baseline = RandomRetrievalBaseline(corpus_ids, random_seed=42)
    res = baseline.retrieve(query_count=3, k=3)
    assert len(res) == 3
    for r in res:
        assert len(r) == 3
        assert all(doc_id in corpus_ids for doc_id in r)


def test_trivial_routing_baseline():
    baseline = TrivialRoutingBaseline(default_decision=RoutingDecision.AUTO_HANDLE)
    preds = baseline.predict(5)
    assert preds == ["auto-handle"] * 5


def test_tfidf_logistic_regression_baseline():
    train_texts = [
        "battery draining rapidly after update",
        "phone dying within an hour battery percentage drops",
        "wifi not connecting keeps dropping signal",
        "bluetooth pairing failed cannot find device",
        "icloud storage full cannot backup photos",
        "photos disappeared from icloud drive",
        "apple pay declined card payment store receipt",
        "subscription charge refund app store purchase",
    ]
    train_labels = [
        IntentCode.INT_BATTERY,
        IntentCode.INT_BATTERY,
        IntentCode.INT_CONN,
        IntentCode.INT_CONN,
        IntentCode.INT_ICLOUD,
        IntentCode.INT_ICLOUD,
        IntentCode.INT_STORE,
        IntentCode.INT_STORE,
    ]

    clf = TfIdfLogisticRegressionIntentBaseline(random_state=42)
    assert not clf.is_fitted
    clf.fit(train_texts, train_labels)
    assert clf.is_fitted
    assert len(clf.classes_) == 4

    test_queries = [
        "my battery is draining very fast",
        "cannot connect to wifi or bluetooth",
        "need refund for accidental subscription purchase in app store",
    ]
    preds = clf.predict(test_queries)
    assert len(preds) == 3
    assert preds[0] == IntentCode.INT_BATTERY
    assert preds[1] == IntentCode.INT_CONN
    assert preds[2] == IntentCode.INT_STORE

    # Probabilities
    probs = clf.predict_proba(test_queries)
    assert probs is not None
    assert probs.shape == (3, 4)


def test_tfidf_logistic_regression_empty_and_unfitted():
    clf = TfIdfLogisticRegressionIntentBaseline(random_state=42)
    preds = clf.predict(["some query"])
    assert preds == [IntentCode.INT_OUT_OF_SCOPE]
    assert clf.predict_proba(["query"]) is None
