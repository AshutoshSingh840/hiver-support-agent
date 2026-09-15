"""Unit tests for intent classifier, taxonomy, compound intent scoring, and boundary resolution."""

import pytest
from src.intent.classifier import IntentClassifier
from src.models import IntentCode


def test_classify_battery_intent():
    clf = IntentClassifier()
    text = "My iPhone 8 battery drains completely within 2 hours of charging."
    pred = clf.predict(1, text)
    assert pred.label == IntentCode.INT_BATTERY
    assert pred.confidence >= 0.70
    assert "battery" in pred.supporting_excerpt.lower()


def test_classify_icloud_intent():
    clf = IntentClassifier()
    text = "My photos are not syncing to iCloud drive and storage is full."
    pred = clf.predict(2, text)
    assert pred.label == IntentCode.INT_ICLOUD
    assert pred.confidence >= 0.70


def test_classify_out_of_scope():
    clf = IntentClassifier()
    text = "Hey there, having a nice day outside!"
    pred = clf.predict(3, text)
    assert pred.label == IntentCode.INT_OUT_OF_SCOPE


def test_classify_empty_query():
    clf = IntentClassifier()
    pred = clf.predict(4, "")
    assert pred.label == IntentCode.INT_OUT_OF_SCOPE


def test_compound_intent_favors_battery_over_ios_update_mention():
    clf = IntentClassifier()
    text = "Updated to iOS 11.1 and now my battery drains in 3 hours. Horrific battery life!"
    pred = clf.predict(5, text)
    assert pred.label == IntentCode.INT_BATTERY
    assert pred.confidence >= 0.70
    assert "battery" in pred.supporting_excerpt.lower()


def test_compound_intent_favors_connectivity_over_ios_mention():
    clf = IntentClassifier()
    text = "Since installing iOS 11, my wifi disconnects constantly and bluetooth won't pair."
    pred = clf.predict(6, text)
    assert pred.label == IntentCode.INT_CONN
    assert pred.confidence >= 0.70


def test_compound_intent_favors_icloud_over_ios_mention():
    clf = IntentClassifier()
    text = "After iOS 11 update, my photos library stopped syncing to iCloud drive."
    pred = clf.predict(7, text)
    assert pred.label == IntentCode.INT_ICLOUD
    assert pred.confidence >= 0.70


def test_true_ios_system_update_failure_classified_as_ios():
    clf = IntentClassifier()
    text = "Software update failed and now my phone is bricked and stuck on apple logo. Restore error."
    pred = clf.predict(8, text)
    assert pred.label == IntentCode.INT_IOS
    assert pred.confidence >= 0.70


def test_version_number_does_not_dominate_problem_symptom():
    clf = IntentClassifier()
    text = "iPhone 7 on 11.1 screen cracked and touch is completely unresponsive."
    pred = clf.predict(9, text)
    assert pred.label == IntentCode.INT_HARDWARE
    assert pred.confidence >= 0.70


def test_ambiguous_compound_query_flags_confidence():
    clf = IntentClassifier()
    # Query with exactly balanced competing primary keywords from different domains
    text = "wifi battery"
    pred = clf.predict(10, text)
    assert pred.confidence <= 0.60


def test_colloquial_ios_glitch_phrases_classified_as_ios():
    clf = IntentClassifier()
    cases = [
        "Why does my phone change the letter I to an exclamation point and question mark box?!",
        "Autocorrect glitch is driving me crazy on my iPhone after updating",
        "Keyboard lag and typing glitch makes it impossible to type",
        "This new update ruined my phone completely and nothing works right",
        "The symbols are glitching on my screen when typing"
    ]
    for idx, text in enumerate(cases, 100):
        pred = clf.predict(idx, text)
        assert pred.label == IntentCode.INT_IOS, f"Failed for '{text}': got {pred.label}"
        assert pred.confidence >= 0.70


def test_apple_pay_and_itunes_purchase_classified_as_store():
    clf = IntentClassifier()
    pay_text = "Apple Pay declined at checkout but charged my credit card"
    pred_pay = clf.predict(200, pay_text)
    assert pred_pay.label == IntentCode.INT_STORE
    assert pred_pay.confidence >= 0.70

    itunes_text = "iTunes purchase disappeared from my music library and won't download"
    pred_itunes = clf.predict(201, itunes_text)
    assert pred_itunes.label == IntentCode.INT_STORE
    assert pred_itunes.confidence >= 0.70


def test_compound_update_and_hardware_symptoms_favors_hardware():
    clf = IntentClassifier()
    text = "Updated iOS and now my speaker is buzzing with distorted sound and crackling"
    pred = clf.predict(300, text)
    assert pred.label == IntentCode.INT_HARDWARE
    assert pred.confidence >= 0.70


