"""Integration test verifying automated leakage checks."""

import json
from pathlib import Path
import pytest
from src.data.splitter import verify_split_leakage


def test_leakage_check_passes_on_disjoint_sets():
    retrieval = {1, 2, 3, 4, 5}
    golden = {10, 11, 12, 13}
    passed, overlap = verify_split_leakage(retrieval, golden)
    assert passed is True
    assert len(overlap) == 0


def test_leakage_check_fails_on_overlap():
    retrieval = {1, 2, 3, 4, 5}
    leaked_golden = {5, 6, 7, 8}
    passed, overlap = verify_split_leakage(retrieval, leaked_golden)
    assert passed is False
    assert overlap == {5}
