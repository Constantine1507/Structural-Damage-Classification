"""Regression tests for the canonical tenth submission audit."""
from scripts.audit_submission10 import FINAL_SHA, TRAIN_SHA, TEST_SHA, labels
import pandas as pd
import pytest

def test_checksums_are_complete():
    """Each SHA-256 digest has 64 lowercase hex characters."""
    for h in (FINAL_SHA,TRAIN_SHA,TEST_SHA):
        assert len(h)==64 and all(c in '0123456789abcdef' for c in h)

def test_grade_parser():
    """Both competition grade formats map exactly to 1–5."""
    assert labels(pd.Series(['Grade 1','Grade 2','3',4,'5'])).tolist()==[1,2,3,4,5]

def test_grade_parser_rejects_invalid():
    """Unexpected classes fail closed rather than being silently corrected."""
    with pytest.raises(ValueError):
        labels(pd.Series(['Grade 9']))
