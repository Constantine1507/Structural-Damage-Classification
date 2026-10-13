"""Deterministic checks for integrity of the committed prediction artifacts."""
from pathlib import Path
import json
import pytest
from scripts.inspect_results import inspect, ROOT


def test_committed_results():
    """Check incumbent/candidate format and archived bytes without training models."""
    result = inspect()
    assert result["rows"] == 228629
    assert result["changed"] == 685
    assert result["best_public"] == 0.60627


def test_hash_tamper_detection(tmp_path):
    """Modification of a saved CSV must be noticed before score claims are made."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "submissions").mkdir()
    src = json.loads((ROOT / "reports" / "submission10_followup_results.json").read_text())
    (tmp_path / "reports" / "submission10_followup_results.json").write_text(json.dumps(src))
    for key in ("submission10", "candidate11"):
        (tmp_path / "submissions" / src[key]["filename"]).write_text("building_id,damage_grade\n0001,1\n")
    with pytest.raises(ValueError, match="Hash mismatch"):
        inspect(tmp_path)
