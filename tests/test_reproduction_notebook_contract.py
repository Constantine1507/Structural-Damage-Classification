"""Regression checks for the reproducibility notebook and final submission."""
import json
from pathlib import Path

from scripts.audit_submission10 import FINAL_SHA, TEST_SHA, TRAIN_SHA


ROOT = Path(__file__).resolve().parents[1]


def test_sha256_reference_values():
    """Keep the full, exact SHA-256 digests in the verifier."""
    for value in (FINAL_SHA, TEST_SHA, TRAIN_SHA):
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")


def test_notebook_integrity_hash_is_current():
    """Prevent stale source cells with shortened dataset fingerprints."""
    nb = json.loads((ROOT / "notebooks/09_end_to_end_verified.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert TEST_SHA in code
    assert FINAL_SHA in code
    assert TRAIN_SHA in code
    assert "d732c3070b878ebc1346907d50e6a6a9db0eadf0790631b37c03d3ba6c5c" not in code
    assert "verified_evidence.json" in code


def test_notebook_cells_have_stable_ids():
    """Keep notebooks compatible with current Jupyter validation."""
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if path.name not in {"09_end_to_end_verified.ipynb", "01_eda.ipynb", "02_modeling.ipynb", "03_submission.ipynb"}:
            continue
        nb = json.loads(path.read_text())
        ids = [c.get("id") for c in nb["cells"]]
        assert all(ids), f"{path.name}: missing cell id"
        assert len(ids) == len(set(ids)), f"{path.name}: duplicated cell id"


def test_notebook_sources_use_zip_input():
    """Check required data inputs without depending on Path quote/spacing style."""
    for filename in ("01_eda.ipynb", "02_modeling.ipynb", "03_submission.ipynb"):
        nb = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
        )
        # The EDA and model-fitting notebooks read the original competition ZIP.
        # Notebook 03 now audits archived final predictions; it does not retrain
        # the separate notebook02_modular model or require its private bundle.
        assert "structural-damage-classification.zip" in code, filename
        assert "read_competition_data(ROOT/'data')" not in code, filename
        compile(code, filename, "exec")
        if filename == "02_modeling.ipynb":
            assert "artifacts/notebook02_modular" in code
        if filename == "03_submission.ipynb":
            assert "ZipFile" in code
            assert "sample_submission.csv" in code
            assert "010_Constantine(10).csv" in code
            assert "010_Constantine(12).csv" in code
            assert "artifacts/notebook02_modular" not in code
