"""Structural checks for the archived confusion matrix metadata."""
import json
from pathlib import Path
import numpy as np

def test_historical_v7_counts_and_score():
    """Match 60,000 row labels and recalculated per-class Macro F1."""
    path=Path(__file__).resolve().parents[1]/"reports"/"historical_counts_20261014.json"
    meta=json.loads(path.read_text())["historical_v7"]
    cm=np.asarray(meta["matrix"])
    assert cm.shape==(5,5) and cm.sum()==60000
    precision=np.diag(cm)/cm.sum(axis=0)
    recall=np.diag(cm)/cm.sum(axis=1)
    score=np.mean(2*precision*recall/(precision+recall))
    assert abs(score-meta["macro_f1"])<1e-12
