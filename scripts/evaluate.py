"""Reproducible Macro-F1 scoring and calibration using only calibration labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


CLASSES = np.array([1, 2, 3, 4, 5])


def macro_f1(y, predictions):
    """Return unweighted average F1 across exactly five classes."""
    return float(f1_score(y, predictions, labels=CLASSES, average="macro", zero_division=0))


def decide(probabilities, offsets=None):
    """Apply per-class log-probability offsets and return class labels."""
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 5:
        raise ValueError("Expected probability matrix with five columns")
    offsets = np.zeros(5) if offsets is None else np.asarray(offsets, dtype=float)
    return CLASSES[np.argmax(np.log(np.clip(p, 1e-12, 1)) + offsets[None, :], axis=1)]


def tune_offsets(probabilities, y, seed=42, passes=4):
    """Coordinate-search offsets on calibration labels only; keep zero-offset control."""
    rng = np.random.default_rng(seed)
    candidates = [np.zeros(5)]
    for _ in range(3):
        candidates.append(rng.normal(0, 0.15, size=5))
    best = np.zeros(5)
    best_score = macro_f1(y, decide(probabilities, best))
    for starting in candidates:
        trial = starting.copy()
        for width in (0.35, 0.16, 0.07, 0.03)[:passes]:
            changed = True
            steps = 0
            while changed and steps < 3:
                changed = False
                steps += 1
                for cls in rng.permutation(5):
                    for delta in (-width, width):
                        temp = trial.copy()
                        temp[cls] += delta
                        temp -= temp.mean()
                        score = macro_f1(y, decide(probabilities, temp))
                        # Require meaningful improvements to avoid noise fitting.
                        if score > best_score + 0.0001:
                            best_score, best, trial = score, temp.copy(), temp.copy()
                            changed = True
    return best, best_score


def experiment_report(y, pred, output, metadata=None):
    """Write class F1, classification report, confusion matrix and experiment context."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "macro_f1": macro_f1(y, pred),
        "classification_report": classification_report(y, pred, labels=CLASSES,
                                                         output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "metadata": metadata or {}
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
