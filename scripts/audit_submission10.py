"""Audit submission 010 and the historical v6/v7 validation checkpoint.

This script preserves the canonical CSV. Historical confirmation is adaptive and
must not be mistaken for independent Kaggle test-label evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import platform
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

TRAIN_SHA = "231c9bf0430faa9c1d2281b4edbcd9ce0d7a863c6ff67042a8fbc992469d37bb"
TEST_SHA = "d732c3070b878ebc1346907d50e6a6c1a6a9db0eadf0790631b37c03d3ba6c5c"
FINAL_SHA = "2764f5f59a3924b982a72584edb4facdd8aeabde16519a8ce0e04e45975d0767"
CLASSES = [1, 2, 3, 4, 5]


def checksum(data):
    """Return SHA256 of original file bytes."""
    return hashlib.sha256(data).hexdigest()


def read_inputs(source):
    """Read required competition CSV bytes from ZIP or folder."""
    path = Path(source)
    names = ("train.csv", "test.csv", "sample_submission.csv")
    if path.is_dir():
        return {n: (path / n).read_bytes() for n in names}
    with ZipFile(path) as archive:
        entries = {Path(n).name: n for n in archive.namelist() if Path(n).name in names}
        if set(entries) != set(names):
            raise ValueError("Input archive missing train, test, or sample_submission")
        return {n: archive.read(entries[n]) for n in names}


def labels(values):
    """Read Grade 1..5 or numeric 1..5 with strict validation."""
    values = values.astype("string").str.strip().str.extract(r"^(?:Grade\s*)?([1-5])$", expand=False)
    if values.isna().any():
        raise ValueError("Unrecognized damage_grade values")
    return values.astype(int).to_numpy()


def validate_submission(source, submission, output):
    """Check hash, schema, order, classes, and exact replay of submission 010."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = read_inputs(source)
    if checksum(data["train.csv"]) != TRAIN_SHA or checksum(data["test.csv"]) != TEST_SHA:
        raise ValueError("Original training/test dataset SHA mismatch")
    raw = Path(submission).read_bytes()
    if checksum(raw) != FINAL_SHA:
        raise ValueError("Submission file differs from frozen tenth submission")
    frame = pd.read_csv(io.BytesIO(raw), dtype={"building_id": str})
    test = pd.read_csv(io.BytesIO(data["test.csv"]), dtype={"building_id": str})
    sample = pd.read_csv(io.BytesIO(data["sample_submission.csv"]), dtype={"building_id": str})
    train = pd.read_csv(io.BytesIO(data["train.csv"]), dtype={"building_id": str})
    if list(frame.columns) != ["building_id", "damage_grade"]:
        raise ValueError("Unexpected submission columns")
    if len(frame) != 228629 or len(test) != len(frame):
        raise ValueError("Unexpected number of predictions")
    if not frame.building_id.is_unique or frame.building_id.isna().any():
        raise ValueError("Duplicated/missing building_id in submission")
    if not frame.building_id.equals(sample.building_id) or set(test.building_id) != set(frame.building_id):
        raise ValueError("Submission ordering or IDs differ from competition test")
    if train.building_id.isin(test.building_id).any() or "damage_grade" in test:
        raise ValueError("Train/test data separation invalid")
    y_pred = labels(frame.damage_grade)
    target = labels(train.damage_grade)
    copied = output / "010_Constantine(10)_verified.csv"
    shutil.copyfile(submission, copied)
    if copied.read_bytes() != raw:
        raise AssertionError("Exact artifact replay failed")
    report = {
        "submission": "010_Constantine(10).csv",
        "sha256": checksum(raw), "n_predictions": len(y_pred),
        "grade_counts": {str(c): int((y_pred == c).sum()) for c in CLASSES},
        "source_train_sha256": TRAIN_SHA, "source_test_sha256": TEST_SHA,
        "submission_ids_exactly_match_sample_order": True,
        "replay_byte_identical": True,
        "public_macro_f1_reported": 0.60627, "private_macro_f1": None,
        "model_weights_retrained_to_exact_010": False,
        "public_score_source": "previous Kaggle screenshot; no hidden labels available",
        "python": platform.python_version(), "pandas": pd.__version__,
    }
    (output / "submission10_integrity.json").write_text(json.dumps(report, indent=2))
    return report, target, y_pred


def verify_historical(checkpoint, train_target, final_pred, output):
    """Validate v6/v7 pooled arrays, each source fold, and the v7 native replay."""
    output = Path(output)
    summary = {}
    with ZipFile(checkpoint) as archive:
        for version in ("v6", "v7"):
            prefix = f"research_{version}/confirmation/"
            with np.load(io.BytesIO(archive.read(prefix + "pooled.npz"))) as z:
                idx, truth, prediction = z["indices"].copy(), z["truth"].copy()+1, z["prediction"].copy()+1
            if len(idx) != 60000 or len(np.unique(idx)) != 60000 or not np.array_equal(train_target[idx], truth):
                raise AssertionError(f"{version}: held-out indices/labels do not match raw train")
            pairs = {}
            for fold in range(3):
                with np.load(io.BytesIO(archive.read(prefix + f"fold_{fold}.npz"))) as z:
                    x, y, p = z["indices"].copy(), z["truth"].copy()+1, z["prediction"].copy()+1
                if len(x) != 20000 or not np.array_equal(train_target[x], y):
                    raise AssertionError(f"{version}: damaged fold {fold}")
                pairs.update({int(i): (int(yy), int(pp)) for i, yy, pp in zip(x, y, p)})
            if len(pairs) != len(idx) or any(pairs[int(i)] != (int(y), int(p)) for i, y, p in zip(idx, truth, prediction)):
                raise AssertionError(f"{version}: pooled/fold predictions disagree")
            cm = confusion_matrix(truth, prediction, labels=CLASSES)
            result = {
                "n_validation": len(idx), "macro_f1": float(f1_score(truth,prediction,labels=CLASSES,average="macro")),
                "accuracy": float(accuracy_score(truth,prediction)),
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(truth,prediction,labels=CLASSES,output_dict=True,zero_division=0),
                "checkpoint_member": prefix + "pooled.npz",
                "independent_holdout": False,
                "note": "Historical confirmation folds were seen during adaptive model development."
            }
            pd.DataFrame(cm,index=[f"Actual G{c}" for c in CLASSES],
                         columns=[f"Pred G{c}" for c in CLASSES]).to_csv(output / f"{version}_confusion_counts.csv")
            pd.DataFrame(cm/cm.sum(axis=1,keepdims=True),index=[f"Actual G{c}" for c in CLASSES],
                         columns=[f"Pred G{c}" for c in CLASSES]).to_csv(output / f"{version}_confusion_normalized.csv")
            (output / f"{version}_metrics.json").write_text(json.dumps(result,indent=2))
            summary[version] = result
        native_bytes = archive.read("research_v7/full_refit/native_replay_submission.csv")
        native = pd.read_csv(io.BytesIO(native_bytes), dtype={"building_id": str})
        reference = labels(native.damage_grade)
        if len(reference) != len(final_pred):
            raise AssertionError("Native v7 predictions have wrong row count")
        changes = int((reference != final_pred).sum())
        pd.crosstab(pd.Series(reference,name="v7 native"),pd.Series(final_pred,name="final 010")).to_csv(
            output / "v7_to_final010_transitions.csv")
        summary["v7_native_vs_final010"] = {
            "changed_predictions": changes, "total_test_rows": len(final_pred),
            "archived_native_csv_sha256": checksum(native_bytes),
            "raw_model_exact_reproduction_of_final010": False,
            "test_label_confusion_matrix_available": False,
        }
    (output / "historical_verification.json").write_text(json.dumps(summary,indent=2))
    return summary


def run(data, submission, output, checkpoint=None):
    """Execute the canonical integrity check, then optionally the checkpoint audit."""
    result, target, preds = validate_submission(data,submission,output)
    summary = {"submission10": result}
    if checkpoint:
        summary["historical"] = verify_historical(checkpoint,target,preds,output)
    return summary


def main():
    """Command-line entry point with an optional external historical archive."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default="data/structural-damage-classification.zip")
    p.add_argument("--submission",default="submissions/010_Constantine(10).csv")
    p.add_argument("--checkpoint",default=None,help="Optional 454MB historical checkpoint ZIP")
    p.add_argument("--output",default="artifacts/submission10_audit")
    args=p.parse_args()
    r=run(args.data,args.submission,args.output,args.checkpoint)
    print("010 SHA256:",r["submission10"]["sha256"])
    print("Rows:",r["submission10"]["n_predictions"])
    if "historical" in r:
        print("Historical v7 Macro F1:",r["historical"]["v7"]["macro_f1"])
        print("Archived v7 differences:",r["historical"]["v7_native_vs_final010"]["changed_predictions"])

if __name__ == "__main__":
    main()
