"""Audit stored submission CSVs and historical score records without retraining.

Usage:
    python -m scripts.inspect_results
    python -m scripts.inspect_results --data data
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "submission10_followup_results.json"


def sha256(path):
    """Return SHA-256 of the exact bytes of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(root=ROOT, data=None):
    """Validate recorded submissions; optionally check sample ID order from raw data."""
    root = Path(root)
    manifest = json.loads((root / "reports" / "submission10_followup_results.json").read_text(encoding="utf-8"))
    frames = {}
    for key in ("submission10", "candidate11"):
        info = manifest[key]
        path = root / "submissions" / info["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing submission: {path}")
        actual_hash = sha256(path)
        if actual_hash != info["sha256"]:
            raise ValueError(f"Hash mismatch for {path.name}. Expected {info['sha256']}, got {actual_hash}")
        df = pd.read_csv(path, dtype={"building_id": str})
        if df.columns.tolist() != ["building_id", "damage_grade"]:
            raise ValueError(f"Invalid submission columns: {path.name}")
        if len(df) != info["rows"] or df["building_id"].isna().any() or not df["building_id"].is_unique:
            raise ValueError(f"Invalid row count/IDs: {path.name}")
        if df.isna().any().any() or not df.damage_grade.isin(range(1, 6)).all():
            raise ValueError(f"Invalid classes/missing values: {path.name}")
        frames[key] = df
        print(f"{path.name}: {len(df):,} rows, SHA-256 OK, grade distribution = "
              f"{df.damage_grade.value_counts().sort_index().to_dict()}")
    a, b = frames["submission10"], frames["candidate11"]
    if not a["building_id"].equals(b["building_id"]):
        raise ValueError("Candidate and incumbent have different building_id order")
    changed = int((a.damage_grade != b.damage_grade).sum())
    if changed != manifest["candidate11"]["changed_vs_submission10"]:
        raise ValueError(f"Changed-prediction count mismatch: {changed}")
    print(f"Changed predictions 010 -> candidate 011: {changed}")
    if data is not None:
        from .preprocess import read_competition_data
        _, _, sample = read_competition_data(data)
        for key, df in frames.items():
            if not sample.building_id.astype(str).reset_index(drop=True).equals(df.building_id):
                raise ValueError(f"{key}: sample_submission ID order differs")
        print("All submission IDs match the official sample_submission.csv order.")
    print("Best public score (screenshot/user report):", manifest["submission10"]["public_macro_f1"])
    print("Candidate 011 Kaggle score: UNVERIFIED; private leaderboard: UNKNOWN.")
    print("No hidden labels were used to calculate a Kaggle score.")
    return {"rows": len(a), "changed": changed, "best_public": manifest["submission10"]["public_macro_f1"]}


def main():
    """Read CLI flags and run the results audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Optional Kaggle dataset directory or ZIP")
    args = parser.parse_args()
    inspect(data=args.data)


if __name__ == "__main__":
    main()
