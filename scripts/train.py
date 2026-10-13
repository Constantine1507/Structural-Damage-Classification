"""Train/tune/audit non-neural classifiers and export a competition-ready submission.

Run from repository root: python -m scripts.train --data data --quick
"""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from .preprocess import read_competition_data, parse_grades, build_feature_pipeline, build_submission
from .evaluate import CLASSES, macro_f1, decide, tune_offsets, experiment_report


def specs(quick=False, seed=42):
    """Define bounded LightGBM/CatBoost candidates; no neural networks."""
    rounds = 140 if quick else 950
    catrounds = 150 if quick else 1050
    common = dict(objective="multiclass", num_class=5, verbosity=-1, n_jobs=-1,
                  random_state=seed, max_depth=-1, subsample_freq=1)
    return [
        ("lgb_raw", "lgbm", True, LGBMClassifier(**common, n_estimators=rounds,
                learning_rate=0.045, num_leaves=47, min_child_samples=55,
                colsample_bytree=0.86, subsample=0.88, reg_lambda=3.0)),
        ("lgb_balanced", "lgbm", True, LGBMClassifier(**common, n_estimators=rounds,
                learning_rate=0.035, num_leaves=31, min_child_samples=80,
                colsample_bytree=0.9, reg_lambda=5.0, class_weight="balanced")),
        ("lgb_no_geo", "lgbm", False, LGBMClassifier(**common, n_estimators=rounds,
                learning_rate=0.04, num_leaves=47, min_child_samples=60,
                colsample_bytree=0.88, reg_lambda=3.0)),
        ("cat_native", "catboost", True, CatBoostClassifier(iterations=catrounds,
                depth=7, learning_rate=0.05, loss_function="MultiClass", l2_leaf_reg=6,
                random_seed=seed, verbose=False, thread_count=-1,
                allow_writing_files=False))
    ]


def ordered_probabilities(model, X):
    """Map estimator class order into columns for Grades 1..5."""
    predictions = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.named_steps["model"].classes_, dtype=int)
    result = np.zeros((len(X), 5), dtype=float)
    for i, grade in enumerate(classes):
        result[:, int(grade) - 1] = predictions[:, i]
    return result


def candidates(probs, y_cal):
    """Evaluate individual models and simple blends without touching audit labels."""
    names = list(probs)
    out = []
    for name in names:
        out.append((name, {name: 1.0}, probs[name]))
    ranked = sorted(names, key=lambda name: macro_f1(y_cal, decide(probs[name])), reverse=True)
    for a in ranked[:3]:
        for b in ranked[:3]:
            if a >= b:
                continue
            for weight in (0.35, 0.5, 0.65):
                out.append((f"{a}_{b}_{weight}", {a: weight, b: 1-weight},
                            weight * probs[a] + (1-weight) * probs[b]))
    return out


def train(args):
    """Keep calibration/model selection separate from a single untouched audit holdout."""
    train_df, test_df, sample = read_competition_data(args.data)
    y = parse_grades(train_df["damage_grade"])
    X = train_df.drop(columns=["damage_grade"])
    if not np.array_equal(np.unique(y), CLASSES):
        raise ValueError("Expected all five training labels 1..5")
    idx = np.arange(len(y))
    dev, audit = train_test_split(idx, test_size=0.15, stratify=y, random_state=args.seed)
    fit_idx, cal = train_test_split(dev, test_size=0.15 / 0.85, stratify=y[dev],
                                    random_state=args.seed + 1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Train={len(fit_idx)} calibration={len(cal)} untouched_audit={len(audit)}")
    predictions_cal, predictions_audit = {}, {}
    definitions = {name:(family, geo, estimator)
                   for name, family, geo, estimator in specs(args.quick, args.seed)}
    t0 = time.monotonic()
    for name, (family, geo, estimator) in definitions.items():
        pipe = build_feature_pipeline(X.iloc[fit_idx], estimator, family, geo)
        pipe.fit(X.iloc[fit_idx], y[fit_idx])
        predictions_cal[name] = ordered_probabilities(pipe, X.iloc[cal])
        predictions_audit[name] = ordered_probabilities(pipe, X.iloc[audit])
        print(f"{name}: CAL Macro-F1={macro_f1(y[cal], decide(predictions_cal[name])):.6f}")
    leaderboard = []
    for name, weights, p in candidates(predictions_cal, y[cal]):
        raw = macro_f1(y[cal], decide(p))
        offsets, adjusted = tune_offsets(p, y[cal], seed=args.seed, passes=2 if args.quick else 4)
        # Prefer uncalibrated decision if offset gain is too small.
        if adjusted < raw + 0.0015:
            offsets, adjusted = np.zeros(5), raw
        leaderboard.append((adjusted, name, weights, offsets, raw))
    leaderboard.sort(key=lambda item: item[0], reverse=True)
    score_cal, winner, weights, offsets, raw = leaderboard[0]
    # Audit labels are referenced ONLY AFTER the choice is frozen.
    p_audit = sum(weight * predictions_audit[name] for name, weight in weights.items())
    p_baseline = predictions_audit["lgb_raw"]
    audit_baseline = macro_f1(y[audit], decide(p_baseline))
    audit_candidate = macro_f1(y[audit], decide(p_audit, offsets))
    audit_raw = macro_f1(y[audit], decide(p_audit))
    print(f"Selected={winner} CAL={score_cal:.6f} AUDIT={audit_candidate:.6f}"
          f" (raw_audit={audit_raw:.6f}, lgb_baseline_audit={audit_baseline:.6f})")
    experiment_report(y[audit], decide(p_audit, offsets), out/"audit_metrics.json", {
        "winner": winner, "calibration_f1":score_cal,
        "audit_lgb_raw_f1":audit_baseline, "audit_uncalibrated_f1":audit_raw,
        "selection_used_audit_labels":False, "quick":args.quick,
        "rows_fit":len(fit_idx), "rows_cal":len(cal), "rows_audit":len(audit),
        "train_seconds":round(time.monotonic()-t0, 1)
    })
    selection = {"name":winner, "weights":weights, "offsets":offsets.tolist(),
                 "cal_f1":score_cal, "audit_f1":audit_candidate,
                 "baseline_audit_f1":audit_baseline,
                 "all_cal_scores":[{"name":n, "cal_score":s, "raw_score":r}
                    for s,n,w,o,r in leaderboard]}
    (out/"selection.json").write_text(json.dumps(selection,indent=2), encoding="utf-8")
    # Refit using all available labeled rows, never using hidden Kaggle targets.
    models, p_test = {}, np.zeros((len(test_df), 5), dtype=float)
    for name, weight in weights.items():
        family, geo, estimator = definitions[name]
        pipe = build_feature_pipeline(X, estimator, family, geo)
        pipe.fit(X, y)
        p_test += weight * ordered_probabilities(pipe, test_df)
        models[name] = pipe
    predictions = decide(p_test, offsets)
    build_submission(sample, test_df, predictions, out/"submission.csv", "auto")
    joblib.dump({"models":models, "weights":weights, "offsets":offsets.tolist(),
                 "classes":CLASSES.tolist()}, out/"model_bundle.joblib")
    print(f"Saved: {out/'submission.csv'} | Models: {out/'model_bundle.joblib'}")
    return selection


def main():
    """Parse CLI arguments and execute the reproducible experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="Kaggle CSV folder or competition ZIP")
    parser.add_argument("--output", default="submissions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Short smoke run, not final scoring")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
