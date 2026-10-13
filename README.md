# Nepal Earthquake Structural Damage Classification (2015)

**ITS — Supervised Machine Learning 2026 | Kaggle multiclass classification | No neural networks**

[Competition invitation](https://www.kaggle.com/t/a9196b92680b4492aeb06aeaee94f654) · [Assignment instructions (Google Docs)](https://docs.google.com/document/d/1e8SbPqX3BX6_hsmawrtFu9kn80WotSfbORxYNg6cNBE/edit)

## Problem and scientific context

The 25 April 2015 Gorkha earthquake (Mw 7.8) caused widespread structural damage across Nepal. I predict **damage_grade** (Grade 1–5) from pre-event building characteristics and survey fields. The metric is **Macro F1**, i.e., the arithmetic mean of five per-class F1 scores; it does not collapse to accuracy.

This is **not** the frequently shared 3-grade DrivenData Richter's Predictor task. Do not mix those labels, features, or leaderboard scores with this five-grade competition.

## Assessment rubric (as specified in the assignment)

| Component | Weight |
|:--|--:|
| Kaggle public leaderboard | 15% |
| Kaggle private leaderboard | 35% |
| Repository tree, workflow, Pipeline + ColumnTransformer, docstrings, usage instructions, interpretation | **50%** |

## Repository layout

```text
Structural-Damage-Classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                      # train.csv, test.csv, sample_submission.csv (not committed)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_submission.ipynb
├── scripts/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_preprocess.py
├── models/                    # reserved for saved model checkpoints
├── submissions/               # generated CSV + audit JSON + joblib (ignored by Git)
└── reports/
    └── figures/              # generated analysis figures
```

## Installation and reproducible usage

Python 3.10+ is recommended. From a terminal in the repository root:

```bash
git clone https://github.com/Constantine1507/Structural-Damage-Classification.git
cd Structural-Damage-Classification
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

1. Accept the [competition rules on Kaggle](https://www.kaggle.com/t/a9196b92680b4492aeb06aeaee94f654), then obtain **its exact** train.csv, test.csv, and sample_submission.csv. Place them in `data/` or provide the original competition ZIP via `--data path/to/file.zip`.
2. Open notebooks in order for exploratory analysis, fitting/evaluation, and validation of the output. Or run the same modular code:

```bash
python -m pytest -q
python -m scripts.train --data data --quick      # functionality smoke test; NOT final model
python -m scripts.train --data data              # full non-neural ensemble experiment
python -m scripts.predict --data data --model submissions/model_bundle.joblib --output submissions/submission.csv
```

3. Submit `submissions/submission.csv` via the competition's **Submit Predictions** tab. The CSV columns are exactly `building_id,damage_grade`, in the order of `sample_submission.csv`, without an index. **Important source discrepancy:** the assignment describes integer 1–5 predictions, but the actual uploaded `sample_submission.csv` stores labels as `Grade 1`–`Grade 5`. The repository now defaults to matching the exact official sample style (`--label-style auto`). Verify the submission validator if Kaggle rejects either format. Never learn from sample submission's placeholder grade values.

## Methodology / analytical workflow

1. **Integrity checks:** retain the 12-digit building_id as text, reject duplicate/overlapping IDs and mismatched test or sample fields; normalize target text/integer to Grades 1–5.
2. **EDA:** target frequencies, missingness, categorical levels, structural-material indicators, and pre-earthquake numeric distributions.
3. **Feature engineering:** log building age and area, age × floor count, area/floor and height/floor, superstructure material counts. Optional coarse building ID prefixes are treated as possible location proxies, **not** independent geological facts. We NEVER include the complete building_id as an ML feature.
4. **Leakage control:** features derived using training-only reference statistics. Post-earthquake floor/height measurements are dropped when present because they can encode actual damage. Avoid target encoding performed before splitting.
5. **Preprocessing:** an sklearn **Pipeline** encloses a fitted `SeismicFeatures` transformer followed by **ColumnTransformer** (median imputation for numerical columns; imputation + ordinal encoding for LightGBM; native string categories for CatBoost). Unknown categories at inference are supported.
6. **Candidate models:** LightGBM, class-weighted LightGBM, no-geography ablation and CatBoost with native categorical input. The assignment's no-neural-network rule is respected.
7. **Data splits:** stratified FIT 70% / CAL 15% / untouched AUDIT 15%. Model comparison and per-grade log-probability offsets use CAL **only**. AUDIT is evaluated once after selection. This is an estimate of generalization, not Kaggle public/private truth.
8. **Decision optimization:** choose between single classifiers and 2-model probability blends; tune a small set of offsets for **Macro F1** on CAL only, retaining a no-offset control. Refit the selected models on all labeled training data after reporting the audit.
9. **Deliverables:** `submissions/submission.csv`, `model_bundle.joblib`, `selection.json` and `audit_metrics.json`.

## Baseline and reporting integrity

Previous user-reported leaderboard Macro F1 was approximately **0.605–0.60616**. An archived notebook reported **0.601198 local Macro F1**, versus a previously reported public score of **0.60606**. These scores refer to older experiments, **not to this repository's newly authored code**.

**Target 0.65 is an aspiration, NOT a verified score.** No updated validation, public score or private score is asserted until train/test files are supplied and experiments actually run. Keep the former submission unless a replacement shows consistent, reproducible gains; test one Kaggle candidate at a time and record its exact version in the submission description.

| Experiment | CAL Macro F1 | Untouched AUDIT Macro F1 | Kaggle public | Status |
|:--|:--|:--|:--|:--|
| Historical solution | unknown | archived local 0.601198 | user reported ~0.605–0.60616 | historical |
| New LightGBM / CatBoost ensemble | pending | pending | pending | run `python -m scripts.train --data data` |

## Interpretation and improvement hypotheses

- **Geographical heterogeneity:** likely affects local shaking intensity and construction standards. Compare the location-prefix model to the `lgb_no_geo` ablation. Strong random-split performance coupled with weak geographic transfer signals memorization risk.
- **Structural features:** foundation type, materials, roof and floor types, stories, age and plinth area can predict vulnerability. Importance should be measured with permutation or SHAP only *after* evaluation, and associations must not be presented as causal effects.
- **Minority-grade Macro F1:** inspect per-class recall/precision and the 5×5 confusion matrix in `audit_metrics.json`. A class weighting improvement can help minority grades but hurt others; aggregate Macro F1 must decide.
- **Ordinal severity:** Grades 1–5 are ordered, but plain multiclass losses treat them as nominal. A possible subsequent experiment is cumulative threshold models `P(y>k)` with monotonic correction and classwise OOF blending. Add it **only** when real calibration results justify the complexity.
- **Overfitting risks:** repeated tweaks against AUDIT or Kaggle public scores amount to validation leakage. Fix random seeds, save experiments, and reserve the audit for final assessment.
- **Out-of-sample uncertainty:** a 0.005 score change may be within sample variability. Prefer repeated stratified folds and group-based robustness checks before asserting superiority.

## Reproducibility limitations

Original data are excluded per assignment requirements and possible Kaggle redistribution rules. Model files, leaderboard confirmation, training runtime and achieved scores are not included without running against the actual competition data. The older CSV is retained as a **historical baseline only**, not silently replaced.

## Attribution and ethical use

Data source: Central Bureau of Statistics Nepal, post-earthquake survey as distributed by the competition organizer. Predictions aid retrospective data-science evaluation; they are not an engineering safety inspection.

## Leakage audit update (13 October 2026)

From the original 533,465-row train and 228,629-row test ZIP: zero missing values; no explicit post-earthquake floor or height attributes; train/test building IDs are disjoint. The sample submission has almost uniform randomly-looking placeholder labels across the five grades, so its damage_grade **must never be used for training**. Among exact-feature duplicates, 10,695 groups (29,104 training rows, 5.46%) have conflicting grade labels, indicating non-identifiability from recorded predictors. Archived prior work explicitly recorded `selection_is_adaptive=True` and `pristine_independent_test=False`: repeated tuning on historical validation can bias estimates. Follow the fit/cal/audit protocol; OOF-encode target-based location statistics and do not reuse audit labels for model selection. See `reports/LEAKAGE_AUDIT.md`.
