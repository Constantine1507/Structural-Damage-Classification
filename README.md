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
│   ├── 03_submission.ipynb
│   ├── 04_experiment_review.ipynb
│   └── 05_submission10_and_refinement.ipynb
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
├── submissions/               # curated final CSVs; other run outputs ignored
└── reports/
    ├── EXPERIMENT_RESULTS.md
    ├── experiment_history.json
    ├── LEAKAGE_AUDIT.md
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

3. Treat the generated CSV as a **candidate**. The default output follows the explicit assignment: integer `damage_grade` 1–5, columns `building_id,damage_grade`, sample ID order, and no index. The uploaded sample uses `Grade 1` strings; `--label-style grade` remains an explicit option if the competition validator requires it. Never train on the sample placeholder labels. Use distinct `--output submissions/run_name` directories to preserve prior runs. Do not overwrite the incumbent CSV.

## Methodology / analytical workflow

1. **Integrity checks:** retain the 12-digit building_id as text, reject duplicate/overlapping IDs and mismatched test or sample fields; normalize target text/integer to Grades 1–5.
2. **EDA:** target frequencies, missingness, categorical levels, structural-material indicators, and pre-earthquake numeric distributions.
3. **Feature engineering:** log building age and area, age × floor count, area/floor and height/floor, superstructure material counts. Optional coarse building ID prefixes are treated as possible location proxies, **not** independent geological facts. We NEVER include the complete building_id as an ML feature.
4. **Leakage control:** features derived using training-only reference statistics. Post-earthquake floor/height measurements are dropped when present because they can encode actual damage. Avoid target encoding performed before splitting.
5. **Preprocessing:** an sklearn **Pipeline** encloses a fitted `SeismicFeatures` transformer followed by **ColumnTransformer** (median imputation for numerical columns; imputation + ordinal encoding for LightGBM; native string categories for CatBoost). Unknown categories at inference are supported.
6. **Candidate models:** LightGBM, class-weighted LightGBM, no-geography ablation and CatBoost with native categorical input. The assignment's no-neural-network rule is respected.
7. **Data splits:** stratified FIT 70% / CAL 15% / AUDIT 15% reserved within this run. Model comparison and per-grade log-probability offsets use CAL **only**. AUDIT is evaluated once after selection. Historical use of the same dataset means this split cannot be called globally pristine. This is not Kaggle public/private truth.
8. **Decision optimization:** choose between single classifiers and 2-model probability blends; tune a small set of offsets for **Macro F1** on CAL only, retaining a no-offset control. Refit the selected models on all labeled training data after reporting the audit.
9. **Deliverables:** `submissions/submission.csv`, `model_bundle.joblib`, `selection.json` and `audit_metrics.json`.

## Submission aktif: ke-10, public 0,60627

**`010_Constantine(10).csv` menjadi acuan terbaru**, dengan public Macro F1 **0,60627 berdasarkan laporan pengguna** pada 13 Oktober 2026. Private score belum diketahui. Target 0,65 belum tercapai.

- [Submission ke-10](submissions/010_Constantine(10).csv)
- [Kandidat berikutnya: 011_Constantine_candidate.csv](submissions/011_Constantine_candidate.csv)
- [Notebook pemeriksaan dan hasil](notebooks/05_submission10_and_refinement.ipynb)
- [Laporan lengkap](reports/SUBMISSION_10_AND_NEXT.md)

Dua putaran lanjutan selesai: **45 konfigurasi koreksi** dengan tiga finalis diperiksa pada tiga fold, serta **empat model tambahan / 12 blend**. Kandidat berikutnya mengubah **685 prediksi**, dengan local historical confirmation **0,601289** versus **0,601188** untuk resep submission 10. Interval delta mencakup nol; belum ada skor Kaggle kandidat baru.

Skor 0,601380 pada riwayat terdahulu berasal dari varian yang berbeda dan **bukan skor submission 10**. `scripts/train.py` adalah workflow model modular terpisah; public 0,60627 tidak berasal dari pipeline itu. File CSV final dan JSON hasil dipertahankan, tetapi checkpoint lengkap putaran baru tidak tersedia setelah pergantian lingkungan sementara. Notebook 05 memeriksa hasil yang tersimpan, bukan melatih ulang model.

## Interpretation and improvement hypotheses

- **Geographical heterogeneity:** likely affects local shaking intensity and construction standards. Compare the location-prefix model to the `lgb_no_geo` ablation. Strong random-split performance coupled with weak geographic transfer signals memorization risk.
- **Structural features:** foundation type, materials, roof and floor types, stories, age and plinth area can predict vulnerability. Importance should be measured with permutation or SHAP only *after* evaluation, and associations must not be presented as causal effects.
- **Minority-grade Macro F1:** inspect per-class recall/precision and the 5×5 confusion matrix in `audit_metrics.json`. A class weighting improvement can help minority grades but hurt others; aggregate Macro F1 must decide.
- **Ordinal severity:** Grades 1–5 are ordered, but plain multiclass losses treat them as nominal. A possible subsequent experiment is cumulative threshold models `P(y>k)` with monotonic correction and classwise OOF blending. Add it **only** when real calibration results justify the complexity.
- **Overfitting risks:** repeated tweaks against AUDIT or Kaggle public scores amount to validation leakage. Fix random seeds, save experiments, and reserve the audit for final assessment.
- **Out-of-sample uncertainty:** a 0.005 score change may be within sample variability. Prefer repeated stratified folds and group-based robustness checks before asserting superiority.

## Reproducibility limitations

Original data are excluded per assignment requirements and possible Kaggle redistribution rules. Model checkpoints and original train/test CSVs are not committed. The measured historical experiment results are included with explicit provenance; the updated main training pipeline has not been run on the full dataset. Submission ke-10 dan kandidat berikutnya kini tersimpan sebagai CSV prediksi final; data mentah tetap dikecualikan.

## Attribution and ethical use

Data source: Central Bureau of Statistics Nepal, post-earthquake survey as distributed by the competition organizer. Predictions aid retrospective data-science evaluation; they are not an engineering safety inspection.

## Leakage audit update (13 October 2026)

From the original 533,465-row train and 228,629-row test ZIP: zero missing values; no explicit post-earthquake floor or height attributes; train/test building IDs are disjoint. The sample submission has almost uniform randomly-looking placeholder labels across the five grades, so its damage_grade **must never be used for training**. Among exact-feature duplicates, 10,695 groups (29,104 training rows, 5.46%) have conflicting grade labels, indicating non-identifiability from recorded predictors. Archived prior work explicitly recorded `selection_is_adaptive=True` and `pristine_independent_test=False`: repeated tuning on historical validation can bias estimates. Follow the fit/cal/audit protocol; OOF-encode target-based location statistics and do not reuse audit labels for model selection. See `reports/LEAKAGE_AUDIT.md`.

## Pemetaan rubrik repository (50%)

| Persyaratan | Bukti di repository |
|:--|:--|
| Tree structure | Struktur folder di atas; data, notebook, scripts, tests, reports terpisah |
| Workflow analisis | Notebook 01 EDA → 02 pemodelan → 03 submission → 04 ulasan eksperimen |
| Pipeline + ColumnTransformer | `build_feature_pipeline` di `scripts/preprocess.py`, digunakan untuk fit dan refit |
| Docstring function/class | Seluruh fungsi dan class Python pada `scripts/` |
| Cara penggunaan | Instalasi, perintah train/predict, dan format input/output pada README |
| Laporan interpretasi | Bagian hasil, hipotesis, laporan eksperimen, confusion matrix dan F1 per kelas |

Pemenuhan struktur tidak menjamin nilai penuh; penilaian akhir tetap oleh pengajar. Public dan private score menyumbang 15% dan 35% secara terpisah.
