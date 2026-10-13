# Leakage and reproducibility audit — 13 October 2026

> Pembaruan: baca [audit mendalam setelah skor 0,60621](deep_audit/ANALYSIS_060621.md). Notebook asli memakai konteks transduktif dan ID numerik lengkap; keduanya perlu dibedakan dari pipeline modular baru.

## Dataset and direct leakage checks

- Source: user-uploaded original five-class ZIP. Train: **533,465 × 17**. Test: **228,629 × 16**.
- Unique `building_id` in each split, with no overlap. Missing values: **0**.
- No `_post_eq` columns are present in the uploaded data; the pre-earthquake variables do not explicitly disclose observed damage.
- `building_id` is a unique identifier, not a target. Its prefixes appear geographically meaningful but are proxies, not true coordinates. They can cause local memorization and should undergo no-geography ablation and group holdout stress testing.
- `superstructure` is **one categorical field** containing 11 material names, not eleven binary flags. Converting it into a sum of independent boolean indicators would be incorrect.

## Hidden-target/placeholder risk

The official `sample_submission.csv` grade distribution is almost uniform: Grade 1: 45,579; Grade 2: 45,795; Grade 3: 45,520; Grade 4: 45,959; Grade 5: 45,776. Training grade counts are 55,170 / 61,080 / 95,488 / 128,691 / 193,036. The sample therefore contains placeholder labels, **not hidden test targets**. Never use it as y_test, pseudo-labels, target priors, or evaluation truth.

## Label disagreement among identical observables

Of 533,465 training rows, **29,104 rows (5.46%)** belong to **10,695 groups** of identical recorded features (excluding building_id and grade) that contain more than one true grade. This does not necessarily imply data corruption: exposure, detailed construction, or damage variability not captured by these features may differ. Do not drop conflicting rows without evidence.

## Validation leakage to avoid

1. **Full-dataset supervised target encoding**: for each prefix or district, computing class proportions before train/cal/audit split contaminates validation with its own label. Train rows must use out-of-fold statistics; held-out rows must use *only* fit-partition aggregation.
2. **Adaptive audit reuse**: older notebook metadata includes `selection_is_adaptive=True` and `pristine_independent_test=False`; prior `0.601198` validation is not a pristine independent estimate. Do not claim it represents a new Kaggle score.
3. **Threshold/ensemble selection on audit**: tune on CAL and evaluate AUDIT once after freezing the recipe; repeated score peeking recreates validation overfitting.
4. **Geographical data shift**: random split shares wards between train/test and represents the competition's likely random split; a group-ward split tests transfer to unseen wards but measures a different and harder deployment scenario.
5. **CSV schema mismatch**: although problem text states integers 1–5, the official sample_submission in this upload uses strings `Grade 1` through `Grade 5`. The default writer follows the explicit assignment with integer grades 1–5, while preserving sample ID order. String labels remain an explicit option if required by the competition validator.

## Current conclusions

The historically reported public score around 0.605–0.606 is not exceeded simply by making new predictions. The newer CPU validation experiments are recorded separately; a verified public/private score >0.65 is **not established**. Preserve the old submission until reproducible independent validation and an actual Kaggle submission prove improvement.
