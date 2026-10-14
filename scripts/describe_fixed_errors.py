"""Describe frozen audit errors without using them to select another model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def main():
    """Export class deltas and supported subgroup diagnostics for interpretation only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='artifacts/fixed_run')
    args = parser.parse_args()
    out = Path(args.output)
    cached = np.load(out / 'audit_predictions.npz')
    actual, prediction, baseline = (cached[k] for k in ['y', 'predicted', 'baseline'])
    train = pd.read_csv(Path(args.data) / 'train.csv', dtype={'building_id': str})
    frame = train.iloc[cached['indices']].reset_index(drop=True)
    frame['actual'] = actual
    frame['prediction'] = prediction
    frame['correct'] = actual == prediction
    result = []
    for key in ['superstructure', 'foundation_type', 'district_id']:
        for value, group in frame.groupby(key):
            if len(group) < 200:
                continue
            row = {'feature': key, 'value': str(value), 'rows': len(group),
                   'accuracy': float(group.correct.mean())}
            for label in [2, 3]:
                subset = group[group.actual == label]
                row[f'grade{label}_support'] = len(subset)
                row[f'grade{label}_recall'] = float(subset.correct.mean()) if len(subset) >= 50 else None
            result.append(row)
    pd.DataFrame(result).to_csv(out / 'subgroup_diagnostics.csv', index=False)
    class_delta = []
    current = precision_recall_fscore_support(actual, prediction, labels=range(1, 6), zero_division=0)
    prior = precision_recall_fscore_support(actual, baseline, labels=range(1, 6), zero_division=0)
    for i in range(5):
        class_delta.append({'grade': i + 1, 'support': int(current[3][i]),
                            'baseline_f1': float(prior[2][i]), 'candidate_f1': float(current[2][i]),
                            'delta_f1': float(current[2][i] - prior[2][i])})
    error = actual != prediction
    stats = {
        'purpose': 'Post-selection descriptive analysis only; not used for tuning',
        'class_comparison': class_delta,
        'error_rows': int(error.sum()),
        'adjacent_grade_errors': int((np.abs(actual - prediction) == 1).sum()),
        'adjacent_share_of_errors': float((np.abs(actual - prediction) == 1).sum() / error.sum()),
        'baseline_wrong_candidate_correct': int(((actual != baseline) & (actual == prediction)).sum()),
        'baseline_correct_candidate_wrong': int(((actual == baseline) & (actual != prediction)).sum()),
        'subgroup_warning': 'Descriptive, varying class mix; no causal claim or geography extrapolation guarantee',
    }
    (out / 'error_diagnostics.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
