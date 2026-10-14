"""Resume the frozen winning recipe without repeating selection or looking at new scores."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from .leakage_fixed import SafeFeatures, make_columns
from .preprocess import read_competition_data, parse_grades, build_submission
from .run_fixed_experiment import decide


def main():
    """Cache feature matrices, save each model, and checkpoint the long CatBoost fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='artifacts/fixed_run')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output)
    protocol = json.loads((out / 'protocol.json').read_text())
    frozen = json.loads((out / 'frozen_selection.json').read_text())
    winner = frozen['winner']
    train, test, sample = read_competition_data(args.data)
    for name, expected in protocol['data_sha256'].items():
        actual = hashlib.sha256((Path(args.data) / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Input checksum mismatch: {name}')
    X = train.drop(columns='damage_grade')
    y = parse_grades(train.damage_grade)
    signature = hashlib.sha256(Path('scripts/leakage_fixed.py').read_bytes()).hexdigest()
    meta_path = out / 'refit_cache_metadata.json'
    cache_ok = meta_path.exists() and all((out / name).exists() for name in ['full_preprocessor.joblib', 'full_train_matrix.npy', 'full_test_matrix.npy'])
    if cache_ok:
        meta = json.loads(meta_path.read_text())
        if meta['feature_source_sha256'] != signature or meta['data_sha256'] != protocol['data_sha256']:
            raise ValueError('Feature cache does not match current code and source data')
        features, columns, categorical = joblib.load(out / 'full_preprocessor.joblib')
        print('LOADED_FEATURE_CACHE', flush=True)
    else:
        print('FULL_CROSSFIT_FEATURES_START', flush=True)
        features = SafeFeatures(seed=protocol['seed'])
        frame = features.fit_transform(X, y)
        columns, categorical = make_columns(frame)
        matrix = columns.fit_transform(frame, y).astype('float32')
        np.save(out / 'full_train_matrix.npy', matrix)
        del frame, matrix
        matrix = columns.transform(features.transform(test)).astype('float32')
        np.save(out / 'full_test_matrix.npy', matrix)
        del matrix
        joblib.dump((features, columns, categorical), out / 'full_preprocessor.joblib', compress=3)
        meta = {'feature_source_sha256': signature, 'data_sha256': protocol['data_sha256']}
        meta_path.write_text(json.dumps(meta, indent=2))
        print('FEATURE_CACHE_SAVED', flush=True)
    A = np.load(out / 'full_train_matrix.npy', mmap_mode='r')
    T = np.load(out / 'full_test_matrix.npy', mmap_mode='r')
    models = {}
    p = np.zeros((len(test), 5))
    for name, weight in winner['weights'].items():
        path = out / f'full_model_{name}.joblib'
        if path.exists():
            model = joblib.load(path)
            print('LOADED_MODEL', name, flush=True)
        else:
            print('FULL_FIT_START', name, frozen['rounds'][name], flush=True)
            start = time.monotonic()
            if name == 'cat_numeric':
                model = CatBoostClassifier(
                    iterations=frozen['rounds'][name], depth=8, learning_rate=.05,
                    loss_function='MultiClass', l2_leaf_reg=8, thread_count=args.threads,
                    random_seed=protocol['seed'], verbose=100,
                    allow_writing_files=True, train_dir=str(out / 'catboost_refit'),
                    save_snapshot=True, snapshot_file=str((out / 'catboost_refit.snapshot').resolve()),
                    snapshot_interval=60,
                )
                model.fit(A, y)
            else:
                params = protocol['candidates'][name].copy()
                if 'class_weight' in params:
                    params['class_weight'] = {int(k): v for k,v in params['class_weight'].items()}
                model = LGBMClassifier(
                    **params, n_estimators=frozen['rounds'][name], objective='multiclass',
                    verbosity=-1, n_jobs=args.threads, random_state=protocol['seed'],
                    colsample_bytree=.85, subsample=.9, subsample_freq=1,
                )
                model.fit(A, y, categorical_feature=categorical)
            joblib.dump(model, path, compress=3)
            print('MODEL_SAVED', name, round(time.monotonic()-start, 1), flush=True)
        assert np.array_equal(np.asarray(model.classes_).astype(int), np.arange(1,6))
        p += weight * model.predict_proba(T)
        models[name] = model
    bundle = {'features': features, 'columns': columns, 'models': models,
              'weights': winner['weights'], 'offsets': winner['offsets'], 'classes': [1,2,3,4,5]}
    joblib.dump(bundle, out / 'model_bundle.joblib', compress=3)
    np.save(out / 'test_probabilities.npy', p.astype('float32'))
    csv_path = out / '010_Constantine(12).csv'
    submission = build_submission(sample, test, decide(p, winner['offsets']), csv_path, 'numeric')
    prior = pd.read_csv('submissions/010_Constantine(10).csv', dtype={'building_id': str})
    assert prior.building_id.equals(submission.building_id)
    report = json.loads((out / 'metrics.json').read_text())
    report.update({
        'changed_vs_submission10': int((submission.damage_grade != prior.damage_grade).sum()),
        'submission_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'test_class_counts': submission.damage_grade.value_counts().sort_index().to_dict(),
        'full_refit_threads': args.threads,
        'full_refit_resumed_without_reselection': True,
    })
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    print('FINISHED', csv_path, report['submission_sha256'], flush=True)


if __name__ == '__main__':
    main()
