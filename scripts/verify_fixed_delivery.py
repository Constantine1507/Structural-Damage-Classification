"""Check saved-model reproducibility and export descriptive feature importance."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .leakage_fixed import make_fitted_pipeline
from .run_fixed_experiment import decide


def main():
    """Compare reloaded Pipeline predictions with cached probabilities on a fixed sample."""
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--output',default='artifacts/fixed_run');args=ap.parse_args()
    out=Path(args.output);bundle=joblib.load(out/'model_bundle.joblib')
    test=pd.read_csv(Path(args.data)/'test.csv',dtype={'building_id':str});idx=np.unique(np.linspace(0,len(test)-1,64).astype(int));q=test.iloc[idx]
    p=np.zeros((len(q),5));importance=np.zeros(len(bundle['columns'].get_feature_names_out()))
    for name,weight in bundle['weights'].items():
        model=bundle['models'][name];assert np.array_equal(np.asarray(model.classes_).astype(int),np.arange(1,6))
        pipe=make_fitted_pipeline(bundle['features'],bundle['columns'],model);p+=weight*pipe.predict_proba(q)
        if hasattr(model,'booster_'):v=model.booster_.feature_importance(importance_type='gain')
        else:v=np.asarray(model.feature_importances_)
        importance+=weight*v/max(v.sum(),1e-12)
    cached=np.load(out/'test_probabilities.npy')[idx]
    np.testing.assert_allclose(p,cached,atol=1e-7,rtol=1e-6)
    csv=pd.read_csv(out/'012_Constantine_leakage_fixed.csv',dtype={'building_id':str})
    assert np.array_equal(decide(p,bundle['offsets']),csv.damage_grade.to_numpy()[idx])
    result={'sample_rows':len(idx),'reloaded_pipeline_probabilities_match_cache':True,'max_absolute_difference':float(np.abs(p-cached).max()),'reloaded_pipeline_labels_match_csv':True,'classes_checked':[1,2,3,4,5]}
    (out/'bundle_verification.json').write_text(json.dumps(result,indent=2))
    table=pd.DataFrame({'feature':bundle['columns'].get_feature_names_out(),'normalized_model_importance':importance}).sort_values('normalized_model_importance',ascending=False)
    table.to_csv(out/'feature_importance.csv',index=False)
    print(json.dumps(result));print(table.head(12).to_string(index=False))

if __name__=='__main__':main()
