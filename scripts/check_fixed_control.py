"""Small label-permutation control; a diagnostic, not a proof of no leakage."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from .leakage_fixed import SafeFeatures, make_columns, make_fitted_pipeline
from .run_fixed_experiment import score


def main():
    """Train a complete pipeline on shuffled labels and evaluate held-out shuffled labels."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',default='data');args=parser.parse_args()
    data=pd.read_csv(Path(args.data)/'train.csv',dtype={'building_id':str}).sample(8000,random_state=20261014)
    y=data.pop('damage_grade').str.extract(r'([1-5])',expand=False).astype(int).to_numpy()
    y=np.random.default_rng(20261014).permutation(y)
    a,b=train_test_split(np.arange(len(data)),test_size=.25,random_state=20261014,stratify=y)
    frame=SafeFeatures().fit_transform(data.iloc[a],y[a]);columns,_=make_columns(frame)
    pipe=make_fitted_pipeline(SafeFeatures(),columns,LGBMClassifier(n_estimators=100,num_leaves=15,n_jobs=1,verbosity=-1,random_state=20261014,class_weight='balanced'))
    pipe.fit(data.iloc[a],y[a]);pred=pipe.predict(data.iloc[b]).astype(int)
    report={'rows':len(data),'fit_rows':len(a),'heldout_rows':len(b),'labels_permuted_before_split':True,'macro_f1':score(y[b],pred),'majority_macro_f1':score(y[b],np.full(len(b),np.bincount(y[a]).argmax())),'interpretation':'Low heldout score is consistent with absence of an obvious target shortcut in this small control; it does not certify every code path.'}
    Path('artifacts/fixed_run/permutation_control.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
