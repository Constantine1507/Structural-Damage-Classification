"""Predict from the private fitted bundle using the public Pipeline implementation."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .leakage_fixed import make_fitted_pipeline
from .preprocess import build_submission
from .run_fixed_experiment import decide


def predict_bundle(bundle, frame):
    """Run each fitted model's Pipeline on raw rows and blend calibrated probabilities."""
    p=np.zeros((len(frame),5))
    # Transform once for speed; the equivalent raw-input pipeline is exposed below.
    transformed=bundle['columns'].transform(bundle['features'].transform(frame)).astype('float32')
    for name,weight in bundle['weights'].items():
        p+=weight*bundle['models'][name].predict_proba(transformed)
    return p,decide(p,bundle['offsets'])


def main():
    """Read test/sample and export validated integer predictions from an existing bundle."""
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundle',required=True);ap.add_argument('--test',required=True);ap.add_argument('--sample',required=True);ap.add_argument('--output',default='submission.csv');args=ap.parse_args()
    bundle=joblib.load(args.bundle);test=pd.read_csv(args.test,dtype={'building_id':str});sample=pd.read_csv(args.sample,dtype={'building_id':str})
    p,pred=predict_bundle(bundle,test)
    for name in bundle['models']:
        pipe=make_fitted_pipeline(bundle['features'],bundle['columns'],bundle['models'][name])
        direct=bundle['models'][name].predict_proba(bundle['columns'].transform(bundle['features'].transform(test.iloc[:20])).astype('float32'))
        np.testing.assert_allclose(pipe.predict_proba(test.iloc[:20]),direct,atol=1e-6)
    build_submission(sample,test,pred,Path(args.output),'numeric')

if __name__=='__main__':main()
