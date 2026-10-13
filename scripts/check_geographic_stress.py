"""Matched-size random versus geographic split diagnostic; not a selection signal."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from lightgbm import LGBMClassifier
from .leakage_fixed import SafeFeatures, make_columns, make_fitted_pipeline
from .run_fixed_experiment import score


def main():
    """Compare fixed settings on two split designs using the same sampled population."""
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');args=ap.parse_args()
    full=pd.read_csv(Path(args.data)/'train.csv',dtype={'building_id':str})
    test_ids=pd.read_csv(Path(args.data)/'test.csv',usecols=['building_id'],dtype=str).building_id.str[:6]
    ids=full.building_id.str.zfill(12)
    mapping=pd.DataFrame({'prefix6':ids.str[:6],'ward_id':full.ward_id}).groupby('prefix6').ward_id.nunique()
    diagnostic={'prefix2_matches_district_fraction':float((ids.str[:2].astype(int)==full.district_id).mean()),
      'unique_prefix6':int(ids.str[:6].nunique()),'unique_official_ward':int(full.ward_id.nunique()),
      'prefix6_mapping_to_multiple_official_wards':int((mapping>1).sum()),
      'test_prefix6_unseen_rows':int((~test_ids.isin(ids.str[:6])).sum()),
      'test_prefix6_seen_fraction':float(test_ids.isin(ids.str[:6]).mean()),
      'note':'Prefix6 is not assumed to be the same administrative identifier as ward_id.'}
    Path('artifacts/fixed_run/id_proxy_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    x=full.sample(60000,random_state=20261015).reset_index(drop=True)
    y=x.pop('damage_grade').str.extract(r'([1-5])',expand=False).astype(int).to_numpy();groups=x.building_id.str[:6]
    ga,gb=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261015).split(x,y,groups))
    ra,rb=train_test_split(np.arange(len(x)),test_size=len(gb),stratify=y,random_state=20261015)
    result={'rows':len(x),'used_for_candidate_selection':False,'warning':'Different split designs and a smaller training population: do not interpret this gap as estimated Kaggle optimism or compare these scores directly with full-run scores.','splits':{}}
    for name,a,b in [('random',ra,rb),('group_prefix6',ga,gb)]:
        t=time.monotonic();features=SafeFeatures(seed=20261015);F=features.fit_transform(x.iloc[a],y[a]);ct,catidx=make_columns(F);A=ct.fit_transform(F,y[a]).astype('float32');B=ct.transform(features.transform(x.iloc[b])).astype('float32')
        m=LGBMClassifier(n_estimators=180,num_leaves=31,learning_rate=.04,min_child_samples=70,reg_lambda=8,n_jobs=2,verbosity=-1,random_state=20261015)
        m.fit(A,y[a],categorical_feature=catidx);pred=m.predict(B).astype(int)
        result['splits'][name]={'fit_rows':len(a),'heldout_rows':len(b),'macro_f1':score(y[b],pred),'shared_prefix6_groups':len(set(groups.iloc[a])&set(groups.iloc[b])),'heldout_class_counts':{str(c):int((y[b]==c).sum()) for c in range(1,6)},'seconds':round(time.monotonic()-t,1)}
        print(name,result['splits'][name],flush=True)
    Path('artifacts/fixed_run/geographic_stress.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
