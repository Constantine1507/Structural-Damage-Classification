"""Audit original feature code with label-sentinel tests and raw data contracts.

The notebook is parsed, never executed as a whole. Only function definitions and
four literal feature constants are loaded for isolated tests.
"""
import ast
import argparse
import json
import hashlib
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold,train_test_split


def main():
    """Run deterministic checks against supplied original files and two final CSVs."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--submissions',default='submissions')
    parser.add_argument('--output',default='reports/deep_audit/check_results.json')
    args=parser.parse_args()
    data=Path(args.data)
    trainpath=next(data.rglob('train.csv'));testpath=next(data.rglob('test.csv'))
    d=pd.read_csv(trainpath,dtype={'building_id':str});t=pd.read_csv(testpath,dtype={'building_id':str})
    target=d.pop('damage_grade');y=target.astype(str).str.extract(r'([1-5])$',expand=False).astype(int).to_numpy()-1
    report={'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (trainpath,testpath)},
       'train_rows':len(d),'test_rows':len(t),'missing_train':int(d.isna().sum().sum()),'missing_test':int(t.isna().sum().sum()),
       'duplicate_train_ids':int(d.building_id.duplicated().sum()),'duplicate_test_ids':int(t.building_id.duplicated().sum()),
       'id_overlap':len(set(d.building_id)&set(t.building_id)),
       'post_eq_columns':[c for c in d if 'post_eq' in c or 'damage' in c],
       'test_geo_unseen':{c:int((~t[c].isin(d[c])).sum()) for c in ['district_id','vdcmun_id','ward_id']}}
    features=[c for c in d if c!='building_id']
    keys=pd.util.hash_pandas_object(d[features],index=False)
    testkeys=pd.util.hash_pandas_object(t[features],index=False)
    groups=pd.DataFrame({'key':keys,'y':y}).groupby('key').agg(n=('y','size'),classes=('y','nunique'))
    conflicts=groups[groups.classes>1]
    report['exact_profile_duplicate_rows_all']=int(d.duplicated(features,keep=False).sum())
    report['conflicting_profile_groups']=len(conflicts)
    report['conflicting_profile_rows']=int(conflicts.n.sum())
    report['test_profile_seen_in_train_rows']=int(testkeys.isin(keys).sum())
    report['test_prefix11_seen_in_train_rows']=int(t.building_id.str[:11].isin(d.building_id.str[:11]).sum())
    notebook=json.loads(Path(args.notebook).read_text())
    source='\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    ns={'np':np,'pd':pd,'cdist':cdist,'cKDTree':cKDTree,'StratifiedKFold':StratifiedKFold,'train_test_split':train_test_split,'time':time,'gc':gc}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for key in node.targets:
                if isinstance(key,ast.Name) and key.id in ['COMPACT','PAIR_SEED','SEED7','SEED']:
                    ns[key.id]=ast.literal_eval(node.value)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited notebook definitions>','exec'),ns)
    indices=np.random.default_rng(20261013).choice(len(d),size=2000,replace=False)
    small=d.iloc[indices].reset_index(drop=True);ys=y[indices].copy()
    fit,query=train_test_split(np.arange(len(small)),test_size=.25,stratify=ys,random_state=42)
    altered=ys.copy();altered[query]=(altered[query]+1)%5
    checks=[]
    for name in ['compact_frame','kernel_features','physical_kernels','pair_features','hierarchical_profiles','numeric_neighbours']:
        before=ns[name](small,ys,fit)
        after=ns[name](small,altered,fit)
        pd.testing.assert_frame_equal(before,after,check_exact=True)
        checks.append({'builder':name,'rows':len(small),'fit_rows':len(fit),'heldout_rows':len(query),'changing_all_heldout_labels_changes_features':False,'columns':len(before.columns)})
        print('SENTINEL_PASS',name,flush=True)
    report['heldout_label_sentinel_tests']=checks
    original=ns['contextual_covariates'](small)
    extra=small.iloc[[0]].copy();extra['age_building']=199;extra['plinth_area_sq_ft']*=10
    extended=ns['contextual_covariates'](pd.concat([small,extra],ignore_index=True)).iloc[:len(small)]
    report['adding_unlabelled_query_changes_context_training_rows']=int((original!=extended).any(axis=1).sum())
    report['original_compact_includes_full_numeric_id']='id_numeric' in ns['COMPACT']
    files={'submission10':'010_Constantine(10).csv','candidate11':'011_Constantine_candidate.csv'}
    csvs={key:pd.read_csv(Path(args.submissions)/name,dtype={'building_id':str}) for key,name in files.items()}
    a,b=csvs['submission10'],csvs['candidate11']
    assert a.building_id.equals(t.building_id) and b.building_id.equals(t.building_id)
    changed=a.damage_grade!=b.damage_grade
    report['changed_rows']=int(changed.sum());report['changed_fraction']=float(changed.mean())
    report['changed_transitions']=[{'from':int(x),'to':int(z),'rows':int(n)} for (x,z),n in pd.DataFrame({'old':a.loc[changed,'damage_grade'],'new':b.loc[changed,'damage_grade']}).value_counts().items()]
    report['changed_by_ground_floor']=t.loc[changed,'ground_floor_type'].value_counts().to_dict()
    report['grade_count_delta']={str(k):int(b.damage_grade.eq(k).sum()-a.damage_grade.eq(k).sum()) for k in range(1,6)}
    report['user_reported_public']={'submission10':.60627,'candidate11':.60621,'delta':-.00006}
    report['changed_floor_rates']={k:{'all_test_rows':int(t.ground_floor_type.eq(k).sum()),'changed_rows':n,'fraction_changed':n/int(t.ground_floor_type.eq(k).sum())} for k,n in report['changed_by_ground_floor'].items()}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
