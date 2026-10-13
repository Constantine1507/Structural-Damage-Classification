"""Frozen non-neural experiment: fit / calibration / selection / audit / full refit."""
import argparse, hashlib, json, time, gc
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn, lightgbm
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from .preprocess import read_competition_data, parse_grades, build_submission
from .leakage_fixed import SafeFeatures, make_columns


def score(y,pred):
    """Macro F1 over exactly five classes."""
    m=np.bincount((np.asarray(y)-1)*5+np.asarray(pred)-1,minlength=25).reshape(5,5)
    return float(np.mean(2*np.diag(m)/np.maximum(m.sum(0)+m.sum(1),1)))


def decide(p,b):
    """Predict with class offsets fitted separately from selection and audit."""
    return (np.log(np.clip(p,1e-12,1))+b).argmax(1)+1


def offsets(p,y):
    """Bounded coordinate search on calibration labels only."""
    b=np.zeros(5);best=score(y,decide(p,b))
    for step in [.2,.1,.05,.02]:
        for _ in range(2):
            for c in range(5):
                for sign in [-1,1]:
                    trial=b.copy();trial[c]+=sign*step;trial-=trial.mean()
                    s=score(y,decide(p,trial))
                    if s>best+.00005: b,best=trial,s
    return b


def main():
    """Select on one partition and evaluate the frozen choice once on another."""
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',default='artifacts/fixed_run');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train,test,sample=read_competition_data(args.data);y=parse_grades(train.damage_grade);X=train.drop(columns='damage_grade');seed=20261014
    fit,other=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
    ca,rem=train_test_split(other,test_size=.65,stratify=y[other],random_state=seed+1)
    sel,audit=train_test_split(rem,test_size=6/13,stratify=y[rem],random_state=seed+2)
    np.savez_compressed(out/'splits.npz',fit=fit,calibration=ca,selection=sel,audit=audit)
    defs={'lgb47':dict(num_leaves=47,learning_rate=.035,min_child_samples=70,reg_lambda=8),
      'lgb95':dict(num_leaves=95,learning_rate=.025,min_child_samples=100,reg_lambda=12),
      'lgb31_weighted':dict(num_leaves=31,learning_rate=.04,min_child_samples=90,reg_lambda=10,class_weight={int(c):float((len(y[fit])/(5*(y[fit]==c).sum()))**.5) for c in range(1,6)})}
    protocol={'seed':seed,'rows':{k:len(v) for k,v in [('fit',fit),('calibration',ca),('selection',sel),('audit',audit)]},'candidates':defs,'catboost':{'iterations':1100,'depth':8,'learning_rate':.05},'n_estimators_max':1800,'early_stopping_on_calibration':100,'audit_used_for_selection':False,'historically_pristine_holdout_claimed':False,'test_covariates_used_in_fit':False,'full_numeric_id_feature':False,'neighbor_id_fragments_used':True,'data_sha256':{n:hashlib.sha256((Path(args.data)/n).read_bytes()).hexdigest() for n in ['train.csv','test.csv']},'versions':{'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));print('PROTOCOL_FROZEN',protocol['rows'],flush=True)
    features=SafeFeatures(seed=seed);print('CROSSFIT_FEATURES_START',flush=True)
    F=features.fit_transform(X.iloc[fit],y[fit]);ct,catidx=make_columns(F)
    A=ct.fit_transform(F,y[fit]).astype('float32');del F;gc.collect()
    B=ct.transform(features.transform(X.iloc[ca])).astype('float32')
    C=ct.transform(features.transform(X.iloc[sel])).astype('float32')
    D=ct.transform(features.transform(X.iloc[audit])).astype('float32');print('FEATURES_READY',A.shape,flush=True)
    q=X.iloc[audit[:25]].copy();before=features.transform(q);altered=q.copy();altered['damage_grade']=5
    pd.testing.assert_frame_equal(before,features.transform(altered))
    extra=pd.concat([q,X.iloc[fit[:5]]],ignore_index=True)
    pd.testing.assert_frame_equal(before,features.transform(extra).iloc[:len(q)].reset_index(drop=True))
    assert 'id_numeric' not in before and 'building_id' not in before
    (out/'invariants.json').write_text(json.dumps({'query_label_column_invariance':True,'query_batch_invariance':True,'no_full_id_feature':True},indent=2))
    pca={};psel={};paudit={};rounds={}
    for name,params in defs.items():
        print('FIT_START',name,flush=True);t=time.monotonic()
        m=LGBMClassifier(**params,n_estimators=1800,objective='multiclass',verbosity=-1,n_jobs=4,random_state=seed,colsample_bytree=.85,subsample=.9,subsample_freq=1)
        m.fit(A,y[fit],categorical_feature=catidx,eval_set=[(B,y[ca])],callbacks=[early_stopping(100,verbose=False),log_evaluation(300)])
        pca[name]=m.predict_proba(B);psel[name]=m.predict_proba(C);paudit[name]=m.predict_proba(D);rounds[name]=m.best_iteration_
        print('FIT_DONE',name,'rounds',rounds[name],'seconds',round(time.monotonic()-t,1),'CAL_RAW',score(y[ca],decide(pca[name],np.zeros(5))),flush=True)
        np.savez_compressed(out/f'{name}_validation.npz',cal=pca[name],selection=psel[name],audit=paudit[name]);del m;gc.collect()
    name='cat_numeric';print('FIT_START',name,flush=True)
    m=CatBoostClassifier(iterations=1100,depth=8,learning_rate=.05,loss_function='MultiClass',l2_leaf_reg=8,thread_count=4,random_seed=seed,allow_writing_files=False,verbose=250)
    m.fit(A,y[fit],eval_set=(B,y[ca]),early_stopping_rounds=100)
    pca[name]=m.predict_proba(B);psel[name]=m.predict_proba(C);paudit[name]=m.predict_proba(D);rounds[name]=m.tree_count_
    np.savez_compressed(out/f'{name}_validation.npz',cal=pca[name],selection=psel[name],audit=paudit[name])
    names=list(pca);mixtures=[{n:1.} for n in names]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for w in [.25,.5,.75]:mixtures.append({a:w,b:1-w})
    mixtures.append({n:1/len(names) for n in names});results=[]
    for weights in mixtures:
        pc=sum(w*pca[n] for n,w in weights.items());pv=sum(w*psel[n] for n,w in weights.items());calibrated=offsets(pc,y[ca])
        for strength in [0.,.5,1.]:
            b=calibrated*strength;results.append({'weights':weights,'offsets':b.tolist(),'cal_f1':score(y[ca],decide(pc,b)),'selection_f1':score(y[sel],decide(pv,b))})
    results.sort(key=lambda r:r['selection_f1'],reverse=True);winner=results[0]
    (out/'frozen_selection.json').write_text(json.dumps({'winner':winner,'rounds':rounds,'all_candidates':results},indent=2));print('SELECTION_FROZEN',winner,flush=True)
    p=sum(w*paudit[n] for n,w in winner['weights'].items());prediction=decide(p,winner['offsets']);baseline=decide(paudit['lgb47'],np.zeros(5))
    report={'winner':winner,'audit_f1':score(y[audit],prediction),'audit_raw_lgb47_f1':score(y[audit],baseline),'audit_classification':classification_report(y[audit],prediction,labels=[1,2,3,4,5],output_dict=True),'audit_confusion_matrix':confusion_matrix(y[audit],prediction,labels=[1,2,3,4,5]).tolist(),'public_kaggle_score':None,'previous_public_best_user_report':.60627,'rounds':rounds}
    groups=X.iloc[audit].building_id.str[:6].to_numpy();codes,inv=np.unique(groups,return_inverse=True);cnt=np.zeros((len(codes),2,25),dtype=np.int32)
    for j,pp in enumerate([prediction,baseline]):np.add.at(cnt[:,j],(inv,(y[audit]-1)*5+pp-1),1)
    rng=np.random.default_rng(seed);diff=[]
    for _ in range(500):
        cms=cnt[rng.integers(0,len(codes),len(codes))].sum(0).reshape(2,5,5);s=[np.mean(2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)) for cm in cms];diff.append(s[0]-s[1])
    report['paired_group_bootstrap_delta_ci95']=np.quantile(diff,[.025,.975]).tolist()
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print('AUDIT_RESULT',report['audit_f1'],report['paired_group_bootstrap_delta_ci95'],flush=True)
    np.savez_compressed(out/'audit_predictions.npz',indices=audit,y=y[audit],probabilities=p,predicted=prediction,baseline=baseline)
    del A,B,C,D,ct,features,m;gc.collect();print('FULL_REFIT_FEATURES',flush=True)
    features=SafeFeatures(seed=seed);F=features.fit_transform(X,y);ct,catidx=make_columns(F);A=ct.fit_transform(F,y).astype('float32');del F;gc.collect()
    T=ct.transform(features.transform(test)).astype('float32');ptest=np.zeros((len(test),5));models={}
    for name,w in winner['weights'].items():
        print('FULL_REFIT',name,rounds[name],flush=True)
        if name in defs:
            m=LGBMClassifier(**defs[name],n_estimators=rounds[name],objective='multiclass',verbosity=-1,n_jobs=4,random_state=seed,colsample_bytree=.85,subsample=.9,subsample_freq=1);m.fit(A,y,categorical_feature=catidx)
        else:
            m=CatBoostClassifier(iterations=rounds[name],depth=8,learning_rate=.05,loss_function='MultiClass',l2_leaf_reg=8,thread_count=4,random_seed=seed,allow_writing_files=False,verbose=250);m.fit(A,y)
        ptest+=w*m.predict_proba(T);models[name]=m
    outcsv=out/'012_Constantine_leakage_fixed.csv';result=build_submission(sample,test,decide(ptest,winner['offsets']),outcsv,'numeric');np.save(out/'test_probabilities.npy',ptest.astype('float32'))
    joblib.dump({'features':features,'columns':ct,'models':models,'weights':winner['weights'],'offsets':winner['offsets'],'classes':[1,2,3,4,5]},out/'model_bundle.joblib',compress=3)
    prior=pd.read_csv('submissions/010_Constantine(10).csv',dtype={'building_id':str});assert result.building_id.equals(prior.building_id)
    report['changed_vs_submission10']=int((result.damage_grade!=prior.damage_grade).sum());report['submission_sha256']=hashlib.sha256(outcsv.read_bytes()).hexdigest();report['test_class_counts']=result.damage_grade.value_counts().sort_index().to_dict()
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print('FINISHED',outcsv,report['submission_sha256'],flush=True)

if __name__=='__main__':main()
