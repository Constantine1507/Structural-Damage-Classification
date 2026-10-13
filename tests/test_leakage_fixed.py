"""Behavioral guards against query contamination and accidental self-target lookup."""
import numpy as np
import pandas as pd
from scripts.leakage_fixed import SafeFeatures, make_columns, make_fitted_pipeline
from lightgbm import LGBMClassifier


def example():
    """Create synthetic buildings with enough examples per class for cross-fitting."""
    n=120
    return pd.DataFrame({'building_id':[f'280201{i:06d}' for i in range(n)],
       'district_id':28,'vdcmun_id':2802,'ward_id':280201,
       'count_floors_pre_eq':np.tile([1,2,3],40),'age_building':np.arange(n)%50,
       'plinth_area_sq_ft':300+np.arange(n),'height_ft_pre_eq':15,
       'land_surface_condition':'Flat','foundation_type':'Stone','roof_type':'Timber',
       'ground_floor_type':'Mud','other_floor_type':'Wood','position':'Detached',
       'plan_configuration':'Rectangular','superstructure':'Stone'}), np.tile(np.arange(1,6),24)


def test_queries_cannot_change_frozen_features():
    """Adding unrelated query rows or a fake label column must not change any original row."""
    x,y=example();f=SafeFeatures().fit(x.iloc[:90],y[:90]);q=x.iloc[90:100].copy()
    expected=f.transform(q)
    extended=pd.concat([q,x.iloc[100:]],ignore_index=True)
    pd.testing.assert_frame_equal(expected,f.transform(extended).iloc[:10].reset_index(drop=True))
    q['damage_grade']=5;pd.testing.assert_frame_equal(expected,f.transform(q))
    assert 'building_id' not in expected and 'id_numeric' not in expected


def test_single_reference_target_changes_neighbor_features():
    """Positive control: reference labels genuinely influence the supervised component."""
    x,y=example();q=x.iloc[90:]
    before=SafeFeatures.neighbor_values(x.iloc[:90],y[:90],q)
    after=SafeFeatures.neighbor_values(x.iloc[:90],np.ones(90,dtype=int),q)
    assert not np.allclose(before,after)


def test_pipeline_fit_predict_unseen_categories():
    """Pipeline must predict with unseen categories without fitting on query data."""
    x,y=example();feats=SafeFeatures();frame=feats.fit_transform(x.iloc[:90],y[:90]);columns,_=make_columns(frame)
    model=LGBMClassifier(n_estimators=8,num_leaves=7,n_jobs=1,verbosity=-1)
    pipeline=make_fitted_pipeline(SafeFeatures(),columns,model)
    pipeline.fit(x.iloc[:90],y[:90]);q=x.iloc[90:].copy();q['roof_type']='UNSEEN'
    p=pipeline.predict_proba(q)
    assert p.shape==(30,5) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
