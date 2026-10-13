"""Train-only spatial context, cross-fitted neighbor targets and reusable pipelines."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class SafeFeatures(BaseEstimator, TransformerMixin):
    """Use ID fragments only as spatial proxies; never expose full numeric IDs.

    Neighbor targets for training rows are cross-fitted. Query rows never enter
    fitted reference statistics. ID ordering is a hypothesis, not coordinates.
    """
    def __init__(self, neighbors=True, seed=20261014):
        self.neighbors=neighbors
        self.seed=seed

    def raw(self, X):
        """Construct deterministic pre-earthquake and categorical interaction features."""
        x=X.drop(columns=['building_id','damage_grade'],errors='ignore').copy().reset_index(drop=True)
        if any('post_eq' in c or 'damage' in c for c in x):
            raise ValueError('Post-event feature rejected')
        ids=X.building_id.astype(str).str.zfill(12).reset_index(drop=True)
        for c in ['district_id','vdcmun_id','ward_id']:
            x[c]=x[c].astype(str)
        for k in [4,6,8]:
            x[f'prefix{k}']=ids.str[:k]
        f=x.count_floors_pre_eq.clip(lower=1)
        h=x.height_ft_pre_eq.clip(lower=1)
        a=x.plinth_area_sq_ft.clip(lower=1)
        age=x.age_building
        x['age_999']=(age==999).astype(int)
        x['age_bin']=(age.clip(upper=200)//10).astype(str)
        x['height_per_floor']=h/f
        x['total_area']=a*f
        x['slenderness']=h/np.sqrt(a)
        x['log_area']=np.log1p(a)
        x['log_age']=np.log1p(age.clip(upper=200))
        x['area_per_height']=a/h
        for name, cols in {
            'ward_material':['prefix6','superstructure'],
            'ward_foundation':['prefix6','foundation_type'],
            'newward_material':['ward_id','superstructure'],
            'mun_material':['vdcmun_id','superstructure'],
            'material_foundation':['superstructure','foundation_type'],
            'material_roof':['superstructure','roof_type'],
            'foundation_ground':['foundation_type','ground_floor_type'],
            'material_floors':['superstructure','count_floors_pre_eq']}.items():
            x[name]=x[cols].astype(str).agg('|'.join,axis=1)
        return x

    def fit(self, X, y=None):
        """Freeze all group statistics and neighbor references using this fit partition."""
        x=self.raw(X)
        self.stats_={}
        for key in ['prefix6','ward_id','ward_material']:
            self.stats_[key]=x.groupby(key)[['log_age','log_area','height_per_floor','count_floors_pre_eq']].agg(['mean','std'])
            self.stats_[key]['count']=x.groupby(key).size()
        self.ref_=X.reset_index(drop=True).copy()
        self.y_=None if y is None else np.asarray(y).copy()
        return self

    def context(self, X):
        """Map frozen train-only context; unknown groups receive missing values or zero counts."""
        x=self.raw(X)
        for key, table in self.stats_.items():
            for c in ['log_age','log_area','height_per_floor','count_floors_pre_eq']:
                mean=x[key].map(table[c]['mean'])
                x[f'{key}_{c}_mean']=mean
                x[f'{key}_{c}_dev']=x[c]-mean
            x[f'{key}_log_count']=np.log1p(x[key].map(table['count']).fillna(0))
        return x

    @staticmethod
    def neighbor_values(ref, labels, query):
        """Smooth reference labels over nearby ID ordering within the same old ward.

        No query labels, global target statistics or test rows are used as references.
        Bandwidths and categorical penalties are fixed before model comparison.
        """
        rid=ref.building_id.astype(str).str.zfill(12)
        qid=query.building_id.astype(str).str.zfill(12)
        groups=pd.Index(rid.str[:6].unique())
        rg=groups.get_indexer(rid.str[:6]);qg=groups.get_indexer(qid.str[:6])
        rs=rid.str[6:].astype(np.int64).to_numpy();qs=qid.str[6:].astype(np.int64).to_numpy()
        order=np.argsort(rg.astype(np.int64)*1000001+rs,kind='stable')
        positions=rg[order].astype(np.int64)*1000001+rs[order]
        cols=['superstructure','foundation_type','roof_type','ground_floor_type','other_floor_type']
        rcat=ref[cols].to_numpy();qcat=query[cols].to_numpy()
        rn=np.column_stack([np.log1p(ref.age_building.clip(upper=200)),np.log1p(ref.plinth_area_sq_ft.clip(lower=1)),ref.count_floors_pre_eq,ref.height_ft_pre_eq/ref.count_floors_pre_eq.clip(lower=1)])
        qn=np.column_stack([np.log1p(query.age_building.clip(upper=200)),np.log1p(query.plinth_area_sq_ft.clip(lower=1)),query.count_floors_pre_eq,query.height_ft_pre_eq/query.count_floors_pre_eq.clip(lower=1)])
        scale=np.maximum(rn.std(0),.1)
        rn=rn/scale;qn=qn/scale
        prior=np.bincount(labels-1,minlength=5)/len(labels)
        out=np.empty((len(query),36),dtype=np.float32)
        for start in range(0,len(query),512):
            end=min(start+512,len(query));g=qg[start:end]
            p=np.searchsorted(positions,g.astype(np.int64)*1000001+qs[start:end])
            ix=p[:,None]+np.arange(-64,64)
            valid=(ix>=0)&(ix<len(order))
            r=order[np.clip(ix,0,len(order)-1)]
            valid &= (rg[r]==g[:,None])&(g[:,None]>=0)
            distance=np.abs(rs[r]-qs[start:end,None]).astype(float)
            distance[~valid]=np.inf
            physical=(((rn[r]-qn[start:end,None,:])**2)*np.array([.35,.25,.9,.35])).sum(2)
            for j,weight in enumerate([1.5,.7,.5,.5,.5]):
                physical+=(rcat[r,j]!=qcat[start:end,None,j])*weight
            offset=0
            for bw in [10.,100.,1000.]:
                for power in [0.,2.]:
                    w=1/(1+distance/bw)/(1+physical)**power
                    total=w.sum(1)
                    for cls in range(5):
                        out[start:end,offset+cls]=((w*(labels[r]==cls+1)).sum(1)+2*prior[cls])/(total+2)
                    out[start:end,offset+5]=np.log1p(total)
                    offset+=6
        return out

    def fit_transform(self, X, y=None, **fit_params):
        """Cross-fit supervised neighbor features; each row's own fold is excluded."""
        self.fit(X,y)
        result=self.context(X)
        if self.neighbors:
            z=np.zeros((len(X),36),np.float32)
            for a,b in StratifiedKFold(3,shuffle=True,random_state=self.seed).split(X,y):
                z[b]=self.neighbor_values(X.iloc[a],np.asarray(y)[a],X.iloc[b])
            result=pd.concat([result,pd.DataFrame(z,columns=[f'neighbor_{i}' for i in range(36)])],axis=1)
        return result

    def transform(self, X):
        """Generate features with fixed references; changing the query batch cannot alter a row."""
        result=self.context(X)
        if self.neighbors:
            z=self.neighbor_values(self.ref_,self.y_,X)
            result=pd.concat([result,pd.DataFrame(z,columns=[f'neighbor_{i}' for i in range(36)])],axis=1)
        return result


def make_columns(frame):
    """Fit category dictionaries on training only and cross-fit multiclass target encoding."""
    nums=frame.select_dtypes(include='number').columns.tolist()
    cats=[c for c in frame if c not in nums]
    return ColumnTransformer([
        ('numeric',SimpleImputer(strategy='median'),nums),
        ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),cats),
        ('target',TargetEncoder(target_type='multiclass',smooth=30,cv=3,shuffle=True,random_state=20261014),cats)
    ],sparse_threshold=0), list(range(len(nums),len(nums)+len(cats)))


class ToFloat32(BaseEstimator, TransformerMixin):
    """Preserve the numerical precision used during the cached training experiment."""
    def fit(self, X, y=None):
        """No parameters are learned."""
        return self

    def transform(self, X):
        """Cast the matrix without learning from query values."""
        return np.asarray(X, dtype=np.float32)


def make_fitted_pipeline(features, columns, model):
    """Assemble the fitted preprocessing and estimator for reproducible raw-data prediction."""
    return Pipeline([('features',features),('columns',columns),('float32',ToFloat32()),('model',model)])
