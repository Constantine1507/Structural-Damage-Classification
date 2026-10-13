"""Target-independent building, material and geographic feature construction."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SeismicFeatures(BaseEstimator, TransformerMixin):
    """Create seismic-vulnerability proxies without using labels or test statistics.

    Full building IDs are never passed to a classifier. The shorter prefixes
    are optional spatial proxies; evaluate with and without them to detect
    reliance on memorized geographic patterns.
    """

    def __init__(self, use_geo=True):
        """Set whether coarse ID prefixes and reference-frequency features are enabled."""
        self.use_geo = use_geo

    def fit(self, X, y=None):
        """Learn unsupervised prefix counts using only the training partition."""
        ids = X["building_id"].astype("string").fillna("").str.zfill(12)
        self.freq_ = {}
        if self.use_geo:
            for n in (4, 6):
                self.freq_[n] = ids.str[:n].value_counts().to_dict()
        return self

    def transform(self, X):
        """Return a fresh DataFrame with safe ratios, counts and material flags."""
        x = X.copy().reset_index(drop=True)
        ids = x.pop("building_id").astype("string").fillna("").str.zfill(12)
        for name in list(x.columns):
            if name.startswith("has_"):
                x[name] = pd.to_numeric(x[name], errors="coerce")
        numeric = ("age_building", "plinth_area_sq_ft", "height_ft_pre_eq",
                   "count_floors_pre_eq", "count_floors_post_eq")
        for name in numeric:
            if name in x:
                x[name] = pd.to_numeric(x[name], errors="coerce")
        if "count_floors_post_eq" in x:
            # Post-earthquake attributes could leak damage severity.
            x = x.drop(columns=["count_floors_post_eq"])
        if "height_ft_post_eq" in x:
            x = x.drop(columns=["height_ft_post_eq"])
        pre = x.get("count_floors_pre_eq", pd.Series(np.nan, index=x.index))
        height = x.get("height_ft_pre_eq", pd.Series(np.nan, index=x.index))
        area = x.get("plinth_area_sq_ft", pd.Series(np.nan, index=x.index))
        age = x.get("age_building", pd.Series(np.nan, index=x.index))
        x["pre_height_per_floor"] = height / pre.replace(0, np.nan)
        x["plinth_per_floor"] = area / pre.replace(0, np.nan)
        x["age_log"] = np.log1p(age.clip(lower=0))
        x["plinth_log"] = np.log1p(area.clip(lower=0))
        x["age_x_floors"] = age * pre
        flag_cols = [c for c in x if c.startswith("has_superstructure_")]
        if flag_cols:
            x["n_superstructure_types"] = x[flag_cols].fillna(0).sum(axis=1)
        if self.use_geo:
            for n in (2, 4, 6):
                x[f"id_prefix_{n}"] = ids.str[:n].astype(str)
            for n in (4, 6):
                x[f"id_prefix_{n}_reference_count"] = (
                    ids.str[:n].map(self.freq_[n]).fillna(0).astype(float)
                )
        return x.replace([np.inf, -np.inf], np.nan)
