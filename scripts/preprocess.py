"""Input contracts and scikit-learn ColumnTransformer pipelines."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from .feature_engineering import SeismicFeatures


def read_competition_data(path="data"):
    """Load train/test/sample from a folder or ZIP (including nested ZIP paths)."""
    path = Path(path)
    required = ("train.csv", "test.csv", "sample_submission.csv")
    if path.is_file() and path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            mapping = {}
            for item in archive.namelist():
                name = Path(item).name.lower()
                if name in required:
                    if name in mapping:
                        raise ValueError(f"Duplicate {name} inside ZIP")
                    mapping[name] = item
            missing = set(required) - set(mapping)
            if missing:
                raise FileNotFoundError(f"Missing from ZIP: {sorted(missing)}")
            tables = {n: pd.read_csv(archive.open(mapping[n]), dtype={"building_id": str})
                      for n in required}
    else:
        if path.is_dir():
            names = {p.name.lower(): p for p in path.rglob("*.csv")}
            missing = set(required) - set(names)
            if missing:
                raise FileNotFoundError(f"Missing from {path}: {sorted(missing)}. Download data from Kaggle first.")
            tables = {n: pd.read_csv(names[n], dtype={"building_id": str}) for n in required}
        else:
            raise FileNotFoundError(f"Data source does not exist: {path}")
    train, test, sample = (tables[n] for n in required)
    for title, frame in (("train", train), ("test", test), ("sample", sample)):
        if "building_id" not in frame or frame.building_id.isna().any() or not frame.building_id.is_unique:
            raise ValueError(f"{title}: building_id missing, null, or duplicate")
    if "damage_grade" not in train or "damage_grade" not in sample:
        raise ValueError("Missing target 'damage_grade'")
    if "damage_grade" in test:
        raise ValueError("Test must not contain target")
    if set(train.columns) - {"damage_grade"} != set(test.columns):
        raise ValueError("Train and test feature columns differ")
    if len(test) != len(sample) or set(test.building_id) != set(sample.building_id):
        raise ValueError("Sample IDs must exactly match test IDs")
    if set(train.building_id) & set(test.building_id):
        raise ValueError("Train/test building_id overlap")
    return train, test, sample


def parse_grades(series):
    """Accept integer classes 1..5 or strings 'Grade 1'..'Grade 5'."""
    parsed = series.astype("string").str.strip().str.extract(
        r"^(?:Grade\s*)?([1-5])$", expand=False
    )
    if parsed.isna().any():
        raise ValueError(f"Unexpected damage grade values: {series[parsed.isna()].head().tolist()}")
    return parsed.astype(int).to_numpy()


class ToText(BaseEstimator, TransformerMixin):
    """Convert missing categorical features to stable strings for CatBoost."""

    def fit(self, X, y=None):
        """No learned parameters are required."""
        return self

    def transform(self, X):
        """Convert categorical input to 2D numpy string array."""
        return pd.DataFrame(X).fillna("__missing__").astype(str).to_numpy(dtype=object)


def make_preprocessor(X_reference, family="lgbm"):
    """Construct a fitted-at-training-time ColumnTransformer and category positions."""
    nums = [c for c in X_reference if pd.api.types.is_numeric_dtype(X_reference[c])]
    cats = [c for c in X_reference if c not in nums]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=False))])
    if family == "catboost":
        categorical = Pipeline([("text", ToText())])
    else:
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])
    transformers = []
    if nums:
        transformers.append(("numeric", numeric, nums))
    if cats:
        transformers.append(("categorical", categorical, cats))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0), list(range(len(nums), len(nums) + len(cats)))


def build_feature_pipeline(X_reference, estimator, family="lgbm", use_geo=True):
    """Combine feature generation, ColumnTransformer and estimator in one Pipeline."""
    feats = SeismicFeatures(use_geo=use_geo).fit_transform(X_reference)
    columns, cat_indices = make_preprocessor(feats, family)
    if family == "catboost":
        estimator.set_params(cat_features=cat_indices)
    return Pipeline([("features", SeismicFeatures(use_geo=use_geo)),
                     ("columns", columns), ("model", estimator)])


def build_submission(sample, test, predicted, output, label_style="numeric"):
    """Align predictions by building_id; assert order, values and schema before saving."""
    if len(test) != len(predicted):
        raise ValueError("Prediction length differs from test")
    lookup = pd.Series(np.asarray(predicted).astype(int), index=test.building_id.astype(str))
    if not lookup.index.is_unique:
        raise ValueError("Duplicate test IDs")
    result = pd.DataFrame({"building_id": sample.building_id.astype(str)})
    result["damage_grade"] = result.building_id.map(lookup)
    if result.damage_grade.isna().any() or not result.damage_grade.isin(range(1, 6)).all():
        raise ValueError("Submission ID mismatch or invalid grade")
    result["damage_grade"] = result.damage_grade.astype(int)
    if label_style == "grade":
        result["damage_grade"] = "Grade " + result.damage_grade.astype(str)
    elif label_style != "numeric":
        raise ValueError("label_style must be 'numeric' or 'grade'")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    reread = pd.read_csv(output, dtype={"building_id": str})
    pd.testing.assert_frame_equal(reread, result)
    return result
