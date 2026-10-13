"""Small deterministic tests for target parsing, transformers and submission contracts."""
import numpy as np
import pandas as pd
from scripts.feature_engineering import SeismicFeatures
from scripts.preprocess import parse_grades, build_submission, make_preprocessor


def test_grades():
    """Integer and Grade labels map to the same 1..5 targets."""
    assert parse_grades(pd.Series(["Grade 1","2","Grade 3","4","Grade 5"])).tolist() == [1,2,3,4,5]


def test_geo_is_fit_only():
    """An unseen test prefix must get zero reference count, not target information."""
    train = pd.DataFrame({"building_id":["123456000001", "123456000002"], "age_building":[3,4]})
    test = pd.DataFrame({"building_id":["998877000001"],"age_building":[5]})
    tf = SeismicFeatures().fit(train)
    assert tf.transform(test)["id_prefix_6_reference_count"].iloc[0] == 0


def test_column_transformer():
    """ColumnTransformer makes numerical matrices with unseen categorical values."""
    frame = pd.DataFrame({"age":[3., None, 7.], "material":["brick","wood","brick"]})
    proc, _ = make_preprocessor(frame)
    proc.fit(frame)
    arr = proc.transform(pd.DataFrame({"age":[2.],"material":["unseen"]}))
    assert arr.shape == (1,2)
    assert np.isfinite(arr.astype(float)).all()


def test_submission(tmp_path):
    """Output must retain sample ID order with exactly five allowed integer classes."""
    test = pd.DataFrame({"building_id":["123","456"]})
    sample = pd.DataFrame({"building_id":["456","123"], "damage_grade":[0,0]})
    result = build_submission(sample, test, [1,5], tmp_path/"submission.csv")
    assert result.damage_grade.tolist() == [5,1]
    assert list(result) == ["building_id","damage_grade"]
