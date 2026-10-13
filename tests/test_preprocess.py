"""Small deterministic tests for target parsing, transformers and submission contracts."""
import numpy as np
import pandas as pd
import pytest
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


def test_assignment_numeric_default(tmp_path):
    """Grade-string placeholders must not override the required integer schema."""
    sample = pd.DataFrame({"building_id": ["002", "001"], "damage_grade": ["Grade 1", "Grade 2"]})
    test = pd.DataFrame({"building_id": ["001", "002"]})
    result = build_submission(sample, test, [2, 5], tmp_path / "numeric.csv")
    assert result.damage_grade.tolist() == [5, 2]
    assert result.building_id.tolist() == ["002", "001"]
    with pytest.raises(ValueError, match="integer-valued"):
        build_submission(sample, test, [1.9, 5], tmp_path / "invalid.csv")
    with pytest.raises(ValueError, match="Sample IDs"):
        build_submission(sample.iloc[[0, 0]], test, [2, 5], tmp_path / "duplicate.csv")


def test_no_geography_ablation():
    """The no-geography control excludes explicit geography as well as ID prefixes."""
    frame = pd.DataFrame({"building_id": ["123456000001"], "district_id": [12],
                          "vdcmun_id": [1234], "ward_id": [123456], "age_building": [5]})
    transformed = SeismicFeatures(use_geo=False).fit_transform(frame)
    assert not any(c in transformed for c in ("building_id", "district_id", "vdcmun_id", "ward_id"))
    assert not any(c.startswith("id_prefix_") for c in transformed)
