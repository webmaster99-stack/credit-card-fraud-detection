from types import SimpleNamespace

import pandas as pd

from conftest import make_clean_frame
from fraud.serving.export import (
    EXAMPLE_COLUMNS,
    build_cities,
    build_metadata,
    render_model_card,
    sample_examples,
)


def test_build_cities_one_row_per_city_with_label() -> None:
    df = make_clean_frame(n=200).assign(state="IL")
    df.loc[:99, "city"] = "Peoria"
    cities = build_cities(df)

    assert list(cities.columns) == ["label", "city", "state", "lat", "long", "city_pop"]
    assert cities["label"].tolist() == ["Peoria, IL", "Springfield, IL"]
    assert cities["label"].is_unique


def test_sample_examples_takes_both_classes_reproducibly() -> None:
    df = make_clean_frame(n=400)
    a = sample_examples(df, per_class=5, seed=42)
    b = sample_examples(df, per_class=5, seed=42)

    assert list(a.columns) == EXAMPLE_COLUMNS
    assert a["is_fraud"].value_counts().to_dict() == {1: 5, 0: 5}
    pd.testing.assert_frame_equal(a, b)


def test_sample_examples_never_asks_for_more_rows_than_a_class_has() -> None:
    df = make_clean_frame(n=50).assign(is_fraud=0)
    df.loc[0, "is_fraud"] = 1
    assert sample_examples(df, per_class=10, seed=42)["is_fraud"].sum() == 1


def test_metadata_and_model_card_state_lineage_and_who_owns_the_test_metrics() -> None:
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="abc123"),
        data=SimpleNamespace(
            tags={"git_commit": "deadbee", "dataset_name": "sparkov", "dvc_data_md5": "md5.dir"},
            params={"step": "xgboost", "feature_set": "v1", "calibration": "sigmoid"},
            metrics={"precision": 0.5, "recall": 0.921},
        ),
    )
    meta = build_metadata(run, "3", "demo", threshold=0.0123)
    card = render_model_card(meta)

    assert meta["model_version"] == "3" and meta["alias"] == "demo"
    assert meta["dataset_version"] == "n/a"  # a missing lineage tag is stated, not invented
    for expected in ("deadbee", "md5.dir", "abc123", "0.0123", "0.921", "xgboost"):
        assert expected in card
    assert "has **not** been evaluated on the test split" in card
    assert "belongs to the registry champion" in card
