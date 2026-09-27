"""Pipeline-level tests: fitting rules, output contract and train/serve parity."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_clean_frame
from fraud.features import PIPELINE_VERSION
from fraud.features.pipeline import (
    FEATURE_SETS,
    build_pipeline,
    feature_list,
    transform_with_context,
)

TRAIN_ROWS = 250


@pytest.fixture
def split_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = make_clean_frame(n=400)
    return df.iloc[:TRAIN_ROWS].reset_index(drop=True), df.iloc[TRAIN_ROWS:].reset_index(drop=True)


def test_pipeline_version_is_semver_tagged() -> None:
    assert PIPELINE_VERSION.startswith("features-")
    major, minor, patch = PIPELINE_VERSION.removeprefix("features-").split(".")
    assert all(part.isdigit() for part in (major, minor, patch))


def test_unknown_feature_set_is_rejected(features_cfg: dict) -> None:
    with pytest.raises(ValueError, match="Unknown feature set"):
        build_pipeline("v3", features_cfg)


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_output_is_finite_unique_and_matches_feature_list(
    feature_set: str, features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, valid = split_frames
    pipe = build_pipeline(feature_set, features_cfg)
    out = pipe.fit_transform(train)
    assert len(out) == len(train)
    assert out.columns.is_unique
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    listing = feature_list(pipe, out, feature_set)
    assert [f["name"] for f in listing["features"]] == list(out.columns)
    assert listing["pipeline_version"] == PIPELINE_VERSION
    assert list(pipe.transform(valid).columns) == list(out.columns)


def test_v2_is_v1_plus_history_features(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, _ = split_frames
    v1 = build_pipeline("v1", features_cfg).fit(train).get_feature_names_out()
    v2 = build_pipeline("v2", features_cfg).fit(train).get_feature_names_out()
    assert set(v1) < set(v2)
    assert {"txn_count_1h", "amt_vs_card_mean", "is_first_category_use"} <= set(v2) - set(v1)


def test_scalers_and_encoders_are_fitted_on_train_only(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, valid = split_frames
    shifted_valid = valid.assign(amt=valid["amt"] * 1000, state="ZZ")
    a = build_pipeline("v1", features_cfg).fit(train)
    b = build_pipeline("v1", features_cfg).fit(train)
    # Fitting never looks at other data; transforming does not refit.
    a.transform(shifted_valid)
    pd.testing.assert_frame_equal(a.transform(train), b.transform(train))
    scaler = a.named_steps["encode"].named_transformers_["num"].named_steps["scale"]
    expected_mean = np.log1p(train["amt"]).mean()
    assert scaler.mean_[0] == pytest.approx(expected_mean)


def test_unseen_category_and_state_encode_to_zeros(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, valid = split_frames
    pipe = build_pipeline("v1", features_cfg).fit(train)
    row = valid.iloc[[0]].assign(category="crypto_atm", state="ZZ")
    out = pipe.transform(row)
    onehot = [c for c in out.columns if c.startswith(("category_", "state_"))]
    assert onehot
    assert (out[onehot].to_numpy() == 0).all()


def test_missing_raw_column_fails_loudly(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, valid = split_frames
    pipe = build_pipeline("v2", features_cfg).fit(train)
    with pytest.raises(ValueError, match="missing from the input"):
        pipe.transform(valid.drop(columns=["card_id"]))


def test_v1_parity_batch_vs_one_row_at_a_time(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Train/serve parity (stateless): scoring a row alone equals scoring it in a batch."""
    train, valid = split_frames
    pipe = build_pipeline("v1", features_cfg).fit(train)
    batch = pipe.transform(valid)
    single = pd.concat([pipe.transform(valid.iloc[[i]]) for i in range(len(valid))])
    pd.testing.assert_frame_equal(batch, single, rtol=1e-9, atol=1e-12)


def test_v2_parity_offline_batch_vs_online_with_stored_history(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Train/serve parity (stateful): a live row scored against stored earlier rows matches the
    offline computation over the whole time-ordered table."""
    train, valid = split_frames
    pipe = build_pipeline("v2", features_cfg).fit(train)
    offline = pipe.transform(pd.concat([train, valid], ignore_index=True)).iloc[len(train) :]
    offline = offline.reset_index(drop=True)

    online_rows = []
    history = pd.concat([train], ignore_index=True)
    for i in range(len(valid)):
        row = valid.iloc[[i]]
        online_rows.append(transform_with_context(pipe, row, context=history))
        history = pd.concat([history, row], ignore_index=True)  # the store gains the new txn
    online = pd.concat(online_rows, ignore_index=True)
    pd.testing.assert_frame_equal(offline, online, rtol=1e-9, atol=1e-9)


def test_v2_csv_batch_scores_within_the_file_in_any_row_order(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, valid = split_frames
    pipe = build_pipeline("v2", features_cfg).fit(train)
    expected = pipe.transform(valid)
    shuffled = valid.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(pipe.transform(shuffled).loc[valid.index], expected)


def test_context_rows_are_not_returned_and_v1_ignores_them(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, valid = split_frames
    x = valid.iloc[10:15]
    v1 = build_pipeline("v1", features_cfg).fit(train)
    with_ctx = transform_with_context(v1, x, context=train)
    assert len(with_ctx) == len(x)
    assert with_ctx.index.equals(x.index)
    pd.testing.assert_frame_equal(with_ctx, v1.transform(x))
    v2 = build_pipeline("v2", features_cfg).fit(train)
    assert len(transform_with_context(v2, x, context=train)) == len(x)


def test_history_context_actually_changes_v2_features(
    features_cfg: dict, split_frames: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, valid = split_frames
    x = valid.iloc[[0]]
    v2 = build_pipeline("v2", features_cfg).fit(train)
    alone = v2.transform(x)
    with_history = transform_with_context(v2, x, context=train)
    assert alone["is_first_card_txn"].iloc[0] == 1
    assert with_history["is_first_card_txn"].iloc[0] == 0
