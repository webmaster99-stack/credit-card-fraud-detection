import numpy as np
import pandas as pd
import pytest
import yaml

from fraud.params import REPO_ROOT, load_params
from fraud.ulb import PIPELINE_VERSION
from fraud.ulb.data import (
    PCA_COLUMNS,
    SCHEMA,
    TARGET,
    assign_split,
    split_frame,
    split_spec_text,
)
from fraud.ulb.features import build_pipeline, feature_list
from fraud.ulb.paths import PROCESSED_DIR, RAW_DIR
from fraud.ulb.train import best_step, fit_step, lineage_overrides

SPLIT = {"train_end_hour": 28, "valid_end_hour": 40}


def make_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=PCA_COLUMNS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_792, n)))
    df["Amount"] = rng.exponential(50, n)
    df[TARGET] = (rng.random(n) < 0.1).astype(int)
    # Make fraud learnable so the ladder step has something to find.
    df.loc[df[TARGET] == 1, "V1"] += 4.0
    return df


def test_schema_accepts_ulb_shape() -> None:
    SCHEMA.validate(make_frame())


def test_schema_rejects_extra_column() -> None:
    with pytest.raises(Exception, match="column"):
        SCHEMA.validate(make_frame().assign(extra=1))


def test_split_is_time_ordered_and_covers_every_row() -> None:
    df = make_frame()
    parts = split_frame(df, SPLIT)
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"]["Time"].max() < 28 * 3600 <= parts["valid"]["Time"].min()
    assert parts["valid"]["Time"].max() < 40 * 3600 <= parts["test"]["Time"].min()


def test_split_rejects_unordered_hours() -> None:
    with pytest.raises(ValueError, match="train_end_hour"):
        assign_split(pd.Series([0.0]), {"train_end_hour": 40, "valid_end_hour": 39})


def test_split_boundary_goes_to_the_later_window() -> None:
    labels = assign_split(pd.Series([28 * 3600.0, 40 * 3600.0]), SPLIT)
    assert list(labels) == ["valid", "test"]


def test_feature_pipeline_output_columns() -> None:
    df = make_frame()
    pipe = build_pipeline(load_params()["ulb"]["features"])
    out = pipe.fit_transform(df.drop(columns=TARGET))
    assert list(out.columns) == ["log_amt", "hour_sin", "hour_cos", *PCA_COLUMNS]
    assert out["log_amt"].mean() == pytest.approx(0.0, abs=1e-9)
    spec = feature_list(pipe, out)
    assert spec["pipeline_version"] == PIPELINE_VERSION == "features-ulb-1.0.0"
    assert spec["n_features"] == 31


def test_time_of_day_is_cyclic() -> None:
    pipe = build_pipeline(load_params()["ulb"]["features"])
    df = make_frame().drop(columns=TARGET)
    pipe.fit(df)
    day = df.iloc[:1].assign(Time=100.0)
    next_day = df.iloc[:1].assign(Time=100.0 + 86_400)
    a, b = pipe.transform(day), pipe.transform(next_day)
    assert a[["hour_sin", "hour_cos"]].to_numpy() == pytest.approx(
        b[["hour_sin", "hour_cos"]].to_numpy()
    )


def test_fit_step_finds_the_planted_signal() -> None:
    df = make_frame(3000)
    train, valid = df.iloc[:2000], df.iloc[2000:].reset_index(drop=True)
    ulb = load_params()["ulb"]
    pipeline, metrics = fit_step("logreg", ulb["steps"]["logreg"], train, valid, ulb, seed=42)
    assert metrics["met_budget"]
    assert metrics["recall"] > 0.8
    assert pipeline.named_steps["model"].threshold_ == metrics["threshold"]


def test_best_step_prefers_budget_then_recall() -> None:
    results = [
        {"step": "a", "met_budget": False, "recall": 0.99, "pr_auc": 0.9},
        {"step": "b", "met_budget": True, "recall": 0.70, "pr_auc": 0.8},
        {"step": "c", "met_budget": True, "recall": 0.80, "pr_auc": 0.7},
    ]
    assert best_step(results)["step"] == "c"
    assert best_step(results[:1])["step"] == "a"


def test_lineage_overrides_name_the_ulb_dataset() -> None:
    tags = lineage_overrides(load_params())
    assert tags["dataset_name"] == "ulb"
    assert tags["pipeline_version"] == "features-ulb-1.0.0"
    assert tags["split_spec"] == split_spec_text(load_params()["ulb"]["split"])


def _stages() -> dict:
    return yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]


def _rel(path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_dvc_outs_match_ulb_path_constants() -> None:
    stages = _stages()
    assert stages["ulb_ingest"]["outs"] == [_rel(RAW_DIR)]
    assert stages["ulb_split"]["outs"] == [_rel(PROCESSED_DIR)]
    assert _rel(RAW_DIR) in stages["ulb_split"]["deps"]


def test_ulb_train_never_depends_on_the_test_split() -> None:
    deps = _stages()["ulb_train"]["deps"]
    assert f"{_rel(PROCESSED_DIR)}/train.parquet" in deps
    assert f"{_rel(PROCESSED_DIR)}/valid.parquet" in deps
    assert not any("test" in d and "parquet" in d for d in deps)
    assert _rel(PROCESSED_DIR) not in deps


def test_ulb_and_sparkov_data_do_not_nest() -> None:
    assert not _rel(PROCESSED_DIR).startswith("data/processed/")
