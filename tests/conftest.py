import pandas as pd
import pytest

from fraud.params import load_params


def make_raw(n: int = 6) -> pd.DataFrame:
    """A small frame shaped like the raw Sparkov CSV (deliberately out of time order)."""
    ts = pd.date_range("2019-01-01 00:00:00", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    return pd.DataFrame(
        {
            "Unnamed: 0": range(n),
            "trans_date_trans_time": ts[::-1],
            "cc_num": [4111111111111111, 4222222222222222] * (n // 2),
            "merchant": ["fraud_Acme Inc"] * n,
            "category": ["grocery_pos"] * n,
            "amt": [10.0 + i for i in range(n)],
            "first": ["Ann"] * n,
            "last": ["Lee"] * n,
            "gender": ["F"] * n,
            "street": ["1 Main St"] * n,
            "city": ["Springfield"] * n,
            "state": ["IL"] * n,
            "zip": [62701] * n,
            "lat": [39.8] * n,
            "long": [-89.6] * n,
            "city_pop": [1000] * n,
            "job": ["Engineer"] * n,
            "dob": ["1980-05-01"] * n,
            "trans_num": [f"t{i}" for i in range(n)],
            "unix_time": [1_325_376_000 + i for i in range(n)],
            "merch_lat": [39.9] * n,
            "merch_long": [-89.7] * n,
            "is_fraud": [0, 0, 1, 0, 0, 0][:n],
        }
    )


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def clean_cfg() -> dict:
    return load_params()["clean"]


def make_clean_frame(n: int = 300, cards: int = 5, seed: int | None = None) -> pd.DataFrame:
    """A cleaned-table-shaped frame with several cards, in time order, seeded from params."""
    import numpy as np

    rng = np.random.default_rng(load_params()["seed"] if seed is None else seed)
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 40 * 24 * 3600, n)), unit="s"
    )
    lat, lon = rng.uniform(30, 45, n), rng.uniform(-120, -75, n)
    return pd.DataFrame(
        {
            "trans_ts": ts,
            "card_id": rng.choice([f"{i:016x}" for i in range(cards)], n),
            "merchant": "Acme",
            "category": rng.choice(["grocery_pos", "shopping_net", "travel"], n),
            "amt": rng.uniform(1, 500, n).round(2),
            "gender": rng.choice(["F", "M"], n),
            "city": "Springfield",
            "state": rng.choice(["IL", "TX", "NY"], n),
            "zip": 62701,
            "lat": lat,
            "long": lon,
            "city_pop": rng.integers(100, 1_000_000, n),
            "job": "Engineer",
            "dob": pd.Timestamp("1980-05-01"),
            "merch_lat": lat + rng.uniform(-1, 1, n),
            "merch_long": lon + rng.uniform(-1, 1, n),
            "is_fraud": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    return make_clean_frame()


@pytest.fixture
def features_cfg() -> dict:
    return load_params()["features"]


DEMO_MODEL_CFG = {
    "step": "xgboost",
    "feature_set": "v1",
    "hyperparams": {"n_estimators": 20, "max_depth": 3},
    "calibration": "sigmoid",
    "min_precision": 0.5,
}


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory):
    """A tiny but complete serving bundle, fitted on synthetic data (no DVC or MLflow needed)."""
    from fraud.models.train import fit_champion_candidate
    from fraud.serving import write_bundle
    from fraud.serving.export import EXAMPLE_COLUMNS, build_cities

    params = load_params()
    df = make_clean_frame(n=600, cards=12)
    train, valid = df.iloc[:400].reset_index(drop=True), df.iloc[400:].reset_index(drop=True)
    pipeline, metrics = fit_champion_candidate(
        train, valid, DEMO_MODEL_CFG, params["features"], seed=params["seed"]
    )
    metadata = {
        "model_name": "fraud-classifier",
        "model_version": "test-1",
        "alias": "demo",
        "pipeline_version": "features-test",
        "feature_set": "v1",
        "step": "xgboost",
        "calibration": "sigmoid",
        "threshold": metrics["threshold"],
        "min_precision": 0.5,
        "dataset_name": "sparkov",
        "dataset_version": "v1",
        "split_spec": "test split",
        "git_commit": "abc1234",
        "validation": {"precision": metrics["precision"], "recall": metrics["recall"]},
        "reference_champion_test": {
            "step": "lightgbm",
            "feature_set": "v2",
            "precision": 0.446,
            "recall": 0.986,
            "pr_auc": 0.973,
            "confidence_intervals": {
                "precision": [0.423, 0.468],
                "recall": [0.978, 0.993],
                "pr_auc": [0.965, 0.980],
            },
        },
    }
    examples = valid.groupby("is_fraud").head(3)[EXAMPLE_COLUMNS].reset_index(drop=True)
    return write_bundle(
        tmp_path_factory.mktemp("bundle"),
        pipeline,
        metadata,
        model_card="# test card\n",
        cities=build_cities(df),
        examples=examples,
    )
