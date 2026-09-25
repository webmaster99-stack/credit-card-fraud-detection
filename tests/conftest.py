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
