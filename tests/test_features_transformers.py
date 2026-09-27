import numpy as np
import pandas as pd
import pytest

from fraud.features.transformers import (
    CustomerFeatures,
    GeographyFeatures,
    TransactionFeatures,
    haversine_km,
)


def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_transaction_features_values() -> None:
    X = _frame(
        trans_ts=pd.to_datetime(
            ["2020-01-06 22:00:00", "2020-01-05 03:59:00", "2020-01-05 12:00:00"]
        ),
        amt=[99.0, 0.5, 10.0],
        category=["travel", "home", "home"],
    )
    out = TransactionFeatures(22, 4).transform(X)
    assert list(out.columns) == list(TransactionFeatures().get_feature_names_out())
    np.testing.assert_allclose(out["log_amt"], np.log1p([99.0, 0.5, 10.0]))
    assert out["hour"].tolist() == [22, 3, 12]
    assert out["weekday"].tolist() == [0, 6, 6]  # Monday, Sunday, Sunday
    assert out["is_night"].tolist() == [1, 1, 0]  # 22:00 and 03:59 are night, noon is not


def test_night_window_boundaries_come_from_arguments() -> None:
    X = _frame(
        trans_ts=pd.to_datetime(["2020-01-01 21:00", "2020-01-01 04:00"]),
        amt=[1.0, 1.0],
        category=["home", "home"],
    )
    assert TransactionFeatures(22, 4).transform(X)["is_night"].tolist() == [0, 0]
    assert TransactionFeatures(21, 5).transform(X)["is_night"].tolist() == [1, 1]


def test_customer_features_age_and_logs() -> None:
    X = _frame(
        trans_ts=pd.to_datetime(["2020-05-01"]),
        dob=pd.to_datetime(["1980-05-01"]),
        gender=["F"],
        city_pop=[999],
        state=["IL"],
    )
    out = CustomerFeatures().transform(X)
    assert out["age"].iloc[0] == pytest.approx(40.0, abs=0.01)
    assert out["log_city_pop"].iloc[0] == pytest.approx(np.log1p(999))
    assert out[["gender", "state"]].iloc[0].tolist() == ["F", "IL"]


def test_haversine_known_distances() -> None:
    s = lambda v: pd.Series([v])  # noqa: E731
    assert haversine_km(s(0.0), s(0.0), s(0.0), s(0.0))[0] == 0.0
    # One degree of longitude on the equator is about 111.19 km.
    assert haversine_km(s(0.0), s(0.0), s(0.0), s(1.0))[0] == pytest.approx(111.19, abs=0.05)
    # New York to Los Angeles is about 3936 km.
    d = haversine_km(s(40.7128), s(-74.0060), s(34.0522), s(-118.2437))[0]
    assert d == pytest.approx(3936, rel=0.01)


def test_geography_features_is_symmetric_and_non_negative(clean_frame: pd.DataFrame) -> None:
    out = GeographyFeatures().transform(clean_frame)
    assert (out["distance_km"] >= 0).all()
    swapped = clean_frame.rename(columns={"lat": "merch_lat", "merch_lat": "lat"}).rename(
        columns={"long": "merch_long", "merch_long": "long"}
    )
    np.testing.assert_allclose(
        GeographyFeatures().transform(swapped)["distance_km"], out["distance_km"]
    )


@pytest.mark.parametrize("cls", [TransactionFeatures, CustomerFeatures, GeographyFeatures])
def test_missing_columns_raise_a_clear_error(cls, clean_frame: pd.DataFrame) -> None:
    broken = clean_frame.drop(columns=list(cls.required[:1]))
    with pytest.raises(ValueError, match="missing from the input"):
        cls().transform(broken)


@pytest.mark.parametrize("cls", [TransactionFeatures, CustomerFeatures, GeographyFeatures])
def test_transformers_keep_index_and_declared_columns(cls, clean_frame: pd.DataFrame) -> None:
    X = clean_frame.iloc[::-1]
    out = cls().transform(X)
    assert out.index.equals(X.index)
    assert list(out.columns) == list(cls().get_feature_names_out())
