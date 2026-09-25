import pandas as pd
import pandera.errors
import pytest

from fraud.data.clean import card_ids, clean


def test_identifiers_and_redundant_columns_dropped(raw: pd.DataFrame, clean_cfg: dict) -> None:
    out = clean(raw, clean_cfg)
    for col in ["first", "last", "street", "trans_num", "cc_num", "unix_time", "Unnamed: 0"]:
        assert col not in out.columns
    assert "card_id" in out.columns


def test_card_id_is_salted_stable_and_hides_number(clean_cfg: dict) -> None:
    cc = pd.Series([4111111111111111, 4111111111111111, 4222222222222222])
    a = card_ids(cc, "salt-a", 16)
    assert a.iloc[0] == a.iloc[1] != a.iloc[2]
    assert "4111111111111111" not in a.iloc[0]
    assert card_ids(cc, "salt-b", 16).iloc[0] != a.iloc[0]
    assert (a.str.len() == 16).all()


def test_sorted_by_time_and_merchant_prefix_stripped(raw: pd.DataFrame, clean_cfg: dict) -> None:
    out = clean(raw, clean_cfg)
    assert out["trans_ts"].is_monotonic_increasing
    assert (out["merchant"] == "Acme Inc").all()


def test_clean_does_not_mutate_input(raw: pd.DataFrame, clean_cfg: dict) -> None:
    before = raw.copy()
    clean(raw, clean_cfg)
    pd.testing.assert_frame_equal(raw, before)


def test_schema_rejects_bad_values(raw: pd.DataFrame, clean_cfg: dict) -> None:
    raw.loc[0, "amt"] = -5.0
    with pytest.raises(pandera.errors.SchemaError):
        clean(raw, clean_cfg)


def test_schema_rejects_unknown_category(raw: pd.DataFrame, clean_cfg: dict) -> None:
    raw.loc[0, "category"] = "mystery"
    with pytest.raises(pandera.errors.SchemaError):
        clean(raw, clean_cfg)
