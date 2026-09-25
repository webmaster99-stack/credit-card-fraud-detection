"""Clean the raw Sparkov CSVs: parse types, drop identifiers, hash card numbers, validate.

Usage: uv run python -m fraud.data.clean
"""

import hashlib
from typing import Any

import pandas as pd

from fraud.data.paths import INTERIM_PATH, RAW_DIR
from fraud.data.schemas import clean_schema
from fraud.params import load_params

RAW_TIMESTAMP = "trans_date_trans_time"


def card_ids(cc_num: pd.Series, salt: str, length: int) -> pd.Series:
    """Stable, salted hash of the card number. Hashes each distinct number once."""
    mapping = {
        n: hashlib.sha256(f"{salt}:{n}".encode()).hexdigest()[:length] for n in cc_num.unique()
    }
    return cc_num.map(mapping)


def clean(raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    df = raw.copy()
    df["trans_ts"] = pd.to_datetime(df[RAW_TIMESTAMP])
    df["dob"] = pd.to_datetime(df["dob"])
    df["card_id"] = card_ids(df["cc_num"], cfg["card_id_salt"], cfg["card_id_length"])
    df["merchant"] = df["merchant"].str.removeprefix(cfg["merchant_prefix"])
    df = df.drop(columns=[RAW_TIMESTAMP, "cc_num", *cfg["identifier_columns"]])
    df = df.drop(columns=cfg["redundant_columns"])
    # Stable sort keeps the order deterministic when timestamps tie.
    df = df.sort_values("trans_ts", kind="stable").reset_index(drop=True)
    return clean_schema(cfg["card_id_length"]).validate(df)


def main() -> None:
    params = load_params()
    raw = pd.concat(
        [pd.read_csv(RAW_DIR / name) for name in params["ingest"]["expected_files"]],
        ignore_index=True,
    )
    df = clean(raw, params["clean"])
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERIM_PATH, index=False)
    print(f"Wrote {len(df):,} rows, {df['card_id'].nunique()} cards -> {INTERIM_PATH}")


if __name__ == "__main__":
    main()
