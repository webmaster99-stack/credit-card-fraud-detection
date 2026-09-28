"""Input contract for scoring: the raw transaction columns the stateless (v1) pipeline reads.

Serving accepts the same column names as the cleaned table (`fraud.data.schemas`), minus everything
the v1 pipeline does not use. Extra columns (e.g. `is_fraud`, `card_id`) are allowed and ignored.
Problems are reported as one readable message per column, never a raw pandera traceback.
"""

import re
from collections.abc import Sequence
from typing import Any

import pandas as pd
import pandera.pandas as pa

from fraud.data.schemas import CATEGORIES

REQUIRED_COLUMNS = [
    "trans_ts",
    "amt",
    "category",
    "gender",
    "state",
    "city_pop",
    "dob",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
]

# Rows listed per problem before the message says "and N more".
MAX_ROWS_SHOWN = 5

# Two invented rows for the downloadable template CSV and the CI smoke test: one ordinary evening
# purchase and one large late-night online purchase far from the cardholder's home.
TEMPLATE_ROWS: list[dict[str, Any]] = [
    {
        "trans_ts": "2020-11-05 18:42:10",
        "amt": 54.20,
        "category": "grocery_pos",
        "gender": "F",
        "state": "IL",
        "city_pop": 116_250,
        "dob": "1985-03-02",
        "lat": 39.80,
        "long": -89.64,
        "merch_lat": 39.85,
        "merch_long": -89.70,
    },
    {
        "trans_ts": "2020-11-06 02:17:45",
        "amt": 1043.75,
        "category": "shopping_net",
        "gender": "M",
        "state": "TX",
        "city_pop": 2_296_224,
        "dob": "1972-09-14",
        "lat": 29.76,
        "long": -95.37,
        "merch_lat": 40.71,
        "merch_long": -74.01,
    },
]


class InputValidationError(ValueError):
    """The submitted transactions violate the input contract; `problems` are human-readable."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(f"- {p}" for p in self.problems))


def input_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "trans_ts": pa.Column(pa.DateTime, coerce=True, nullable=False),
            "amt": pa.Column(float, pa.Check.gt(0), coerce=True, nullable=False),
            "category": pa.Column(str, pa.Check.isin(CATEGORIES), nullable=False),
            "gender": pa.Column(str, pa.Check.isin(["F", "M"]), nullable=False),
            "state": pa.Column(str, pa.Check.str_length(2, 2), nullable=False),
            "city_pop": pa.Column(int, pa.Check.ge(0), coerce=True, nullable=False),
            "dob": pa.Column(pa.DateTime, coerce=True, nullable=False),
            "lat": pa.Column(float, pa.Check.in_range(-90, 90), coerce=True, nullable=False),
            "long": pa.Column(float, pa.Check.in_range(-180, 180), coerce=True, nullable=False),
            "merch_lat": pa.Column(float, pa.Check.in_range(-90, 90), coerce=True, nullable=False),
            "merch_long": pa.Column(
                float, pa.Check.in_range(-180, 180), coerce=True, nullable=False
            ),
        },
        strict=False,
    )


def template_frame() -> pd.DataFrame:
    return pd.DataFrame(TEMPLATE_ROWS, columns=REQUIRED_COLUMNS)


def _rows_text(rows: list[int]) -> str:
    shown = ", ".join(str(r) for r in rows[:MAX_ROWS_SHOWN])
    extra = len(rows) - MAX_ROWS_SHOWN
    return f"rows {shown}" + (f" and {extra} more" if extra > 0 else "")


ALLOWED_VALUES = {"category": CATEGORIES, "gender": ["F", "M"]}

# pandera check name -> the requirement in plain words
_REQUIREMENTS = [
    (re.compile(r"greater_than\((.+)\)"), "must be greater than {0}"),
    (re.compile(r"greater_than_or_equal_to\((.+)\)"), "must be {0} or more"),
    (re.compile(r"in_range\((.+), (.+)\)"), "must be between {0} and {1}"),
    (re.compile(r"str_length\((\d+), \1\)"), "must be exactly {0} characters"),
]


def _requirement(check: str) -> str:
    for pattern, template in _REQUIREMENTS:
        match = pattern.fullmatch(check)
        if match:
            return template.format(*match.groups())
    return f"failed the check '{check}'"


def _format_failures(failures: pd.DataFrame) -> list[str]:
    # A column that failed to coerce also fails its plain dtype check; report it once.
    coerced = set(
        failures.loc[failures["check"].astype(str).str.startswith("coerce_dtype"), "column"]
    )
    problems: list[str] = []
    for (column, check), group in failures.groupby(["column", "check"], dropna=False, sort=False):
        check = str(check)
        if column in coerced and check.startswith("dtype("):
            continue
        rows = sorted({int(i) + 1 for i in group["index"].dropna()})
        where = f" ({_rows_text(rows)})" if rows else ""
        examples = ", ".join(repr(v) for v in group["failure_case"].head(3))
        if check == "column_in_dataframe":
            problems.append(f"Missing required column '{group['failure_case'].iloc[0]}'.")
        elif check.startswith("coerce_dtype"):
            problems.append(f"Column '{column}' has values that are not valid{where}: {examples}.")
        elif check == "not_nullable":
            problems.append(f"Column '{column}' has empty values{where}.")
        elif check.startswith("isin(") and column in ALLOWED_VALUES:
            allowed = ", ".join(ALLOWED_VALUES[column])
            problems.append(
                f"Column '{column}' has unexpected values{where}: {examples}. Allowed: {allowed}."
            )
        else:
            problems.append(f"Column '{column}' {_requirement(check)}{where}: {examples}.")
    return problems


def validate_transactions(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """A coerced copy of ``df`` with the required columns typed, or InputValidationError.

    Row numbers in messages are 1-based positions among the data rows (the header is not counted).
    """
    if len(df) == 0:
        raise InputValidationError(["No transactions to score: the input has no rows."])
    if max_rows is not None and len(df) > max_rows:
        raise InputValidationError(
            [f"Too many rows: {len(df):,} submitted, the limit is {max_rows:,} per request."]
        )
    positional = df.reset_index(drop=True)
    try:
        validated: pd.DataFrame = input_schema().validate(positional, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise InputValidationError(_format_failures(exc.failure_cases)) from None
    # A cross-column rule, checked only once every column has a usable type.
    born_late = (validated["dob"] >= validated["trans_ts"]).to_numpy().nonzero()[0] + 1
    if len(born_late):
        raise InputValidationError(
            [
                "Date of birth must be earlier than the transaction time "
                f"({_rows_text(born_late.tolist())})."
            ]
        )
    return validated.set_axis(df.index)


def read_transactions_csv(source: Any, max_rows: int) -> pd.DataFrame:
    """Read an uploaded CSV, refusing more than ``max_rows`` rows before loading them all."""
    try:
        frame = pd.read_csv(source, nrows=max_rows + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise InputValidationError([f"Could not read the file as a CSV: {exc}"]) from None
    return frame
