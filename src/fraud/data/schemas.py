"""Pandera schema for the cleaned transactions table (the contract every later stage relies on)."""

import pandera.pandas as pa

CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]


def clean_schema(card_id_length: int) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "trans_ts": pa.Column(pa.DateTime, nullable=False),
            "card_id": pa.Column(
                str, pa.Check.str_matches(rf"^[0-9a-f]{{{card_id_length}}}$"), nullable=False
            ),
            "merchant": pa.Column(str, nullable=False),
            "category": pa.Column(str, pa.Check.isin(CATEGORIES), nullable=False),
            "amt": pa.Column(float, pa.Check.gt(0), nullable=False),
            "gender": pa.Column(str, pa.Check.isin(["F", "M"]), nullable=False),
            "city": pa.Column(str, nullable=False),
            "state": pa.Column(str, pa.Check.str_length(2, 2), nullable=False),
            "zip": pa.Column(int, nullable=False),
            "lat": pa.Column(float, pa.Check.in_range(-90, 90), nullable=False),
            "long": pa.Column(float, pa.Check.in_range(-180, 180), nullable=False),
            "city_pop": pa.Column(int, pa.Check.ge(0), nullable=False),
            "job": pa.Column(str, nullable=False),
            "dob": pa.Column(pa.DateTime, nullable=False),
            "merch_lat": pa.Column(float, pa.Check.in_range(-90, 90), nullable=False),
            "merch_long": pa.Column(float, pa.Check.in_range(-180, 180), nullable=False),
            "is_fraud": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        },
        strict=True,
        ordered=False,
    )
