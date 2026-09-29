"""Pydantic request/response models.

`TransactionIn`'s fields mirror `fraud.serving.schema.REQUIRED_COLUMNS` plus `card_id` (the online
history key); `tests/test_api_schemas.py` keeps the two in lockstep so the two schemas can never
drift apart. Pydantic only checks types and presence here — the readable, detailed checks (amount
must be positive, category must be a known one, ...) stay in the one place that already has them,
`fraud.serving.schema.validate_transactions`, and its `InputValidationError` becomes a 422.
"""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    trans_ts: datetime
    amt: float
    category: str
    gender: str
    state: str
    city_pop: int
    dob: date
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    # Identifies the card for the online history store; not part of the model's input schema.
    card_id: str = Field(min_length=1)


class ReasonOut(BaseModel):
    feature: str
    label: str
    value: str
    contribution: float
    direction: str
    text: str


class PredictOut(BaseModel):
    request_id: UUID
    fraud_probability: float
    flagged: bool
    threshold: float
    reasons: list[ReasonOut]
    model_name: str
    model_version: str
    pipeline_version: str


class BatchRowOut(BaseModel):
    row: int
    request_id: UUID
    fraud_probability: float
    flagged: bool
    top_reason: str | None = None


class BatchOut(BaseModel):
    n_rows: int
    n_flagged: int
    threshold: float
    model_name: str
    model_version: str
    pipeline_version: str
    results: list[BatchRowOut]


class FeedbackIn(BaseModel):
    request_id: UUID
    is_fraud: bool


class FeedbackOut(BaseModel):
    request_id: UUID
    recorded: bool


class ModelInfoOut(BaseModel):
    model_name: str
    model_version: str
    alias: str
    pipeline_version: str
    feature_set: str
    step: str
    calibration: str
    threshold: float
    min_precision: float
    validation: dict[str, float]
    dataset_name: str
    dataset_version: str
    split_spec: str
    git_commit: str


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    database_ok: bool


class ProblemOut(BaseModel):
    """A readable validation failure, matching `InputValidationError.problems`."""

    detail: list[str]
