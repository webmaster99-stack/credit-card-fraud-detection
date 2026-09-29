"""The FastAPI service: score transactions with the champion model, log every prediction.

Run locally:  uv run uvicorn api.main:app --reload   (needs DATABASE_URL and API_KEY, e.g. in .env)
"""

import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import ConnectionPool
from pydantic import ValidationError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.config import get_settings, serving_defaults
from api.db import (
    close_pool,
    fetch_card_history,
    init_schema,
    insert_prediction,
    insert_predictions_batch,
    open_pool,
    ping,
    record_feedback,
)
from api.deps import get_model, get_pool, limiter, require_api_key
from api.logging_config import RequestLoggingMiddleware, configure_logging
from api.schemas import (
    BatchOut,
    BatchRowOut,
    FeedbackIn,
    FeedbackOut,
    HealthOut,
    ModelInfoOut,
    PredictOut,
    ReasonOut,
    TransactionIn,
)
from fraud.serving import InputValidationError, ServingModel, explain, load_model, predict
from fraud.serving.schema import read_transactions_csv, validate_transactions


def _uses_history(model: ServingModel) -> bool:
    return str(model.metadata.get("feature_set")) == "v2"


def _to_frame(txn: TransactionIn) -> pd.DataFrame:
    return pd.DataFrame([txn.model_dump()])


def _score_dict(model: ServingModel, scored: pd.DataFrame, i: int) -> dict[str, Any]:
    return {
        "fraud_probability": float(scored["fraud_probability"].iloc[i]),
        "flagged": bool(scored["flagged"].iloc[i]),
        "model_name": str(model.metadata["model_name"]),
        "model_version": model.model_version,
        "pipeline_version": model.pipeline_version,
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    app.state.model = load_model(settings.model_source, revision=settings.model_revision)
    app.state.pool = open_pool(settings.database_url)
    init_schema(app.state.pool)
    try:
        yield
    finally:
        close_pool(app.state.pool)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Fraud classifier API", version="1.0", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthOut)
    @limiter.exempt  # type: ignore[untyped-decorator]  # slowapi ships no type stubs
    def health(request: Request, response: Response) -> HealthOut:
        model_loaded = getattr(request.app.state, "model", None) is not None
        pool = getattr(request.app.state, "pool", None)
        database_ok = ping(pool) if pool is not None else False
        if not (model_loaded and database_ok):
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthOut(
            status="ok" if model_loaded and database_ok else "degraded",
            model_loaded=model_loaded,
            database_ok=database_ok,
        )

    @app.get("/v1/model", response_model=ModelInfoOut, dependencies=[Depends(require_api_key)])
    def model_info(model: Annotated[ServingModel, Depends(get_model)]) -> ModelInfoOut:
        m, v = model.metadata, model.metadata["validation"]
        return ModelInfoOut(
            model_name=m["model_name"],
            model_version=model.model_version,
            alias=m["alias"],
            pipeline_version=model.pipeline_version,
            feature_set=m["feature_set"],
            step=m["step"],
            calibration=m["calibration"],
            threshold=model.threshold,
            min_precision=m["min_precision"],
            validation={"precision": v["precision"], "recall": v["recall"]},
            dataset_name=m["dataset_name"],
            dataset_version=m["dataset_version"],
            split_spec=m["split_spec"],
            git_commit=m["git_commit"],
        )

    @app.post(
        "/v1/predict",
        response_model=PredictOut,
        dependencies=[Depends(require_api_key)],
    )
    def predict_one(
        request: Request,
        txn: TransactionIn,
        model: Annotated[ServingModel, Depends(get_model)],
        pool: Annotated[ConnectionPool, Depends(get_pool)],
    ) -> PredictOut:
        context = None
        if _uses_history(model):
            context = fetch_card_history(
                pool, txn.card_id, txn.trans_ts, max_rows=get_settings().max_history_rows
            )
        frame = _to_frame(txn)
        try:
            scored = predict(model, frame, context=context)
            top_k = int(serving_defaults()["top_k_reasons"])
            reasons = explain(model, frame, context=context, top_k=top_k)[0]
        except InputValidationError as err:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, err.problems) from None

        row = txn.model_dump()
        scores = _score_dict(model, scored, 0)
        request_id = insert_prediction(pool, row, scores, source="single")
        return PredictOut(
            request_id=request_id,
            fraud_probability=float(scores["fraud_probability"]),
            flagged=bool(scores["flagged"]),
            threshold=model.threshold,
            reasons=[
                ReasonOut(
                    feature=r.feature,
                    label=r.label,
                    value=r.value,
                    contribution=r.contribution,
                    direction=r.direction,
                    text=r.text,
                )
                for r in reasons
            ],
            model_name=str(scores["model_name"]),
            model_version=str(scores["model_version"]),
            pipeline_version=str(scores["pipeline_version"]),
        )

    @app.post(
        "/v1/predict/batch",
        response_model=BatchOut,
        dependencies=[Depends(require_api_key)],
    )
    async def predict_batch(
        request: Request,
        model: Annotated[ServingModel, Depends(get_model)],
        pool: Annotated[ConnectionPool, Depends(get_pool)],
    ) -> BatchOut:
        defaults = serving_defaults()
        max_rows = int(defaults["max_batch_rows"])
        content_type = request.headers.get("content-type", "")

        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, ["No file uploaded under 'file'."]
                )
            raw = await upload.read()
            try:
                frame = read_transactions_csv(io.BytesIO(raw), max_rows)
            except InputValidationError as err:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, err.problems) from None
        else:
            body = await request.json()
            rows = body.get("transactions", body) if isinstance(body, dict) else body
            if not isinstance(rows, list):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    ["Body must be a JSON list of transactions, or {'transactions': [...]}."],
                )
            if len(rows) > max_rows:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    [f"Too many rows: {len(rows):,} submitted, the limit is {max_rows:,}."],
                )
            try:
                txns = [TransactionIn.model_validate(r) for r in rows]
            except ValidationError as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, [str(e) for e in exc.errors()]
                ) from None
            frame = pd.DataFrame([t.model_dump() for t in txns])

        if "card_id" not in frame.columns:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, ["Missing required column 'card_id'."]
            )
        frame["card_id"] = frame["card_id"].astype(str)

        # Batch mode never touches Postgres for history: card history comes only from other rows
        # in this file, so a batch's results never depend on when it happens to be uploaded
        # (docs/plan.md Phase 2). The heavy scoring below blocks the event loop; fine for a
        # single-instance free-tier deployment, matching the Gradio demo's synchronous model.
        try:
            frame = validate_transactions(frame, max_rows)
            scored = predict(model, frame)
        except InputValidationError as err:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, err.problems) from None

        flagged = scored[scored["flagged"]].sort_values("fraud_probability", ascending=False)
        to_explain = flagged.head(int(defaults["batch_explain_rows"])).index
        reasons_by_row: dict[int, str] = {}
        if len(to_explain):
            reasons = explain(model, frame.loc[to_explain], top_k=1)
            for idx, row_reasons in zip(to_explain, reasons, strict=True):
                reasons_by_row[idx] = row_reasons[0].text

        rows_out = frame.to_dict("records")
        scores_out = [_score_dict(model, scored, i) for i in range(len(frame))]
        request_ids = insert_predictions_batch(pool, rows_out, scores_out, source="batch")  # type: ignore[arg-type]

        results = [
            BatchRowOut(
                row=i + 1,
                request_id=request_ids[i],
                fraud_probability=float(scores_out[i]["fraud_probability"]),
                flagged=bool(scores_out[i]["flagged"]),
                top_reason=reasons_by_row.get(frame.index[i]),
            )
            for i in range(len(frame))
        ]
        return BatchOut(
            n_rows=len(frame),
            n_flagged=int(scored["flagged"].sum()),
            threshold=model.threshold,
            model_name=str(model.metadata["model_name"]),
            model_version=model.model_version,
            pipeline_version=model.pipeline_version,
            results=results,
        )

    @app.post("/v1/feedback", response_model=FeedbackOut, dependencies=[Depends(require_api_key)])
    def feedback(
        body: FeedbackIn, pool: Annotated[ConnectionPool, Depends(get_pool)]
    ) -> FeedbackOut:
        recorded = record_feedback(pool, body.request_id, body.is_fraud)
        if not recorded:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"No prediction found for request_id {body.request_id}."
            )
        return FeedbackOut(request_id=body.request_id, recorded=True)

    return app


app = create_app()
