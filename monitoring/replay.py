"""Drift replay demo: push held-out months through the live API, then again with injected shift.

    uv run python -m monitoring.replay --rows 300            # clean replay, labels posted late
    uv run python -m monitoring.replay --rows 300 --shift    # all amounts x3

Needs API_URL and API_KEY (the live service), DATABASE_URL (to store the result) and the held-out
split from `dvc pull` (`data/processed/<split>.parquet`, default `test`; scoring it here is
monitoring, not tuning). Each run posts its rows to `/v1/predict/batch`, posts the true labels of
the flagged-or-fraud rows to `/v1/feedback` after `--label-delay` seconds (chargebacks arrive
late), and stores a `replay` summary that the web monitoring page shows: per-feature PSI against
the validation reference, plus the model's flag rate and recall on the replayed rows.
Replayed rows do land in the prediction log (source `batch`); use a scratch database if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import httpx
import pandas as pd
from api.db import close_pool, init_schema, open_pool, save_monitoring_report
from dotenv import load_dotenv

from fraud.data.paths import PROCESSED_DIR
from fraud.monitoring.drift import feature_drift, flag_rate_status, performance
from fraud.params import REPO_ROOT, load_params
from fraud.serving.schema import REQUIRED_COLUMNS

REFERENCE_PATH = REPO_ROOT / "data" / "monitoring" / "reference.parquet"
STATS_PATH = REPO_ROOT / "reports" / "monitoring_reference.json"
# The API's default limit is 60 requests/minute; feedback posts stay under it.
FEEDBACK_INTERVAL_S = 1.1
BATCH_CHUNK = 200


def inject_shift(rows: pd.DataFrame, category: str | None, factor: float) -> pd.DataFrame:
    """Inflate amounts by ``factor``: every row, or only those in ``category`` if given.

    A single-category shift is subtle: a small, already-expensive category moves the overall
    amount distribution very little, so it may stay under the PSI alert threshold.
    """
    shifted = rows.copy()
    mask = (
        pd.Series(True, index=shifted.index)
        if category is None
        else shifted["category"] == category
    )
    shifted.loc[mask, "amt"] = shifted.loc[mask, "amt"] * factor
    return shifted


def _payload(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out = rows[[*REQUIRED_COLUMNS, "card_id"]].copy()
    out["trans_ts"] = out["trans_ts"].astype(str)
    out["dob"] = pd.to_datetime(out["dob"]).dt.strftime("%Y-%m-%d")
    records: list[dict[str, Any]] = json.loads(out.to_json(orient="records"))
    return records


def score_rows(client: httpx.Client, rows: pd.DataFrame) -> pd.DataFrame:
    """POST rows to the live batch endpoint; returns them with `request_id` and `flagged`."""
    scored = []
    for start in range(0, len(rows), BATCH_CHUNK):
        chunk = rows.iloc[start : start + BATCH_CHUNK]
        resp = client.post("/v1/predict/batch", json=_payload(chunk))
        resp.raise_for_status()
        results = resp.json()["results"]
        part = chunk.copy()
        part["request_id"] = [r["request_id"] for r in results]
        part["flagged"] = [r["flagged"] for r in results]
        scored.append(part)
    return pd.concat(scored)


def post_labels(client: httpx.Client, scored: pd.DataFrame, delay: float) -> None:
    """Send true labels after a delay, only for rows worth a chargeback: flagged or truly fraud."""
    time.sleep(delay)
    for _, row in scored[scored["flagged"] | scored["is_fraud"]].iterrows():
        client.post(
            "/v1/feedback",
            json={"request_id": row["request_id"], "is_fraud": bool(row["is_fraud"])},
        ).raise_for_status()
        time.sleep(FEEDBACK_INTERVAL_S)


def sample_rows(split: pd.DataFrame, n: int, fraud_share: float, seed: int) -> pd.DataFrame:
    """A time-ordered sample with fraud oversampled to ``fraud_share`` so recall is measurable."""
    n_fraud = min(int(n * fraud_share), int(split["is_fraud"].sum()))
    fraud = split[split["is_fraud"]].sample(n=n_fraud, random_state=seed)
    normal = split[~split["is_fraud"]].sample(n=n - n_fraud, random_state=seed)
    return pd.concat([fraud, normal]).sort_values("trans_ts").reset_index(drop=True)


def summarize(
    scored: pd.DataFrame, reference: pd.DataFrame, stats: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    features = feature_drift(reference, scored, cfg)
    labelled = scored.assign(true_label=scored["is_fraud"])
    return {
        "n_rows": len(scored),
        "features": features,
        "n_drifted": sum(f["drifted"] for f in features),
        "flag": flag_rate_status(
            float(scored["flagged"].mean()), stats["validation_flag_rate"], cfg
        ),
        "performance": performance(labelled, float(stats["test_recall"]), cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fraud-share", type=float, default=0.1)
    parser.add_argument("--label-delay", type=float, default=30.0, help="seconds before labels")
    parser.add_argument("--shift", action="store_true", help="inject the covariate shift")
    parser.add_argument("--shift-category", default=None, help="shift only this category")
    parser.add_argument("--shift-factor", type=float, default=3.0)
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    params = load_params()
    cfg = params["monitoring"]
    split = pd.read_parquet(PROCESSED_DIR / f"{args.split}.parquet")
    split["is_fraud"] = split["is_fraud"].astype(bool)
    rows = sample_rows(split, args.rows, args.fraud_share, int(params["seed"]))
    if args.shift:
        rows = inject_shift(rows, args.shift_category, args.shift_factor)

    with httpx.Client(
        base_url=os.environ["API_URL"], headers={"X-API-Key": os.environ["API_KEY"]}, timeout=120
    ) as client:
        scored = score_rows(client, rows)
        post_labels(client, scored, args.label_delay)

    reference = pd.read_parquet(REFERENCE_PATH)
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    summary = summarize(scored, reference, stats, cfg)
    summary["shift"] = (
        {"category": args.shift_category, "amount_factor": args.shift_factor}
        if args.shift
        else None
    )
    summary["split"] = args.split

    pool = open_pool(os.environ["DATABASE_URL"])
    try:
        init_schema(pool)
        save_monitoring_report(pool, "replay", summary, keep=int(cfg["keep_reports"]))
    finally:
        close_pool(pool)
    drifted = [f["feature"] for f in summary["features"] if f["drifted"]]
    print(f"{'shifted' if args.shift else 'clean'} replay: drifted features = {drifted or 'none'}")


if __name__ == "__main__":
    main()
