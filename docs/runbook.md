# Runbook

What each monitoring alert means, how to check it, and what to do. Thresholds live in
`params.yaml` under `monitoring`. The nightly job (`.github/workflows/monitoring.yml`,
`python -m monitoring.nightly`) exits non-zero when any alert fires, so a red run of
"Nightly monitoring" on GitHub is the alert. Its latest summary is on the web app's Monitoring
page (`/monitoring`), with the Evidently report linked from it.

Fraud labels arrive late (chargebacks take weeks). Input and score checks run daily; the
performance check only has data once `/v1/feedback` labels accumulate.

## Alerts

| Alert layer | Fires when | Meaning | First check |
| --- | --- | --- | --- |
| `service` | 5xx rate on `/v1/predict*` above 1% | The API or database is failing | Render logs; `/health`; Neon status. A cold start after idle is not an error. |
| `data_quality` | 422 rate above 0.5% | Clients send invalid input (or a schema change broke a client) | Which column? Recent 422 bodies in Render logs; recent frontend or client change. |
| `data_drift` | 1+ monitored feature has PSI > 0.25 on 3 consecutive days | Inputs no longer look like the validation reference | Monitoring page drift table, then the Evidently report. Ask: real change (new merchant mix, seasonality) or a broken client (units, currency)? |
| `prediction_drift` | Flag rate outside 0.5x-2x of the validation flag rate | The model flags far more or fewer transactions than expected | Usually follows data drift. A jump up means more false alarms; a drop may be missed fraud. |
| `performance` | Recall on labelled predictions drops more than 10 points below test recall | The model misses fraud it used to catch | Needs >= 30 labels. Check label quality first (are only flagged rows being labelled? that biases recall). |

Days with fewer than `min_rows_per_day` (50) logged predictions are ignored for drift and
flag-rate checks; a quiet demo will not alert. Likewise the `service` and `data_quality` rates only
alert once the lookback window holds `min_requests_for_rates` (50) `/v1/predict*` requests: with a
handful of requests one rejected input is already far over 0.5% (2 of 7 failed the 2026-10-02 run).
Below that the summary's `service.enough_requests` is `false` and the rates are still reported. The
cost: an API failing on a near-idle day does not trip `service`; `/health` is the check for that.

## Checking a drift alert

1. Open `/monitoring`; note which features have PSI above 0.25 and since when.
2. Open the Evidently report for per-feature distributions.
3. Decide: expected change, client bug, or real shift. Fix a client bug at the source. A real shift
   moves on to retraining.

## Retraining (manual)

Retrain when drift is real and persistent, or when performance alerts with trustworthy labels.

1. Put the new data under DVC as a new dataset version (`dataset_version` bump; never overwrite).
2. `uv run dvc repro`, then compare the new model against `champion` on the same validation
   protocol (recall at precision >= 0.50). Do not tune on the test split.
3. Promote only if it wins on that protocol: move the registry alias `champion` to the new version
   and `challenger` to the old one (MLflow on DagsHub), re-export the bundle with
   `uv run python -m fraud.serving.export --alias champion --push`, then tag a release; the
   `v*` tag deploys the API through CI (`docs/infra.md`).
4. Rebuild the drift reference (`dvc repro monitoring_reference`, `dvc push`) so drift is measured
   against data the new model was validated on.

## Drift replay demo

```bash
uv run python -m monitoring.replay --rows 300            # clean: expect no drifted features
uv run python -m monitoring.replay --rows 300 --shift    # all amounts x3: amt drifts
```

Needs `API_URL`, `API_KEY`, `DATABASE_URL` in `.env` and `dvc pull` of `data/processed/test.parquet`
and `data/monitoring/`. Labels are posted `--label-delay` seconds after scoring. Replayed rows stay
in the prediction log with source `replay` (the batch endpoint's `?source=replay`), and the nightly
job excludes them, so the demo never trips or masks a real alert. Against an API older than this
change the parameter is ignored and rows are logged as `batch`; relabel them by hand.

## Secrets the nightly job needs

GitHub secrets `DATABASE_URL` (Neon), `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
(the DagsHub credentials, reused for `dvc pull`).
