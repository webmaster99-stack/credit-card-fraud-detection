# Credit Card Fraud Detection

Fraud classifier for credit card transactions using classical ML, built as a reproducible,
fully lineaged pipeline (DVC + MLflow). See `CLAUDE.md` for project rules and `docs/plan.md`
for the phase plan.

Status: all seven phases are complete (latest tag `v1.2-ulb`). Champion: LightGBM on v2 (card-history)
features, served by a FastAPI backend on Render behind a Next.js frontend on Vercel; a Gradio demo on
Hugging Face Spaces serves a stateless v1 model instead (see Demo). Nightly drift monitoring runs in
GitHub Actions, and a real-data ULB benchmark sits beside the synthetic Sparkov results. Tags:
`v0.0-setup`, `v0.1-eda`, `v0.2-features`, `v0.3-model`, `v0.4-demo`, `v1.0`, `v1.1-monitoring`
(patch `v1.1.1`), `v1.2-ulb`.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in DagsHub and Kaggle credentials
pre-commit install
uv run pytest
```

## Reproduce the pipeline

```bash
uv run dvc pull        # fetch data and models from the DagsHub remote (no Kaggle token needed)
uv run dvc repro       # rebuilds only what changed; on a fresh clone after `dvc pull` nothing reruns
```

Verified from a fresh clone: after `dvc pull` the pipeline is up to date, and a forced retrain reproduces
the metrics exactly. Model binaries and the last float digit of some report values are not bit-identical
across reruns (`docs/infra.md`). Training stages refuse to run on a dirty git tree. `.gitattributes`
pins LF line endings so `dvc.lock` hashes match on Windows, Linux and CI.

`dvc repro ingest` downloads from Kaggle and needs `KAGGLE_API_TOKEN` in `.env`. The stages:

| Stage | Output | What it does |
| --- | --- | --- |
| `ingest` | `data/raw/sparkov` | Downloads the Sparkov CSVs from Kaggle |
| `clean` | `data/interim/transactions.parquet` | Parses types, drops direct identifiers, hashes card numbers, validates with Pandera |
| `split` | `data/processed/{train,valid,test}.parquet` | Time-based split; summary in `reports/split_summary.json` |
| `features` | `reports/feature_list_{v1,v2}.json` | Fits both feature tiers on train and checks them on validation |
| `train` | `data/models/pipeline.joblib` | Fits the champion candidate, calibrates, picks the threshold (logs to MLflow) |
| `train_demo` | `data/models/demo_pipeline.joblib` | The stateless v1 model the Gradio demo serves |
| `evaluate` | `reports/evaluate_metrics.json`, `reports/pr_curve.json` | Validation metrics for the trained champion |
| `monitoring_reference` | `data/monitoring/reference.parquet` | Drift reference for the nightly job |
| `ulb_ingest`, `ulb_split`, `ulb_train` | `data/raw/ulb`, `data/processed_ulb`, `data/models/ulb_pipeline.joblib` | The ULB benchmark (Phase 7) |

Splits are by time (train Jan 2019 to Jun 2020, validation Jul to Sep 2020, test Oct to Dec 2020), never random.
The test split is used once, at the end of Phase 3.

## Feature pipeline

One scikit-learn pipeline per tier, in `src/fraud/features/` (version in `features/__init__.py`, design in
`docs/adr/0003-feature-pipeline-design.md`):

| Tier | Features | Card history |
| --- | --- | --- |
| `v1` (74 columns) | Transaction (log amount, category, hour, weekday, night flag), customer (age, gender, log city population, state), customer-merchant distance | No |
| `v2` (84 columns) | v1 plus velocity (count and spend in the last 1 h / 24 h / 7 d) and behavioural (amount vs card mean, hours since last transaction, first use of category, first transaction on card) | Yes, strictly earlier rows only |

One pipeline version (`features-1.0.0`) covers both tiers; the tier is recorded as `feature_set`.
Encoders, imputers and scalers are fitted on the training split inside the pipeline. The `features` stage fits both
tiers on train, checks them on validation (train as card history) and writes `reports/feature_list_v1.json` and
`reports/feature_list_v2.json`. It never reads the test split. Merchant, job and city are dropped for now (see the ADR).

## Modeling

Ladder protocol in `docs/plan.md` (Phase 3): tune every rung on train (Optuna), score on validation
by the primary metric (recall at precision ≥ 0.50), calibrate, pick a threshold, compare the top two
models' v1 vs v2 feature sets, then evaluate once on test. Full comparison in
`reports/model_ladder.json`; details and caveats in `docs/model_cards/v0.3-model.md`.

| Model | Features | Recall @ P≥0.50 (valid) | PR-AUC (valid) |
| --- | --- | --- | --- |
| Dummy / amount-rule floor | v1 | 0.000 | 0.004 / 0.153 |
| Logistic regression | v1 | 0.312 | 0.265 |
| + splines/interactions | v1 | 0.266 | 0.292 |
| Random forest | v1 | 0.892 | 0.842 |
| LightGBM | v1 | 0.915 | 0.874 |
| XGBoost | v1 | 0.921 | 0.888 |
| Isolation Forest (unsupervised) | v1 | 0.002 | 0.025 |
| XGBoost | v2 | 0.986 | 0.978 |
| **LightGBM (champion)** | **v2** | **0.990** | **0.978** |

Card-history (v2) features drive almost all of the gain over the best v1 model. Reproduce the
champion deterministically:

```bash
uv run dvc repro train evaluate   # fits, calibrates, picks a threshold; validation metrics only
```

**Test-set result (single evaluation, `reports/test_evaluation.json`):** recall 0.986 (CI
0.978–0.993), precision 0.446 (CI 0.423–0.468), PR-AUC 0.973 (CI 0.965–0.980). The 0.50 precision target
is a validation threshold rule and is met there (0.500). Test precision is lower because fraud prevalence
fell from 0.44% to 0.33%: recall (0.990 to 0.986) and the false-alarm rate (0.438% to 0.408%) held or
improved, so the threshold generalized and the base rate moved. Nothing was re-tuned against the spent
test split; the arithmetic is in `docs/model_cards/v0.3-model.md`.

## Demo (Phase 4)

A Gradio app in `demo/` scores one transaction or a CSV (up to 10,000 rows), explains each score with
SHAP reasons, and shows which model version answered. All model code lives in `src/fraud/serving/`
(`load_model`, `predict`, `explain`), which the Phase 5 API also uses.

The demo serves a **stateless v1 model** (registry `fraud-classifier` v3, alias `demo`: xgboost, recall 0.921
at precision 0.50 on validation), not the champion, because a form has no card history
(`docs/adr/0004-demo-serves-the-stateless-v1-model.md`). It has not been scored on the test split.

```bash
uv run dvc repro train_demo                                   # train the demo model (clean git tree)
uv run python -m fraud.models.register <run_id> demo          # alias it in the registry
uv run python -m fraud.serving.export                         # build data/bundle (add --push for the HF Hub)
FRAUD_MODEL_SOURCE=data/bundle uv run python demo/app.py      # run the app locally
uv run python demo/build_space.py                             # stage the Space folder (add --push to upload)
```

Live demo: https://huggingface.co/spaces/ilian-hadzhidimitrov/fraud-classifier-demo (free ZeroGPU slot, so it
may take a moment to wake up; the model itself runs on CPU). Deployment notes: `docs/infra.md`.
Model card: `docs/model_cards/v0.4-demo.md`.

## Full-stack app (Phase 5)

Unlike the demo, this serves the actual **champion** (lightgbm on v2, history-aware features) through
a FastAPI backend (`api/`) with a Postgres-backed online history store, and a Next.js frontend
(`web/`) that never exposes the API key to the browser. See `docs/adr/0005-api-history-store-and-champion-distribution.md`
for how the history store and champion distribution are designed, and `docs/infra.md` for the
deployed state, the fixes it took to get the image serving, and the free-tier caveats.

- Frontend (Vercel): https://fraud-classifier-web.vercel.app
- API (Render, free tier): https://credit-card-fraud-detection-5reg.onrender.com (`/health` is public; every
  `/v1/*` route needs the `X-API-Key` header). Free services sleep when idle, so the first request after a
  quiet period is slow; the frontend shows a "waking up" banner.
- Model card: `docs/model_cards/v1.0-api.md`. The API deploys from CI when a `v*` tag is pushed.

```bash
uv run python -m fraud.serving.export --alias champion --push   # push the champion to its `champion` HF Hub branch
docker compose up -d db                                          # local Postgres (port 5433)
uv run uvicorn api.main:app --reload                              # run the API locally (needs DATABASE_URL, API_KEY)
cd web && npm install && npm run dev                              # run the frontend locally (needs web/.env.local)
```

API endpoints: `POST /v1/predict`, `POST /v1/predict/batch` (CSV or JSON), `POST /v1/feedback`,
`GET /v1/model`, `GET /v1/monitoring/{nightly,replay}`, `GET /v1/monitoring/nightly/report` (the
Evidently HTML), `GET /health`. `POST /v1/predict/batch` takes `?source=replay` to mark drift-replay
rows. OpenAPI docs at `/docs` once the service is running.

## Monitoring (Phase 6)

A scheduled GitHub Action (`.github/workflows/monitoring.yml`, `python -m monitoring.nightly`)
reads the API's prediction log and request log from Postgres and checks five layers: service error
rate, invalid-input rate, input drift (PSI against a validation-split reference), flag-rate drift,
and recall on labelled predictions. Thresholds are in `params.yaml` under `monitoring`; the run
fails when an alert fires. Results and an Evidently report show on the web app's `/monitoring` page
(the ~4 MB report is sent gzipped, to stay under Vercel's function response cap).
`python -m monitoring.replay [--shift]` replays held-out rows through the live API, with and
without every amount inflated x3; against the live API the clean run showed no drift and the shifted
run flagged `amt` (PSI 1.29). `docs/runbook.md` covers alerts and retraining; design in ADR 0006.
Limits: replay rows stay in the prediction log (source `replay`, excluded from the checks), and
performance monitoring needs labels posted to `/v1/feedback`. Small daily samples on high-cardinality
features such as `state` can sit close to the PSI threshold, so early alerts on a low-traffic demo
deserve a look before action.

## What the data looks like

Details are in `notebooks/01_eda.ipynb` and `docs/data_cards/sparkov.md`.

**Caveat: this dataset is simulated and easy.** Most fraud sits in a 22:00 to 03:59 window (84.8% of frauds, 23.5%
of traffic), fraud amounts fall in two fixed bands with a ceiling, and a depth-5 decision tree on five raw columns
already reaches validation ROC AUC 0.96. Metrics on Sparkov will look better than they would on real traffic and
should be read with that in mind. The ULB benchmark (Phase 7, below) puts a number on that.

## ULB benchmark (Phase 7)

The same protocol on real, anonymised data: the Kaggle ULB European-cardholders set (284,807
transactions over two days, 0.173% fraud, PCA features V1–V28; `docs/data_cards/ulb.md`). It has
its own stages (`ulb_ingest`, `ulb_split`, `ulb_train`), feature pipeline (`features-ulb-1.0.0`:
amount scaling and time-of-day only), MLflow experiment (`fraud-detection-ulb`) and registered model
(`fraud-ulb`, v1 `@champion` = XGBoost; model card `docs/model_cards/ulb-v1.2.md`). It is never served; the demo and API only use `fraud-classifier`.
Split by `Time` (train < 28 h, validation 28–40 h, test after 40 h), threshold picked on validation for
precision ≥ 0.50, test scored once (`python -m fraud.ulb.evaluate_test`). The four rungs (logistic
regression, random forest, LightGBM, XGBoost) use fixed, untuned hyperparameters: with 82 frauds in
validation, tuning would mostly fit noise.

| | Sparkov (synthetic) | ULB (real) |
| --- | --- | --- |
| Model | LightGBM, v2 features (tuned) | XGBoost (untuned) |
| Frauds in test | 936 | 77 |
| Validation recall @ precision 0.50 | 0.990 | 0.817 |
| Test recall | 0.986 (CI 0.978–0.993) | 0.792 (CI 0.699–0.877) |
| Test precision | 0.446 (CI 0.423–0.468) | 0.550 (CI 0.457–0.637) |
| Test PR-AUC | 0.973 | 0.803 (CI 0.715–0.882) |
| Precision ≥ 0.50 on test | no | yes, but the CI spans 0.50 |

What the gap says, with care: the rungs are close on ULB (validation recall 0.79–0.82), and Sparkov's
0.99 does not carry over to real data, which supports treating Sparkov numbers as optimistic. The
comparison is not like for like: different models and tuning, ULB has no card history or
merchant fields (so Sparkov's most useful v2 features cannot exist there), and ULB's 77 test frauds
give wide intervals. Unlike Sparkov, ULB showed no validation-to-test gap, over a test window of only
a few hours, so it says nothing about drift over months.

## Links

- Code: https://github.com/webmaster99-stack/credit-card-fraud-detection
- Experiments, data and mirror: https://dagshub.com/webmaster99-stack/credit-card-fraud-detection
- Model repo: https://huggingface.co/ilian-hadzhidimitrov/fraud-classifier
- Demo (Gradio Space): https://huggingface.co/spaces/ilian-hadzhidimitrov/fraud-classifier-demo
- Full-stack frontend (Vercel): https://fraud-classifier-web.vercel.app
- Free-tier limits and infra decisions: `docs/infra.md`
- Design decisions: `docs/adr/`
- Model cards: `docs/model_cards/v0.3-model.md` (champion and Phase 3 analysis), `docs/model_cards/v1.0-api.md` (served champion),
  `docs/model_cards/v0.4-demo.md` (demo model), `docs/model_cards/ulb-v1.2.md` (ULB benchmark)
- Data cards: `docs/data_cards/sparkov.md`, `docs/data_cards/ulb.md`; runbook: `docs/runbook.md`
