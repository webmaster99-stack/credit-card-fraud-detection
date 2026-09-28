# Credit Card Fraud Detection

Fraud classifier for credit card transactions using classical ML, built as a reproducible,
fully lineaged pipeline (DVC + MLflow). See `CLAUDE.md` for project rules and `docs/plan.md`
for the phase plan.

Status: Phase 3 (modeling experiments) complete, tag `v0.3-model`. Champion: LightGBM on v2
(card-history) features, recall 0.986 at precision 0.446 on the single test-set evaluation — see
Modeling, below. Earlier: `v0.2-features` (Phase 2), `v0.1-eda` (Phase 1), `v0.0-setup` (Phase 0).
Phase 4 (Gradio demo) complete, tag `v0.4-demo`: the Space is live (link below) and serves a
stateless v1 model, not the champion (see Demo).

## Setup

```bash
uv sync
cp .env.example .env   # then fill in DagsHub and Kaggle credentials
pre-commit install
uv run pytest
```

## Reproduce the data pipeline

```bash
uv run dvc pull        # fetch data from the DagsHub remote (no Kaggle token needed)
uv run dvc repro       # ingest -> clean -> split -> features; only re-runs what changed
```

`dvc repro ingest` downloads from Kaggle and needs `KAGGLE_API_TOKEN` in `.env`. The stages:

| Stage | Output | What it does |
| --- | --- | --- |
| `ingest` | `data/raw/sparkov` | Downloads the Sparkov CSVs from Kaggle |
| `clean` | `data/interim/transactions.parquet` | Parses types, drops direct identifiers, hashes card numbers, validates with Pandera |
| `split` | `data/processed/{train,valid,test}.parquet` | Time-based split; summary in `reports/split_summary.json` |

Splits are by time (train Jan 2019 to Jun 2020, validation Jul to Sep 2020, test Oct to Dec 2020), never random.
The test split is used once, at the end of Phase 3.

## Feature pipeline

One scikit-learn pipeline per tier, in `src/fraud/features/` (version in `features/__init__.py`, design in
`docs/adr/0003-feature-pipeline-design.md`):

| Tier | Features | Card history |
| --- | --- | --- |
| `v1` (74 columns) | Transaction (log amount, category, hour, weekday, night flag), customer (age, gender, log city population, state), customer-merchant distance | No |
| `v2` (84 columns) | v1 plus velocity (count and spend in the last 1 h / 24 h / 7 d) and behavioural (amount vs card mean, hours since last transaction, first use of category, first transaction on card) | Yes, strictly earlier rows only |

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
0.978–0.993), precision 0.446 (CI 0.423–0.468, short of the 0.50 target — a real, modest
generalization gap, reported honestly rather than fixed by re-tuning against test), PR-AUC 0.973
(CI 0.965–0.980).

## Demo (Phase 4)

A Gradio app in `demo/` scores one transaction or a CSV (up to 10,000 rows), explains each score with
SHAP reasons, and shows which model version answered. All model code lives in `src/fraud/serving/`
(`load_model`, `predict`, `explain`), which the Phase 5 API will reuse.

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

## What the data looks like

Details are in `notebooks/01_eda.ipynb` and `docs/data_cards/sparkov.md`.

**Caveat: this dataset is simulated and easy.** Most fraud sits in a 22:00 to 03:59 window (84.8% of frauds, 23.5%
of traffic), fraud amounts fall in two fixed bands with a ceiling, and a depth-5 decision tree on five raw columns
already reaches validation ROC AUC 0.96. Metrics on Sparkov will look better than they would on real traffic and
should be read with that in mind. A real-data benchmark (ULB) is planned for Phase 7.

## Links

- Code: https://github.com/webmaster99-stack/credit-card-fraud-detection
- Experiments, data and mirror: https://dagshub.com/webmaster99-stack/credit-card-fraud-detection
- Model repo: https://huggingface.co/ilian-hadzhidimitrov/fraud-classifier
- Demo (Gradio Space): https://huggingface.co/spaces/ilian-hadzhidimitrov/fraud-classifier-demo
- Free-tier limits and infra decisions: `docs/infra.md`
- Design decisions: `docs/adr/`
- Model cards: `docs/model_cards/v0.3-model.md` (champion), `docs/model_cards/v0.4-demo.md` (demo model)
