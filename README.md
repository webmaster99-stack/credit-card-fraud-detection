# Credit Card Fraud Detection

Fraud classifier for credit card transactions using classical ML, built as a reproducible,
fully lineaged pipeline (DVC + MLflow). See `CLAUDE.md` for project rules and `docs/plan.md`
for the phase plan.

Status: Phase 1 (data ingestion and EDA) complete, tag `v0.1-eda`.
Phase 0 is tagged `v0.0-setup`. Next: Phase 2 (feature pipeline).

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
uv run dvc repro       # ingest -> clean -> split; only re-runs what changed
```

`dvc repro ingest` downloads from Kaggle and needs `KAGGLE_API_TOKEN` in `.env`. The stages:

| Stage | Output | What it does |
| --- | --- | --- |
| `ingest` | `data/raw/sparkov` | Downloads the Sparkov CSVs from Kaggle |
| `clean` | `data/interim/transactions.parquet` | Parses types, drops direct identifiers, hashes card numbers, validates with Pandera |
| `split` | `data/processed/{train,valid,test}.parquet` | Time-based split; summary in `reports/split_summary.json` |

Splits are by time (train Jan 2019 to Jun 2020, validation Jul to Sep 2020, test Oct to Dec 2020), never random.
The test split is used once, at the end of Phase 3.

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
