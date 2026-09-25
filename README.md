# Credit Card Fraud Detection

Fraud classifier for credit card transactions using classical ML, built as a reproducible,
fully lineaged pipeline (DVC + MLflow). See `CLAUDE.md` for project rules and `docs/plan.md`
for the phase plan.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in DagsHub credentials
pre-commit install
uv run pytest
```

Status: Phase 0 (setup) in progress.
