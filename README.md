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

Status: Phase 0 (setup) complete, tag `v0.0-setup`. Next: Phase 1 (data ingestion and EDA).

## Links

- Code: https://github.com/webmaster99-stack/credit-card-fraud-detection
- Experiments, data and mirror: https://dagshub.com/webmaster99-stack/credit-card-fraud-detection
- Model repo: https://huggingface.co/ilian-hadzhidimitrov/fraud-classifier
- Demo (Gradio Space): https://huggingface.co/spaces/ilian-hadzhidimitrov/fraud-classifier-demo
- Free-tier limits and infra decisions: `docs/infra.md`
