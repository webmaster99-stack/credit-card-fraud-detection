# ADR 0001: Use Sparkov as the primary dataset

- Status: accepted
- Date: 2026-09-25

## Context

The project needs a public credit card fraud dataset that is large enough to make time-based splits, per-card
history features and drift monitoring meaningful, and that carries no real cardholder data.

## Options considered

- **Sparkov simulated transactions** (~1.85M rows, ~0.5% fraud, 2019-2020, CC0). Has timestamps, card identity,
  merchant, category, amount and geography. Simulated.
- **Kaggle ULB European cardholders** (284,807 rows, two days, PCA-anonymised features V1-V28). Real, but
  covers only two days and the features are anonymised, so there is nothing to engineer and no card history.

## Decision

Sparkov is the primary dataset. ULB is added in Phase 7 as a secondary benchmark only.

## Consequences

- Long time span and card identity allow time-based splits, backward-looking velocity features and a drift replay.
- No real cardholder data, so the demo and repo can be public without privacy risk.
- The simulator is easy to beat (see `notebooks/01_eda.ipynb` and `docs/data_cards/sparkov.md`). Metrics must be
  presented with that caveat, and the ULB benchmark provides a harder, real reference point.
