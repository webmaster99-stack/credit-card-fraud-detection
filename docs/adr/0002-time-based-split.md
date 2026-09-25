# ADR 0002: Time-based train/validation/test split

- Status: accepted
- Date: 2026-09-25

## Context

In production the model scores future transactions using only the past. Sparkov fraud arrives in bursts per card
(median 10 frauds within about 45 hours; 89.9% of frauds directly follow another fraud on the same card), so a
random split would place near-duplicate fraud episodes in both train and test and inflate every metric.

## Decision

Split by time, using windows in `params.yaml`:

- train 2019-01-01 to 2020-06-30
- validation 2020-07-01 to 2020-09-30
- test 2020-10-01 to 2020-12-31

The two Kaggle files are concatenated and re-split with these windows (Kaggle's own train/test boundary is
2020-06-21 and is not used). Windows are validated to be ordered and disjoint, and the split fails if any row falls
outside every window. The test split is evaluated once, at the end of Phase 3, and never used for tuning. The EDA
only reads train and validation.

## Consequences

- Metrics are honest estimates of forward performance.
- Fraud prevalence drifts across splits (0.58%, 0.44%, 0.33%), so precision at a fixed threshold is not directly
  comparable between them. The precision budget (>= 0.50) is set on validation and confirmed once on test.
- Cards recur across splits (the same simulated customers), which matches production, where returning cards are the
  norm. Card-level generalisation to brand-new cards is not measured.
- Any card-history feature must look backwards only.
