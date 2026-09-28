# ADR 0004: The Phase 4 demo serves a stateless v1 model under its own registry alias

- Status: accepted
- Date: 2026-09-28

## Context

The Phase 3 champion (`fraud-classifier` v1, `@champion`, lightgbm on v2 features) depends on per-card
transaction history. A form or CSV row in the Gradio demo carries no history, and the online history store
arrives in Phase 5. `docs/plan.md` therefore says the demo ships the v1 stateless pipeline first.

## Decision

- Train the best v1 rung of the ladder (xgboost, recall 0.921 at precision 0.50 on validation) through a
  dedicated `train_demo` DVC stage (`params.yaml` `demo_model`, same code path as `train`), register it as a new
  version of `fraud-classifier` (v3) with the alias **`demo`**, and export that alias to the serving bundle.
  `champion` and `challenger` are not moved: the Phase 3 outcome stands.
- `fraud.serving` is model-agnostic (it wraps any pipeline saved by `train.py`, including an optional `context`
  of earlier transactions for v2). Phase 5 can serve the champion by pointing the export at `@champion` and
  adding the history store; no serving code changes.
- The demo model has **not** been evaluated on the test split (spent on the champion). Its card and About tab
  report its validation metrics and state that the test metrics shown belong to the champion.
- "Try an example" rows are a seeded random sample of real test rows, displayed and scored for illustration only.
  Nothing is tuned on them, and they are not hand-picked, so some frauds shown may be ones the model misses.

## Consequences

- The demo's recall (about 0.92 on validation) is visibly below the champion's (about 0.99). This is the price of
  serving without history and is stated in the About tab.
- The registry now holds three versions: v1 `@champion`, v2 `@challenger`, v3 `@demo`.
- Gender is a model input and can appear among the reasons. That is what the model uses; the fairness caveat from
  the champion's card is repeated in the demo's limitations.
