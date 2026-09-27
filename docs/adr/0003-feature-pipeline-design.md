# ADR 0003: Feature pipeline design (two tiers, backward-only history, versioning)

- Status: accepted
- Date: 2026-09-26

## Context

The same features must be produced in training, in the Gradio demo and in the API, and card-history features must
never see the future. The plan splits the pipeline into a stateless v1 tier and a stateful v2 tier.

## Decision

- **One scikit-learn `Pipeline` per tier**: stateless group transformers (transaction, customer, geography, and for v2
  card history) combined by `FeatureUnion`, then a `ColumnTransformer` (median imputer + standard scaler for numeric
  columns, one-hot for `category`/`gender`/`state`, pass-through for flags). Everything fitted is inside the pipeline,
  so it is fitted on whatever `fit` receives (the training split).
- **History definition**: for a transaction at time `t` on card `c`, history is every other transaction on `c` with a
  timestamp **strictly earlier** than `t`. Velocity windows are `[t - w, t)` for `w` in 1 h, 24 h, 168 h. Same-timestamp
  rows do not see each other, so results do not depend on row order. Behavioural features (amount / prior card mean,
  hours since last transaction, first use of this category, first transaction on the card) use the same definition.
- **Batch versus online**: `CardHistoryFeatures` uses the rows in the frame it is given. `transform_with_context`
  prepends stored earlier rows (Postgres in Phase 5) and returns only the new rows. Offline evaluation on valid uses
  train as context, so valid is not scored as if every card were new on 1 July.
- **Cold start**: a card's first transaction has no history. Its ratio and time-since-last are imputed with the training
  median and flagged by `is_first_card_txn`.
- **High-cardinality columns** (`merchant`, `job`, `city`, `zip`, `lat`/`long` of the customer) are **dropped**, not
  target-encoded. This keeps the pipeline free of a target-dependent, cross-fitted step for now. It has not been shown
  that they add nothing; Phase 3 should compare against a cross-fitted target encoding and reopen this if it helps.
- **Versioning**: a single `PIPELINE_VERSION` (`features-1.0.0`, first release covering both tiers). The tier is
  recorded as `feature_set` in the pipeline-generated `reports/feature_list_<tier>.json`, next to the pipeline version
  and the final column names and dtypes. Semver rules are those in `CLAUDE.md`.
- **The `features` DVC stage does not materialise feature matrices.** The deployable unit is the pipeline fitted with the
  model, and matrices would only add storage. The stage fits both tiers on train, transforms valid with train as
  context, fails on NaN or infinite values, and writes the feature lists and a summary. It reads train and valid only.

## Consequences

- The leakage guarantee is a property of one function and is tested directly (truncation, tampering with the future,
  row shuffling, same-timestamp rows), plus offline-versus-online parity tests.
- The online store in Phase 5 must reproduce the definitions above exactly; the parity test is the contract. It
  currently uses the full earlier history as context, so a store that keeps only the last 7 days plus running
  aggregates (count, sum, categories seen, last timestamp) needs its own parity test.
- Adding a feature is a minor bump, changing an existing feature's output is a major bump, and the feature lists show
  the difference in review.
