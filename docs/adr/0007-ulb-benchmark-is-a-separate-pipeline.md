# ADR 0007: The ULB benchmark is a separate pipeline, not a second config of the Sparkov one

Status: accepted (Phase 7)

## Context

Phase 7 reruns the model ladder on the real, anonymised ULB dataset to check how optimistic the
Sparkov numbers are, and to show the lineage setup copes with several datasets. ULB has no card id,
merchant, geography or dates, only `Time` (seconds), V1-V28 (PCA) and `Amount`.

## Decision

- Own package `src/fraud/ulb/`, own dvc stages, own folders (`data/raw/ulb`, `data/processed_ulb`),
  own feature pipeline version `features-ulb-1.0.0`, own MLflow experiment and registered model
  `fraud-ulb`. Nothing shares a name with the served `fraud-classifier`.
- Reused, not copied: the estimator builders, calibration and threshold classes, metrics, and the
  lineage helper. `start_run` and `lineage_tags` gained optional `experiment_name` and
  `lineage_overrides`, so ULB runs carry the same tag names with ULB values (dataset, DVC hash of
  `data/processed_ulb`, pipeline version, split spec).
- Split by `Time` in hours (train < 28 h, validation 28-40 h, test after). The boundaries were moved
  once from 29/39 h because validation held only 53 frauds; the choice used fraud counts only, before
  any model was fitted.
- Fixed, untuned hyperparameters for logistic regression, random forest, LightGBM and XGBoost. With
  82 validation frauds a search would mostly fit noise. LightGBM's `scale_pos_weight` is 1.0 because
  10.0 made it diverge; this was a stability fix, not tuning on test.
- The test split is scored once by `fraud.ulb.evaluate_test`, deliberately not a dvc stage, and no
  stage depends on `test.parquet`.

## Consequences

- The Sparkov and ULB numbers are comparable in protocol, not in model effort (Sparkov's champion was
  tuned with Optuna and uses card-history features), and the README says so.
- Small counts (77 test frauds) mean wide bootstrap intervals; conclusions stay qualitative.
- Adding a third dataset repeats this pattern rather than parameterising the Sparkov stages.
