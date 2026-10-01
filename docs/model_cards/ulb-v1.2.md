# Model card: ULB benchmark model (`fraud-ulb` v1)

A benchmark, not a served model. It reruns the Sparkov model ladder on the real, anonymised ULB
European-cardholders dataset to check how optimistic the Sparkov numbers are, and to show the lineage
setup handles a second dataset (`docs/adr/0007-ulb-benchmark-is-a-separate-pipeline.md`). It is never
loaded by the demo or the API.

| | |
| --- | --- |
| Model | `xgboost`, calibrated with sigmoid scaling (registry alias `champion`, version 1) |
| Feature pipeline | `features-ulb-1.0.0` (`log_amt`, `hour_sin`, `hour_cos`, V1-V28 unchanged) |
| Decision threshold | `0.0007646489539183676` (calibrated probability, picked on validation to meet precision >= 0.50) |
| Dataset | `ulb` `v1`, `dvc_data_md5` `5129b1cea80bafc8e7303a67037933f9.dir` (`data/processed_ulb`) |
| Split | by `Time`: train < 28 h, validation 28 h to < 40 h, test from 40 h |
| Git commit | `cd2435d` |
| MLflow run | `7f8a83c7f0de454a9441f7fdca52494b` (experiment `fraud-detection-ulb`, DagsHub) |
| Hyperparameters | fixed, untuned: `n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10.0` |

## How it was chosen

The same protocol as Sparkov, with one change: the four rungs use fixed hyperparameters, because with
82 validation frauds a tuning search would mostly fit noise. Each rung is calibrated and gets a
threshold on validation for precision >= 0.50; the best by validation recall at that precision wins.
Full results in `reports/ulb_ladder.json`.

| Model | Recall @ P>=0.50 (valid) | PR-AUC (valid) |
| --- | --- | --- |
| Logistic regression | 0.793 | 0.757 |
| Random forest | 0.817 | 0.754 |
| LightGBM | 0.805 | 0.751 |
| **XGBoost** | **0.817** | **0.796** |

The rungs are close (validation recall 0.79-0.82). XGBoost and random forest tie on recall; XGBoost
wins on PR-AUC. LightGBM needed `scale_pos_weight` 1.0, because at 10.0 it diverged on the ~330
training frauds; that was a stability fix made before any test data was scored.

## Performance

| Split | Precision | Recall | PR-AUC | Expected cost* | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validation | 0.500 | 0.817 | 0.796 | n/a | n/a | n/a | n/a | n/a |
| **Test** (single evaluation) | **0.550** (CI 0.457-0.637) | **0.792** (CI 0.699-0.877) | **0.803** (CI 0.715-0.882) | 0.0062 | 61 | 50 | 16 | 59,815 |

*Mean per-transaction cost, a missed fraud weighted 20x a false alarm.

The test split was scored once, by `python -m fraud.ulb.evaluate_test`
(`reports/ulb_test_evaluation.json`). Precision meets the 0.50 target on test, but the confidence
interval (0.457-0.637) spans it, so this is not a clear pass. There is no validation-to-test gap here
(unlike Sparkov), over a test window of only about eight hours.

## Known limitations

- **Few frauds.** 82 in validation and 77 in test, so every metric has a wide interval and conclusions
  are qualitative.
- **Two days of data.** The "future" in the time split is a few hours, so drift and seasonality are
  untested. Nothing here shows the model holds up over months.
- **No card history, merchant or geography.** ULB has only `Time`, `Amount` and PCA features, so
  Sparkov's most useful (card-history) features cannot exist here. The Sparkov and ULB numbers are not
  like for like (different models, tuning and features); see the README table.
- **Opaque features.** V1-V28 are PCA outputs, so explanations (SHAP) are not in business terms.
- **Not for real decisions.** A benchmark on a public research dataset, not a production fraud system.
  Data card: `docs/data_cards/ulb.md`.
