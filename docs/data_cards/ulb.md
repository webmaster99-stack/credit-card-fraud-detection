# Data card: ULB European cardholders (secondary benchmark)

| | |
| --- | --- |
| Dataset name / version | `ulb` / `v1` (`ulb.dataset` in `params.yaml`) |
| Source | Kaggle, `mlg-ulb/creditcardfraud` (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud); collected by Worldline and the ULB Machine Learning Group |
| Licence | Database Contents License (DbCL) v1.0 for the contents; the dataset page cites the original papers. Use here is a research benchmark only. |
| File used | `creditcard.csv` |
| Rows | 284,807 |
| Period | Two days of transactions in September 2013 (exact dates withheld; `Time` is seconds since the first row) |
| Fraud rate | 0.173% overall (492 frauds) |
| Role in this project | **Benchmark only** (Phase 7). Never served by the demo or the API. |

## Columns

- `Time`: seconds since the first transaction (0 to 172,792).
- `V1`-`V28`: PCA components of the original (undisclosed) features, already anonymised.
- `Amount`: transaction amount in euros.
- `Class`: 1 = fraud.

There is no card identifier, so card-history features (Sparkov `v2`) are impossible here.
No merchant, category, geography or customer attributes exist either.

## How it enters the pipeline

`dvc.yaml` stages `ulb_ingest` (download to `data/raw/ulb`), `ulb_split` (`data/processed_ulb/{train,valid,test}.parquet`
and `reports/ulb_split_summary.json`) and `ulb_train`. The raw file is validated by a Pandera schema
(`src/fraud/ulb/data.py`): exact columns, dtypes, non-negative time and amount, binary label.
`data/processed_ulb` is a sibling of `data/processed` because DVC forbids nested outputs.

## Split (by Time, never random)

| Split | Window | Rows | Frauds | Fraud rate |
| --- | --- | --- | --- | --- |
| train | Time < 28 h | 153,944 | 333 | 0.216% |
| valid | 28 h to < 40 h | 70,921 | 82 | 0.116% |
| test | 40 h onward | 59,942 | 77 | 0.128% |

The hour boundaries were chosen from fraud counts alone, before any model was fitted, after a first
choice (29 h / 39 h) left only 53 frauds in validation, too few to place a threshold. The test split
is scored once, by `fraud.ulb.evaluate_test`.

## Features (`features-ulb-1.0.0`)

Amount scaling and time-of-day only: `log_amt` (log1p, standardised on train), `hour_sin` and `hour_cos`
of `Time` modulo 24 h, and V1-V28 unchanged.

## Known quirks and limitations

- **`Time` is not a clock time.** The dataset does not say when the first transaction happened, so the
  time-of-day feature is the position in a 24 h cycle relative to the start, not a real hour of day.
- **Only 48 hours.** The "future" in the time split is a few hours, so drift and seasonality are not
  tested. A model that looks good here has not been shown to hold up over months.
- **Few frauds per split** (82 in validation, 77 in test), so metrics have wide confidence intervals. The
  test evaluation reports bootstrap intervals for this reason.
- **Duplicates.** The file contains a small number of exact duplicate rows; they are kept, as in most
  published benchmarks.
- **Features are opaque.** SHAP on V1-V28 explains the model but not in business terms.
