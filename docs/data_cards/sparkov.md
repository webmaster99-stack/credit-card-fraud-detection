# Data card: Sparkov simulated credit card transactions

| | |
| --- | --- |
| Dataset name / version | `sparkov` / `v1` (see `params.yaml`) |
| Source | Kaggle, `kartik2112/fraud-detection` (https://www.kaggle.com/datasets/kartik2112/fraud-detection), generated with the open-source Sparkov Data Generation simulator |
| Licence | CC0 (public domain), as recorded for this project |
| Files used | `fraudTrain.csv` (1,296,675 rows), `fraudTest.csv` (555,719 rows) |
| Rows after cleaning | 1,852,394 |
| Period | 2019-01-01 to 2020-12-31 |
| Cards / merchants | 999 simulated cards / 693 merchants |
| Fraud rate | 0.52% overall (9,651 frauds) |
| Real cardholder data | **None.** Everything is simulated, including names and addresses. |

## How it enters the pipeline

`dvc repro` runs three stages: `ingest` (download to `data/raw/sparkov`), `clean` (`data/interim/transactions.parquet`)
and `split` (`data/processed/{train,valid,test}.parquet`, plus `reports/split_summary.json`). Everything is versioned
with DVC and stored on the DagsHub remote. A new dataset or version is a new folder plus a `dataset_version` bump,
never an overwrite.

## Cleaning

- The two Kaggle files are concatenated and **re-split by our own time windows** (below). Kaggle's own boundary
  (train ends 2020-06-21) is not used.
- Timestamps parsed to `trans_ts` (naive, no time zone given by the source). `dob` parsed to a date.
- Direct identifiers dropped: `first`, `last`, `street`, `trans_num`. The card number `cc_num` is replaced by
  `card_id`, a salted SHA-256 prefix, used only for per-card features. The salt is in `params.yaml` and is not a
  secret: the data is simulated, and the hash only keeps the raw number out of downstream tables.
- Dropped as redundant: the unnamed row index and `unix_time` (see quirks).
- The constant `fraud_` prefix on every merchant name is removed.
- The result is validated against a Pandera schema (`src/fraud/data/schemas.py`): dtypes, value ranges, known
  categories, no nulls, no extra columns.

## Split (time-based, never random)

| Split | Window | Rows | Frauds | Fraud rate |
| --- | --- | --- | --- | --- |
| train | 2019-01-01 to 2020-06-30 | 1,326,733 | 7,639 | 0.576% |
| valid | 2020-07-01 to 2020-09-30 | 244,140 | 1,076 | 0.441% |
| test | 2020-10-01 to 2020-12-31 | 281,521 | 936 | 0.332% |

The test split is used once, at the end of Phase 3. The EDA notebook never opens it.

## Known quirks and limitations

- **`unix_time` is wrong.** It equals the transaction timestamp shifted back by exactly 7 years (2,556 or 2,557 days).
  It is dropped and must never be used as a feature.
- **No transactions on 2020-02-29.** The largest gap between consecutive transactions is just over one day.
- **The Kaggle train file is not strictly time-ordered.** `clean` sorts by timestamp (stable sort).
- **Card numbers have 11 to 19 digits** in the source. Only the hash is kept.
- **The task is much easier than real fraud** because the simulator hard-codes patterns:
  - 84.8% of frauds fall between 22:00 and 03:59 (23.5% of traffic);
  - fraud amounts sit in two bands (under $50 and $200 and above) with a ceiling of $1,376, above which 1,932
    legitimate transactions sit;
  - fraud is strongly concentrated in a few categories (shopping_net, misc_net, grocery_pos hold 58% of frauds);
  - a depth-5 decision tree on five raw columns reaches validation ROC AUC 0.96. See `notebooks/01_eda.ipynb`.
- **Customer-merchant distance carries no signal** (identical distributions), unlike real data.
- **Fraud comes in bursts per card** (median 10 frauds within about 45 hours), so random splits would leak.
- Fraud prevalence differs between splits (0.58% to 0.33%). Precision at a fixed threshold depends on prevalence.
- Metrics on this data will look better than they would in production. State that wherever results are reported.
  The ULB benchmark (Phase 7) is the realism check.
