-- Prediction log and online per-card history, in one table (docs/plan.md Phase 5 architecture).
-- Every column the v1 input schema requires (`fraud.serving.schema.REQUIRED_COLUMNS`) is stored
-- verbatim, plus `card_id`, so a later request for the same card can be re-featurized exactly as
-- fraud.features.history expects (trans_ts, card_id, amt, category, ... all present).

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS predictions (
    request_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    card_id TEXT NOT NULL,
    -- Naive, like every other timestamp in this project (Sparkov has no timezone concept and
    -- fraud.features treats trans_ts as wall-clock). TIMESTAMPTZ would come back tz-aware from
    -- Postgres and break pd.concat with a freshly-submitted naive request row (mixed dtype).
    trans_ts TIMESTAMP NOT NULL,
    amt DOUBLE PRECISION NOT NULL,
    category TEXT NOT NULL,
    gender TEXT NOT NULL,
    state TEXT NOT NULL,
    city_pop INTEGER NOT NULL,
    dob DATE NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    long DOUBLE PRECISION NOT NULL,
    merch_lat DOUBLE PRECISION NOT NULL,
    merch_long DOUBLE PRECISION NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL,
    flagged BOOLEAN NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'single',  -- 'single' or 'batch'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    true_label BOOLEAN,
    label_recorded_at TIMESTAMPTZ
);

-- Card history lookups: "every earlier row for this card", ordered by time.
CREATE INDEX IF NOT EXISTS idx_predictions_card_ts ON predictions (card_id, trans_ts);
