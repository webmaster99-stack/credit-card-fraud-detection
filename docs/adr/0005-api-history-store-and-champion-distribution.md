# ADR 0005: The Phase 5 API's history store, batch semantics, and how the champion reaches it

- Status: accepted
- Date: 2026-09-29

## Context

Phase 5 serves the Phase 3 champion (`fraud-classifier@champion`, lightgbm on v2 features), which
needs a card's earlier transactions to compute velocity and behavioural features
(`fraud.features.history`). Offline, "history" is every earlier row in the training/serving
dataframe. Online, it has to come from somewhere durable, and the champion bundle has to reach the
API without the API needing DVC or MLflow credentials (the same constraint Phase 4 solved for the
demo).

## Decisions

**One Postgres table is both the prediction log and the history store.** `api/schema.sql`'s
`predictions` table stores every column the v1 input schema requires, plus `card_id`, the score, and
model/pipeline version, for every request (single and batch). `fraud.features.pipeline.
transform_with_context` already accepts "context" as an arbitrary earlier-transactions frame,
so `api/db.py`'s `fetch_card_history` just queries this same table by `card_id` and hands the result
back as that context - no second table, no separate write path. A future `/v1/feedback` label lands
on the same row it corrects.

**Batch scoring (`/v1/predict/batch`) never reads the history store.** History for a batch comes only
from the other rows in that same request (`fraud.features.history` groups by whatever's in the frame
passed to it, self-referentially, when no context is given). This keeps a batch's results independent
of *when* it happens to be uploaded relative to other traffic, and independent of request order across
concurrent batches - a batch is a pure function of its own file. It does still get logged to Postgres
afterwards, so a later single-transaction request for the same card benefits from it.

**The champion bundle is a branch on the same HF Hub model repo as the demo, not a second repo.**
`fraud.serving.export --alias champion --push` pushes to a `champion` branch (`--revision`); the demo
keeps publishing to `main`. `fraud.serving.load_model` already takes a `revision`; the API passes
`MODEL_REVISION` (default `champion`) so it downloads from that branch at startup. One repo keeps the
model card, feature list and lineage tags alongside each other under one URL instead of splitting them
across repos that could drift out of sync.

**Auth is one shared API key, not per-user keys.** `CLAUDE.md`'s access-model decision is "public
demo protected by an API key; no user accounts" - a single `X-API-Key` header checked against one
`API_KEY` secret matches that exactly, with no user table to add.

**Rate limiting is in-process (`slowapi`), not a shared store.** Render's free tier is a single
instance with no Redis, so a per-instance in-memory limiter is the whole budget; it resets on restart
and would need a shared backend if the service ever scaled to more than one instance.

## Consequences

- A card's history is only as deep as what the API has actually seen since it started logging
  predictions; a brand-new card (or the store shortly after a redeploy that wiped the database) scores
  as a first-ever transaction, same as the offline model would for a card with no prior rows. The
  champion's model card states this.
- `write_bundle` (`src/fraud/serving/model.py`) needed a one-line fix: it built its `feature_list.json`
  probe frame from the input template alone, which has no `card_id`, so a v2 pipeline's history step
  failed feature-name discovery. A synthetic `card_id` fixes it for both the demo and the champion
  bundle without changing what either bundle ships.
- `trans_ts` is stored as `TIMESTAMP` (no time zone), not `TIMESTAMPTZ`: Sparkov and every existing
  transformer treat it as naive wall-clock, and a `TIMESTAMPTZ` round trip would hand back
  timezone-aware values that fail to concatenate with a freshly-submitted naive request row.
