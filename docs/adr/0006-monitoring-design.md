# ADR 0006: Monitoring computes its own metrics; Evidently renders the report

Status: accepted (Phase 6)

## Context

Phase 6 needs daily drift and prediction checks with explicit alert thresholds, plus a report page.
Evidently produces good drift reports, but its API has changed between major versions and its
results are awkward to threshold ("3 days running", "flag rate 0.5x-2x") and to unit-test.

## Decision

- Drift statistics (PSI, Wasserstein) and every alert rule are plain functions in
  `src/fraud/monitoring/drift.py`, tested without a database or Evidently. Thresholds are in
  `params.yaml`.
- Evidently is only used to render the HTML drift report (`monitoring` dependency group, installed
  only in the nightly workflow). If it is missing or fails, the job still stores its summary.
- The reference is a sample of the validation split (never test), built by the
  `monitoring_reference` dvc stage; the nightly job `dvc pull`s only that output.
- Storage reuses the API's Neon database: `api_requests` (service and data-quality layers, since
  rejected requests never reach `predictions`) and `monitoring_reports` (job output, served by
  `/v1/monitoring/*`). The web app reads it through its server-side routes, keeping the API key
  off the browser.
- The job runs as a scheduled GitHub Action and fails when any alert fires; a red run is the alert.
- Drift is monitored on raw inputs plus `age`, `distance_km`, `hour`, not on the v2 velocity
  features, because those depend on card history and would need the history store to recompute.

## Consequences

- Alert logic is verified by unit tests; Evidently version drift can only break the HTML report.
- The nightly job and replay script write to the production database. Replay rows are logged with
  source `replay` (batch endpoint `?source=replay`) and the nightly job excludes them; the first
  replays ran before this existed and were relabelled by a one-off UPDATE.
- Performance monitoring is only as good as the labels posted to `/v1/feedback`.
