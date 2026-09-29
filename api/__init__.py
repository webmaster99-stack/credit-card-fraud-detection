"""FastAPI service: scores transactions with the champion model, backed by `src/fraud/serving`.

Shares the same scoring code as the Gradio demo (Phase 4), so the API and the demo can never
disagree. Postgres holds the prediction log and the online per-card history that the v2 (stateful)
feature pipeline needs.
"""
