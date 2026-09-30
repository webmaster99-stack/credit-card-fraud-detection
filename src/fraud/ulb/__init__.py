"""Phase 7: the ULB European-cardholders benchmark (secondary dataset).

Kept apart from the Sparkov pipeline on purpose: its own feature pipeline version, data folders,
MLflow experiment and registered model, so it can never leak into what the demo and API serve.
"""

PIPELINE_VERSION = "features-ulb-1.0.0"
