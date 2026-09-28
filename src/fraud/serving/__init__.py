"""Serving: load the deployable bundle, score transactions and explain the scores.

Shared by the Gradio demo (Phase 4) and the FastAPI service (Phase 5).
"""

from fraud.serving.explain import Reason
from fraud.serving.model import ServingModel, explain, load_model, predict, write_bundle
from fraud.serving.schema import InputValidationError

__all__ = [
    "InputValidationError",
    "Reason",
    "ServingModel",
    "explain",
    "load_model",
    "predict",
    "write_bundle",
]
