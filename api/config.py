"""API settings, read from the environment (and `.env` locally). No secrets have defaults."""

from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from fraud.params import REPO_ROOT, load_params


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str
    api_key: str
    # Comma-separated Vercel origin(s) allowed to call the API from a browser.
    allowed_origins: str = "http://localhost:3000"
    # A bundle directory, or a Hugging Face Hub repo id; defaults to params.yaml's hf_model_repo.
    model_source: str | None = None
    # The HF Hub branch/tag the champion bundle is pushed to (see fraud.serving.export --alias).
    model_revision: str = "champion"
    rate_limit: str = "60/minute"
    # Rows of a card's stored history fetched as context for one prediction.
    max_history_rows: int = 2000

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # fields come from the environment


def serving_defaults() -> dict[str, Any]:
    """Non-secret serving knobs shared with the demo (`params.yaml`'s `serving` section)."""
    params = load_params()
    serving = params["serving"]
    return {
        "hf_model_repo": serving["hf_model_repo"],
        "max_batch_rows": serving["max_batch_rows"],
        "top_k_reasons": serving["top_k_reasons"],
        "batch_explain_rows": serving["batch_explain_rows"],
    }
