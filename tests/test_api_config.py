import pytest
from api.config import Settings
from pydantic import ValidationError


def test_cors_origins_splits_and_strips(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        database_url="postgresql://u:p@h/db",
        api_key="k",
        allowed_origins="https://a.example, https://b.example ,",
    )
    assert settings.cors_origins == ["https://a.example", "https://b.example"]


def test_settings_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("MODEL_REVISION", "champion")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.database_url == "postgresql://u:p@h/db"
    assert settings.api_key == "secret"
    assert settings.model_revision == "champion"


def test_settings_requires_database_url_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]  # bypass the repo's real .env
