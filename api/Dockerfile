# FastAPI service for Render. Built from the repo root: docker build -f api/Dockerfile .
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first, so code-only changes don't invalidate the layer. Only the `api` group: the
# demo (gradio) and dev (mypy, pytest, ...) groups have no place in the production image.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-default-groups --group api

COPY README.md ./
COPY src ./src
COPY api ./api
RUN uv sync --locked --no-editable --no-default-groups --group api

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
