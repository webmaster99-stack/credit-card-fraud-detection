"""FastAPI dependencies: API key auth, rate limiting, and access to app-wide state."""

from typing import Annotated

from fastapi import Header, HTTPException, Request, status
from psycopg_pool import ConnectionPool
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.config import get_settings
from fraud.serving import ServingModel

# One shared rate limit for every route (see `main.create_app`); `/health` opts out with
# `@limiter.exempt` so Render's and uptime checks' liveness probes are never throttled.
limiter = Limiter(key_func=get_remote_address, default_limits=[get_settings().rate_limit])


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if x_api_key is None or x_api_key != get_settings().api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key.")


def get_model(request: Request) -> ServingModel:
    model: ServingModel = request.app.state.model
    return model


def get_pool(request: Request) -> ConnectionPool:
    pool: ConnectionPool = request.app.state.pool
    return pool
