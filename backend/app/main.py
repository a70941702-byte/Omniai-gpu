"""OmniAI backend entry point.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI

from .api.routes import router
from .config import API_PREFIX, APP_NAME
from .security import audit
from .observability.metrics import MetricsMiddleware

app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(MetricsMiddleware)
app.include_router(router, prefix=API_PREFIX)


@app.on_event("startup")
def _startup() -> None:
    audit.log("system", "server_started", {"app": APP_NAME})


@app.get("/")
def root() -> dict:
    return {"app": APP_NAME, "docs": "/docs", "api": API_PREFIX}
