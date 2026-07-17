"""Neural Nexus Adapter API — FastAPI application wiring.

Async FastAPI service for per-``user_id``/per-``assistant_id`` base-model
inference and LoRA adapter training (see ``FEATURE.md``). Lifespan owns the
shared clients (httpx for Auth0, boto3 for S3, a read-only psycopg pool to the
Anubis LangGraph store) and the in-process registries (training jobs +
semaphore, demo thread store). The vLLM engine is deliberately NOT created
here — it loads lazily on the first inference call so the app boots and
``/health`` responds without touching the GPU.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.cost_routes import cost_route
from src.api.inference_routes import inference_route
from src.api.training_routes import training_route
from src.config import (
    ASYNC_POSTGRES_STORE_URI,
    AWS_DEFAULT_REGION,
    TRAINING_CONCURRENCY,
)
from src.metrics.prometheus import metrics_middleware, registry
from src.security.auth import security_route

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared clients and registries; tear them down on shutdown."""
    import asyncio

    import boto3

    # Explicit timeouts instead of httpx's silent 5 s default: a short connect
    # timeout fails fast on an unreachable host, while a generous read timeout
    # tolerates a slow-but-alive upstream (Auth0).
    app.state.httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0)
    )
    app.state.s3_client = boto3.client("s3", region_name=AWS_DEFAULT_REGION)

    # Read-only pool to the Anubis LangGraph store (the training-data source).
    # Optional: without ASYNC_POSTGRES_STORE_URI the app still serves inference;
    # /train_adapter responds 503 until the store is configured.
    app.state.store_pool = None
    if ASYNC_POSTGRES_STORE_URI:
        from psycopg_pool import AsyncConnectionPool

        app.state.store_pool = AsyncConnectionPool(
            conninfo=ASYNC_POSTGRES_STORE_URI,
            min_size=0,
            max_size=2,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await app.state.store_pool.open()
    else:
        logger.warning(
            "ASYNC_POSTGRES_STORE_URI is unset: /train_adapter will respond 503."
        )

    # In-process registries (per-process; a multi-worker deployment needs Redis).
    app.state.training_jobs = {}
    app.state.training_semaphore = asyncio.Semaphore(TRAINING_CONCURRENCY)
    app.state.threads = {}

    logger.info("Application startup complete (vLLM engine loads on first message).")
    try:
        yield
    finally:
        if app.state.store_pool is not None:
            await app.state.store_pool.close()
        await app.state.httpx_client.aclose()


app = FastAPI(
    title="Neural Nexus Adapter API",
    description=(
        "Adapter-training and inference service: per-user/per-assistant LoRA "
        "adapters trained with TRL GRPO against stylometric rewards, served "
        "with vLLM multi-LoRA inference."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.middleware("http")(metrics_middleware)


@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    """Liveness probe; touches neither the GPU nor any external service."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def documentation():
    return RedirectResponse(url="/docs")


app.include_router(router=security_route)
app.include_router(router=inference_route)
app.include_router(router=training_route)
app.include_router(router=cost_route)
