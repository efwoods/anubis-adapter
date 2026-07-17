"""Prometheus metrics for the adapter service.

The request metrics reuse the main Anubis API's metric names
(``anubis_requests_total`` etc., see ``scaffold/webapp.py``) so the existing
Grafana dashboards read both services; the adapter-specific series
(``adapter_*``) carry the cost/capacity signals FEATURE.md requires the
service to report.
"""

from __future__ import annotations

import uuid
from time import time_ns

from fastapi import Request
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "anubis_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "anubis_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    registry=registry,
)

ACTIVE_REQUESTS = Gauge(
    "anubis_active_requests", "Number of active requests", registry=registry
)

API_RESPONSE_STATUS = Counter(
    "anubis_api_response_status_total",
    "Response status codes",
    ["status"],
    registry=registry,
)

ADAPTER_INFERENCE_TOKENS_TOTAL = Counter(
    "adapter_inference_tokens_total",
    "Completion tokens generated, by base model and attached adapter "
    "('base' when no adapter is attached)",
    ["model", "adapter"],
    registry=registry,
)

ADAPTER_TRAINING_JOBS_ACTIVE = Gauge(
    "adapter_training_jobs_active",
    "Adapter-training jobs currently running",
    registry=registry,
)

ADAPTER_TRAINING_DURATION_SECONDS = Histogram(
    "adapter_training_duration_seconds",
    "Wall-clock duration of completed adapter-training runs",
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800),
    registry=registry,
)

ADAPTER_SIZE_BYTES = Gauge(
    "adapter_size_bytes",
    "Size in bytes of the most recently trained adapter",
    ["user_id", "assistant_id"],
    registry=registry,
)


async def metrics_middleware(request: Request, call_next):
    """Request-id stamping + count/latency/active-gauge observation.

    Mirrors the main Anubis API's middleware so the two services report
    identically shaped request metrics.
    """
    start_time = time_ns()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        latency_ms = (time_ns() - start_time) // 1_000_000

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=str(request.url.path),
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method, endpoint=str(request.url.path)
        ).observe(latency_ms / 1000)
        API_RESPONSE_STATUS.labels(status=response.status_code).inc()

        return response
    except Exception:
        API_RESPONSE_STATUS.labels(status=500).inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()
