"""Cost and break-even reporting endpoint (FEATURE.md metrics surface)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.config import GPU_MEMORY_UTILIZATION
from src.metrics.cost import (
    MODEL_PRICING,
    adapter_storage_cost_per_month,
    basemodel_storage_cost,
    break_even,
    concurrency_capacity,
    training_cost,
)

cost_route = APIRouter()


@cost_route.get("/cost/estimate")
async def cost_estimate(
    model: str = Query("meta-llama/Llama-3.2-1B-Instruct"),
    training_hours: float = Query(1.0, ge=0),
    adapter_size_mb: float = Query(50.0, ge=0),
    monthly_price_per_user: float = Query(10.0),
):
    """Estimate training + storage cost, serving capacity, and break-even.

    Inputs default to the current base model with a one-hour training run and
    a 50 MB adapter; ``monthly_price_per_user`` is the subscription price the
    break-even user count is computed against.
    """
    if model not in MODEL_PRICING:
        raise HTTPException(
            status_code=404,
            detail=f"No pricing entry for model {model!r}. "
            f"Known models: {sorted(MODEL_PRICING)}",
        )

    adapter_size_bytes = int(adapter_size_mb * 1024 * 1024)
    training_cost_report = training_cost(model, training_hours)
    basemodel_storage_report = basemodel_storage_cost(model)
    capacity_report = concurrency_capacity(model, GPU_MEMORY_UTILIZATION)

    # A month of always-on GPU serving plus the volume: the fixed cost the
    # subscription revenue has to cover.
    hours_per_month = 24 * 30
    monthly_gpu_cost = (
        MODEL_PRICING[model]["gpu_cost_per_hour_usd"] * hours_per_month
    )
    monthly_fixed_cost = (
        monthly_gpu_cost
        + basemodel_storage_report["runpod_volume_cost_per_month_usd"]
        + basemodel_storage_report["s3_mirror_cost_per_month_usd"]
    )

    return {
        "model": model,
        "training": training_cost_report,
        "adapter_storage": {
            "adapter_size_bytes": adapter_size_bytes,
            "cost_per_month_usd": round(
                adapter_storage_cost_per_month(adapter_size_bytes), 6
            ),
        },
        "basemodel_storage": basemodel_storage_report,
        "concurrency": capacity_report,
        "monthly_fixed_cost_usd": round(monthly_fixed_cost, 2),
        "break_even": break_even(monthly_fixed_cost, monthly_price_per_user),
    }
