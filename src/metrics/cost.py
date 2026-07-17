"""Cost, capacity, and break-even calculators for adapter training + serving.

Pricing inputs are transcribed from ``FEATURE.md`` (RunPod GPU/volume pricing,
S3 storage tiers). These calculators answer the questions FEATURE.md requires
the service to report:

* memory required for and cost of LoRA adapter training (time, capital),
* LoRA adapter size and cost of storage,
* number of concurrent users served for multi-adapter inference,
* number of concurrent users served for adapter training,
* the break-even point for running inference + training.

After the first real training run, feed the measured numbers back into
``README.md``'s Resource Requirements section using these functions.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from src.config import MAX_LORAS, TRAINING_CONCURRENCY

GIGABYTE_BYTES = 1024**3

# Per-model deployment pricing from FEATURE.md. ``weights_gigabytes`` is the
# bf16 weight footprint (parameters × 2 bytes); the Llama-3.2-1B entry is the
# current resource-limited base model (~2.5 GB weights, single A40 assumed).
MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "weights_gigabytes": 2.5,
        "volume_gigabytes": 50,
        "volume_cost_per_month_usd": 3.50,
        "gpu_configuration": "1xA40 (48 GB vRAM)",
        "gpu_vram_gigabytes": 48,
        "gpu_cost_per_hour_usd": 0.44,
    },
    "meta-llama/Llama-3.2-11B-Vision-Instruct": {
        "weights_gigabytes": 22,
        "volume_gigabytes": 50,
        "volume_cost_per_month_usd": 3.50,
        "gpu_configuration": "1xA40 (48 GB vRAM)",
        "gpu_vram_gigabytes": 48,
        "gpu_cost_per_hour_usd": 0.44,
    },
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "weights_gigabytes": 218,
        "volume_gigabytes": 250,
        "volume_cost_per_month_usd": 17.50,
        "gpu_configuration": "3xA100 SXM (80 GB vRAM each, 240 GB combined)",
        "gpu_vram_gigabytes": 240,
        "gpu_cost_per_hour_usd": 4.47,
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": {
        "weights_gigabytes": 804,
        "volume_gigabytes": 1000,
        "volume_cost_per_month_usd": 70.00,
        "gpu_configuration": "3xH200 SXM (141 GB vRAM each, 423 GB combined)",
        "gpu_vram_gigabytes": 423,
        "gpu_cost_per_hour_usd": 13.17,
    },
}

# S3 standard-storage tiers from FEATURE.md: (tier ceiling in gigabytes,
# price per gigabyte-month in USD). 50 TB, then the next 450 TB, then the rest.
S3_STORAGE_TIERS = [
    (50_000.0, 0.023),
    (500_000.0, 0.022),
    (math.inf, 0.021),
]


def adapter_storage_cost_per_month(size_bytes: int) -> float:
    """Monthly S3 cost in USD to store one adapter of ``size_bytes``.

    Applies the tiered S3 pricing progressively (the first 50 TB at the first
    tier's rate, and so on) — for a single adapter this is effectively the
    first-tier rate, but the function stays correct at any size.
    """
    remaining_gigabytes = size_bytes / GIGABYTE_BYTES
    previous_ceiling = 0.0
    total_cost = 0.0
    for tier_ceiling_gigabytes, price_per_gigabyte_month in S3_STORAGE_TIERS:
        tier_capacity = tier_ceiling_gigabytes - previous_ceiling
        billable_gigabytes = min(remaining_gigabytes, tier_capacity)
        if billable_gigabytes <= 0:
            break
        total_cost += billable_gigabytes * price_per_gigabyte_month
        remaining_gigabytes -= billable_gigabytes
        previous_ceiling = tier_ceiling_gigabytes
    return total_cost


def training_cost(model: str, duration_hours: float) -> Dict[str, Any]:
    """GPU compute cost in USD for one training run of ``duration_hours``."""
    model_pricing = MODEL_PRICING.get(model)
    if model_pricing is None:
        raise KeyError(f"No pricing entry for model {model!r}.")
    gpu_cost = model_pricing["gpu_cost_per_hour_usd"] * duration_hours
    return {
        "model": model,
        "gpu_configuration": model_pricing["gpu_configuration"],
        "gpu_cost_per_hour_usd": model_pricing["gpu_cost_per_hour_usd"],
        "duration_hours": duration_hours,
        "gpu_cost_usd": round(gpu_cost, 4),
    }


def basemodel_storage_cost(model: str) -> Dict[str, Any]:
    """Monthly storage cost of one base model: RunPod volume + S3 mirror."""
    model_pricing = MODEL_PRICING.get(model)
    if model_pricing is None:
        raise KeyError(f"No pricing entry for model {model!r}.")
    weights_bytes = int(model_pricing["weights_gigabytes"] * GIGABYTE_BYTES)
    return {
        "model": model,
        "weights_gigabytes": model_pricing["weights_gigabytes"],
        "runpod_volume_gigabytes": model_pricing["volume_gigabytes"],
        "runpod_volume_cost_per_month_usd": model_pricing["volume_cost_per_month_usd"],
        "s3_mirror_cost_per_month_usd": round(
            adapter_storage_cost_per_month(weights_bytes), 4
        ),
    }


def concurrency_capacity(model: str, gpu_memory_utilization: float) -> Dict[str, Any]:
    """Concurrent-user capacity of one deployment of ``model``.

    * multi-adapter inference — the vLLM engine serves up to ``MAX_LORAS``
      distinct adapters resident simultaneously (``enable_lora`` slot count);
      requests beyond that queue on adapter-slot eviction.
    * adapter training — the training semaphore admits ``TRAINING_CONCURRENCY``
      concurrent jobs into the GPU head-room left by
      ``gpu_memory_utilization``.
    """
    model_pricing = MODEL_PRICING.get(model)
    if model_pricing is None:
        raise KeyError(f"No pricing entry for model {model!r}.")
    inference_vram_gigabytes = (
        model_pricing["gpu_vram_gigabytes"] * gpu_memory_utilization
    )
    training_headroom_gigabytes = model_pricing["gpu_vram_gigabytes"] * (
        1.0 - gpu_memory_utilization
    )
    return {
        "model": model,
        "gpu_configuration": model_pricing["gpu_configuration"],
        "gpu_memory_utilization": gpu_memory_utilization,
        "inference_vram_gigabytes": round(inference_vram_gigabytes, 2),
        "training_headroom_vram_gigabytes": round(training_headroom_gigabytes, 2),
        "concurrent_adapters_servable": MAX_LORAS,
        "concurrent_training_jobs": TRAINING_CONCURRENCY,
    }


def break_even(monthly_fixed_cost: float, price_per_user: float) -> Dict[str, Any]:
    """Users needed per month for revenue to cover ``monthly_fixed_cost``."""
    if price_per_user <= 0:
        return {
            "monthly_fixed_cost_usd": monthly_fixed_cost,
            "price_per_user_usd": price_per_user,
            "break_even_users": None,
            "note": "price_per_user must be positive to break even.",
        }
    return {
        "monthly_fixed_cost_usd": round(monthly_fixed_cost, 2),
        "price_per_user_usd": price_per_user,
        "break_even_users": math.ceil(monthly_fixed_cost / price_per_user),
    }
