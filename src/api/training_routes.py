"""Adapter-training job endpoints, shaped on the Anubis media-job surface.

``POST /train_adapter`` starts a background GRPO LoRA training job over the
caller's stored prompt-completion datasets;
``GET /adapter_training_status`` / ``GET /adapter_training_progress`` (SSE) /
``POST /cancel_adapter_training_job`` monitor and control it — the same
start/status/progress/cancel shape as the main Anubis API's media jobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.training_jobs import (
    TrainingJob,
    add_event,
    create_training_job,
    get_job,
    request_cancel,
)
from src.config import BASE_MODEL
from src.security.auth import get_current_user
from src.training.dataset import fetch_adapter_datasets
from src.training.grpo import run_training

logger = logging.getLogger(__name__)

training_route = APIRouter()


def _get_owned_job(request: Request, job_id: str, user_id: str) -> TrainingJob:
    job = get_job(request.app.state.training_jobs, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id.")
    if job.user_id != user_id:
        raise HTTPException(
            status_code=403, detail="This training job belongs to another user."
        )
    return job


@training_route.post("/train_adapter", status_code=202)
async def train_adapter(
    request: Request,
    assistant_id: str = Form(...),
    model: Optional[str] = Form(None),
    learning_rate: Optional[float] = Form(None),
    num_train_epochs: Optional[float] = Form(None),
    lora_rank: Optional[int] = Form(None),
    lora_alpha: Optional[int] = Form(None),
    num_generations: Optional[int] = Form(None),
    max_completion_length: Optional[int] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """Start a background LoRA GRPO training job for one assistant.

    Searches the Anubis store for the caller's prompt-completion datasets
    (``q_and_a_adapter`` + ``multi_turn_dataset_adapter``); 404 when none
    exist. Returns ``202 {"job_id", "status"}`` immediately; follow progress
    via ``GET /adapter_training_progress?job_id=...``.
    """
    user_id = current_user["identities"][0]["user_id"]

    store_pool = request.app.state.store_pool
    if store_pool is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The Anubis store database is not configured "
                "(ASYNC_POSTGRES_STORE_URI is unset)."
            ),
        )

    fetched_datasets = await fetch_adapter_datasets(store_pool, user_id, assistant_id)
    if not any(
        fetched_dataset["dataset_type"]
        in ("q_and_a_adapter", "multi_turn_dataset_adapter")
        for fetched_dataset in fetched_datasets
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "No adapter training datasets found for this assistant. Upload "
                "media through the Anubis API first."
            ),
        )

    job = create_training_job(
        request.app.state.training_jobs,
        user_id=user_id,
        assistant_id=assistant_id,
        model=model or BASE_MODEL,
    )
    add_event(
        job,
        {
            "type": "training_progress",
            "stage": "fetching_dataset",
            "datasets_found": len(fetched_datasets),
        },
    )

    request_parameters = {
        parameter_name: parameter_value
        for parameter_name, parameter_value in {
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "num_generations": num_generations,
            "max_completion_length": max_completion_length,
        }.items()
        if parameter_value is not None
    }

    job.task = asyncio.create_task(
        run_training(
            job,
            request_parameters,
            fetched_datasets,
            request.app.state.training_semaphore,
            request.app.state.s3_client,
        )
    )

    return JSONResponse({"job_id": job.job_id, "status": job.status}, status_code=202)


@training_route.get("/adapter_training_status")
async def adapter_training_status(
    request: Request,
    job_id: str = Query(...),
    current_user: dict = Depends(get_current_user),
):
    """Snapshot of one training job (status, step, loss/reward, S3 prefix)."""
    user_id = current_user["identities"][0]["user_id"]
    job = _get_owned_job(request, job_id, user_id)
    return job.snapshot()


@training_route.get("/adapter_training_progress")
async def adapter_training_progress(
    request: Request,
    job_id: str = Query(...),
    current_user: dict = Depends(get_current_user),
):
    """SSE progress stream for one training job.

    Replays the job's full event history from index 0 (late joiners see every
    stage), then streams live events until the job settles; the terminal event
    carries the final status snapshot.
    """
    user_id = current_user["identities"][0]["user_id"]
    job = _get_owned_job(request, job_id, user_id)

    async def progress_sse() -> AsyncIterator[str]:
        next_event_index = 0
        while True:
            # Clear BEFORE draining: an event appended between the drain and
            # the wait re-sets the flag, so wait() returns immediately instead
            # of stalling until the following event.
            job._updated.clear()
            while next_event_index < len(job.events):
                payload = job.events[next_event_index]
                next_event_index += 1
                yield f"data: {json.dumps(payload, default=str)}\n\n"

            if job.done.is_set() and next_event_index >= len(job.events):
                terminal_event = {"type": "training_done", **job.snapshot()}
                yield f"data: {json.dumps(terminal_event, default=str)}\n\n"
                return

            await job._updated.wait()

    return StreamingResponse(
        progress_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@training_route.post("/cancel_adapter_training_job")
async def cancel_adapter_training_job(
    request: Request,
    job_id: str = Query(...),
    current_user: dict = Depends(get_current_user),
):
    """Request cancellation of one training job.

    A queued job is cancelled immediately; a running trainer stops
    cooperatively at the next training step.
    """
    user_id = current_user["identities"][0]["user_id"]
    job = _get_owned_job(request, job_id, user_id)
    if job.done.is_set():
        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": "Job already finished.",
        }
    request_cancel(job)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": "Cancellation requested; the job stops at the next training step.",
    }
