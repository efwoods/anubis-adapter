"""In-process registry for background adapter-training jobs.

Adapted from the main Anubis API's media-job registry
(``scaffold/media_job.py``), flattened: adapter training has no batch
master/child split — one ``TrainingJob`` per ``/train_adapter`` request. The
same mechanics carry over: an append-only progress-event buffer with an
``asyncio.Event`` so SSE subscribers replay from index 0 then wait for new
appends (late joiners and multiple concurrent subscribers both work), TTL
cleanup of finished jobs, and a cooperative cancel flag the trainer checks at
every step boundary.

NOTE: the registry is per-process. A multi-worker deployment needs a shared
store (Redis) instead.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Keep finished jobs around briefly so a late/reconnecting progress client can
# still read the final result, then drop them so the registry doesn't grow
# unbounded.
_FINISHED_TTL_SECONDS = 30 * 60
_MAX_JOBS = 1000


@dataclass
class TrainingJob:
    """A single background adapter-training job and its progress buffer."""

    job_id: str
    user_id: str
    assistant_id: str
    model: str
    status: str = "queued"  # queued | running | completed | error | cancelled
    created_at: float = field(default_factory=time.time)
    # Epoch seconds when training actually began (status -> running) and when
    # the job finished (completion, error, or cancellation), set by the runner
    # and finish_job.
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Wall-clock seconds from training start to completion/error, set by finish_job.
    duration_seconds: Optional[float] = None
    # Live training telemetry, updated from the trainer callback.
    current_step: int = 0
    total_steps: Optional[int] = None
    latest_loss: Optional[float] = None
    latest_reward: Optional[float] = None
    # Where the finished adapter landed in S3 (set on success).
    adapter_s3_prefix: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    # Set when a cancel was requested; the trainer stops cooperatively at the
    # next step boundary.
    cancelled: bool = False
    # Append-only history of progress payloads. Subscribers replay from index 0,
    # then wait on ``_updated`` for new appends — this supports late joiners and
    # multiple concurrent subscribers.
    events: List[Dict[str, Any]] = field(default_factory=list)
    _updated: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None

    def snapshot(self) -> Dict[str, Any]:
        """The status dict served by ``GET /adapter_training_status``."""
        return {
            "job_id": self.job_id,
            "assistant_id": self.assistant_id,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "latest_loss": self.latest_loss,
            "latest_reward": self.latest_reward,
            "adapter_s3_prefix": self.adapter_s3_prefix,
            "result": self.result,
            "error": self.error,
            "cancelled": self.cancelled,
        }


def create_training_job(
    registry: Dict[str, TrainingJob],
    *,
    user_id: str,
    assistant_id: str,
    model: str,
) -> TrainingJob:
    """Register and return a new queued training job."""
    _cleanup(registry)
    job = TrainingJob(
        job_id=str(uuid4()),
        user_id=user_id,
        assistant_id=assistant_id,
        model=model,
    )
    registry[job.job_id] = job
    return job


def get_job(registry: Dict[str, TrainingJob], job_id: str) -> Optional[TrainingJob]:
    """Return the job for ``job_id``, or ``None`` if unknown/expired."""
    return registry.get(job_id)


def add_event(job: TrainingJob, payload: Dict[str, Any]) -> None:
    """Append a progress payload and wake any waiting subscribers."""
    job.events.append(payload)
    job._updated.set()


def finish_job(
    job: TrainingJob,
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    cancelled: bool = False,
) -> None:
    """Mark the job completed, errored, or cancelled and wake subscribers.

    Idempotent: a job already ``done`` is left untouched.
    """
    if job.done.is_set():
        return
    job.finished_at = time.time()
    # Measure from when training actually started; fall back to creation time
    # if the job errored before it began running.
    job.duration_seconds = round(
        job.finished_at - (job.started_at or job.created_at), 3
    )
    if cancelled:
        job.status = "cancelled"
        job.cancelled = True
        if result is not None:
            job.result = result
    elif error is not None:
        job.status = "error"
        job.error = error
    else:
        job.status = "completed"
        job.result = result
    job._updated.set()
    job.done.set()


def request_cancel(job: TrainingJob) -> None:
    """Flag ``job`` cancelled; cancel the asyncio task only if still queued.

    A RUNNING trainer must not be torn down mid-step (the executor thread holds
    GPU state) — it observes ``job.cancelled`` at the next step boundary and
    stops cooperatively instead.
    """
    job.cancelled = True
    if job.status == "queued" and job.task is not None and not job.task.done():
        job.task.cancel()


def _cleanup(registry: Dict[str, TrainingJob]) -> None:
    """Drop finished jobs past their TTL; trim oldest if the registry is too large."""
    now = time.time()
    stale_job_ids = [
        job_id
        for job_id, job in registry.items()
        if job.finished_at is not None
        and now - job.finished_at > _FINISHED_TTL_SECONDS
    ]
    for job_id in stale_job_ids:
        registry.pop(job_id, None)

    if len(registry) > _MAX_JOBS:
        oldest_jobs = sorted(registry.values(), key=lambda job: job.created_at)
        for job in oldest_jobs[: len(registry) - _MAX_JOBS]:
            registry.pop(job.job_id, None)
