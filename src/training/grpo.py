"""Background LoRA adapter training with TRL GRPO.

``run_training`` is the coroutine a ``/train_adapter`` request spawns as an
``asyncio.Task``. It stages progress events onto the job's buffer
(``fetching_dataset`` is emitted by the route; here: ``building_dataset``,
``building_reference_profile``, ``loading_base_model``, per-step
``training_step``, ``saving_adapter``, ``uploading_adapter``), runs the
blocking transformers/TRL work in an executor thread, and uploads the finished
adapter to S3 under ``adapters/{user_id}/{assistant_id}/{model}/``.

Training uses a plain ``AutoModelForCausalLM`` load (no bitsandbytes
quantization) with a PEFT LoRA adapter; the base weights are never merged —
PEFT saves only ``adapter_model.safetensors`` + ``adapter_config.json``.
Generation during GRPO is transformers-native (no ``use_vllm``) so training
never contends with the serving engine's vLLM GPU allocation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any, Dict, List

from src.adapters.storage import save_adapter
from src.api.training_jobs import TrainingJob, add_event, finish_job
from src.config import ADAPTER_LOCAL_DIRECTORY
from src.metrics.cost import training_cost
from src.metrics.prometheus import (
    ADAPTER_SIZE_BYTES,
    ADAPTER_TRAINING_DURATION_SECONDS,
    ADAPTER_TRAINING_JOBS_ACTIVE,
)
from src.training.dataset import build_grpo_dataset
from src.training.rewards import build_reference_style_profile

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_PARAMETERS: Dict[str, Any] = {
    "learning_rate": 1e-5,
    "num_train_epochs": 1.0,
    "lora_rank": 16,
    "lora_alpha": 32,
    "num_generations": 4,
    "max_completion_length": 256,
}


def _training_work_directory(job_id: str) -> str:
    """Scratch directory for one job's checkpoints and saved adapter."""
    return os.path.join(ADAPTER_LOCAL_DIRECTORY, "training_jobs", job_id)


async def run_training(
    job: TrainingJob,
    request_parameters: Dict[str, Any],
    fetched_datasets: List[Dict[str, Any]],
    training_semaphore: asyncio.Semaphore,
    s3_client: Any,
) -> None:
    """Run one adapter-training job end to end. Never raises: every outcome
    (success, error, cancellation) lands on the job via ``finish_job``."""
    parameters = {**DEFAULT_TRAINING_PARAMETERS, **request_parameters}
    work_directory = _training_work_directory(job.job_id)
    try:
        add_event(job, {"type": "training_progress", "stage": "building_dataset"})
        training_dataset, reference_completions = await asyncio.to_thread(
            build_grpo_dataset, fetched_datasets
        )
        if len(training_dataset) == 0 or not reference_completions:
            finish_job(
                job,
                error=(
                    "The stored datasets produced no usable prompt-completion "
                    "training examples."
                ),
            )
            return

        add_event(
            job,
            {
                "type": "training_progress",
                "stage": "building_reference_profile",
                "training_examples": len(training_dataset),
                "reference_completions": len(reference_completions),
            },
        )
        reference_style_profile = await asyncio.to_thread(
            build_reference_style_profile, reference_completions
        )

        async with training_semaphore:
            if job.cancelled:
                finish_job(
                    job,
                    cancelled=True,
                    result={"message": "Training cancelled before it started."},
                )
                return

            import time as time_module

            job.started_at = time_module.time()
            job.status = "running"
            ADAPTER_TRAINING_JOBS_ACTIVE.inc()
            event_loop = asyncio.get_running_loop()
            try:
                local_adapter_directory = await event_loop.run_in_executor(
                    None,
                    _run_training_blocking,
                    job,
                    parameters,
                    training_dataset,
                    reference_style_profile,
                    event_loop,
                    work_directory,
                )
            finally:
                ADAPTER_TRAINING_JOBS_ACTIVE.dec()

        if job.cancelled:
            finish_job(
                job,
                cancelled=True,
                result={"message": "Training cancelled mid-run; adapter discarded."},
            )
            return

        add_event(job, {"type": "training_progress", "stage": "uploading_adapter"})
        adapter_s3_prefix_value, adapter_size_bytes = await asyncio.to_thread(
            save_adapter,
            s3_client,
            local_adapter_directory,
            job.user_id,
            job.assistant_id,
            job.model,
        )
        job.adapter_s3_prefix = adapter_s3_prefix_value

        import time as time_module

        training_duration_seconds = time_module.time() - (job.started_at or job.created_at)
        ADAPTER_TRAINING_DURATION_SECONDS.observe(training_duration_seconds)
        ADAPTER_SIZE_BYTES.labels(
            user_id=job.user_id, assistant_id=job.assistant_id
        ).set(adapter_size_bytes)

        try:
            training_cost_report = training_cost(
                job.model, training_duration_seconds / 3600.0
            )
        except KeyError:
            training_cost_report = None

        from src.metrics.cost import adapter_storage_cost_per_month

        finish_job(
            job,
            result={
                "message": "Adapter trained and uploaded successfully.",
                "adapter_s3_prefix": adapter_s3_prefix_value,
                "adapter_size_bytes": adapter_size_bytes,
                "training_duration_seconds": round(training_duration_seconds, 3),
                "training_examples": len(training_dataset),
                "reference_completions": len(reference_completions),
                "training_cost": training_cost_report,
                "adapter_storage_cost_per_month_usd": round(
                    adapter_storage_cost_per_month(adapter_size_bytes), 6
                ),
                "signature_key_phrases": reference_style_profile["key_phrases"][:10],
            },
        )
    except asyncio.CancelledError:
        finish_job(
            job,
            cancelled=True,
            result={"message": "Training job task cancelled."},
        )
        raise
    except Exception as exc:  # noqa: BLE001 - surface every failure via the job
        logger.exception("Adapter training job %s failed: %s", job.job_id, exc)
        finish_job(job, error=str(exc))
    finally:
        shutil.rmtree(work_directory, ignore_errors=True)


def _run_training_blocking(
    job: TrainingJob,
    parameters: Dict[str, Any],
    training_dataset: Any,
    reference_style_profile: Dict[str, Any],
    event_loop: asyncio.AbstractEventLoop,
    work_directory: str,
) -> str:
    """The blocking transformers/PEFT/TRL portion, run in an executor thread.

    Progress events are marshalled back onto the event loop with
    ``call_soon_threadsafe`` (the job's ``asyncio.Event`` must only be set from
    the loop thread). Returns the local directory the adapter was saved to.
    """
    import torch
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
    )
    from trl import GRPOConfig, GRPOTrainer

    from src.training.rewards import build_reward_functions

    def emit_event(payload: Dict[str, Any]) -> None:
        event_loop.call_soon_threadsafe(add_event, job, payload)

    emit_event(
        {
            "type": "training_progress",
            "stage": "loading_base_model",
            "model": job.model,
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(job.model)
    # Plain bf16 load — no quantization_config; the LoRA adapter carries every
    # trained weight and the base model is never merged or modified.
    model = AutoModelForCausalLM.from_pretrained(
        job.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    lora_configuration = LoraConfig(
        r=int(parameters["lora_rank"]),
        lora_alpha=int(parameters["lora_alpha"]),
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    num_generations = int(parameters["num_generations"])
    grpo_configuration = GRPOConfig(
        output_dir=os.path.join(work_directory, "checkpoints"),
        learning_rate=float(parameters["learning_rate"]),
        num_train_epochs=float(parameters["num_train_epochs"]),
        num_generations=num_generations,
        max_completion_length=int(parameters["max_completion_length"]),
        # The effective batch must be divisible by num_generations; one prompt
        # group per device step keeps memory predictable on the shared GPU.
        per_device_train_batch_size=num_generations,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
    )

    class TrainingProgressCallback(TrainerCallback):
        """Push per-step telemetry into the job buffer; honor cooperative cancel."""

        def on_train_begin(self, args, state, control, **kwargs):
            job.total_steps = int(state.max_steps or 0) or None

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            job.current_step = int(state.global_step)
            if "loss" in logs:
                job.latest_loss = float(logs["loss"])
            if "reward" in logs:
                job.latest_reward = float(logs["reward"])
            per_feature_reward_means = {
                key.split("/")[1]: value
                for key, value in logs.items()
                if key.startswith("rewards/") and key.endswith("/mean")
            }
            emit_event(
                {
                    "type": "training_progress",
                    "stage": "training_step",
                    "step": job.current_step,
                    "total_steps": job.total_steps,
                    "loss": job.latest_loss,
                    "reward": job.latest_reward,
                    "per_feature_reward_means": per_feature_reward_means or None,
                }
            )

        def on_step_end(self, args, state, control, **kwargs):
            if job.cancelled:
                control.should_training_stop = True

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=build_reward_functions(reference_style_profile),
        args=grpo_configuration,
        train_dataset=training_dataset,
        processing_class=tokenizer,
        peft_config=lora_configuration,
        callbacks=[TrainingProgressCallback()],
    )
    try:
        trainer.train()

        emit_event({"type": "training_progress", "stage": "saving_adapter"})
        local_adapter_directory = os.path.join(work_directory, "adapter")
        # Adapter only: PEFT writes adapter_model.safetensors +
        # adapter_config.json; base weights are never merged.
        trainer.save_model(local_adapter_directory)
        return local_adapter_directory
    finally:
        # Release GPU memory promptly so the serving engine and the next
        # queued training job get the head-room back.
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
