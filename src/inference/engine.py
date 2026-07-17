"""vLLM serving engine: lazy base-model load + per-user LoRA attachment.

One ``AsyncLLMEngine`` serves every request, created lazily on the first
inference call so the app boots (and ``/health`` responds) without touching
the GPU. LoRA support is enabled at engine construction; adapters are attached
per request via ``LoRARequest`` — vLLM keeps up to ``MAX_LORAS`` adapters
resident and swaps others in on demand, which is what serves many concurrent
users' adapters from one base model.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from src.adapters.storage import adapter_local_directory, download_adapter
from src.config import (
    BASE_MODEL,
    GPU_MEMORY_UTILIZATION,
    MAX_LORA_RANK,
    MAX_LORAS,
    MAX_MODEL_LEN,
)

logger = logging.getLogger(__name__)

_engine: Any = None
_engine_lock = asyncio.Lock()

# lora_name -> LoRARequest for every adapter attached this process lifetime.
_lora_requests_by_name: Dict[str, Any] = {}
# Stable positive integer id per lora_name (vLLM requires a unique int per adapter).
_lora_integer_ids_by_name: Dict[str, int] = {}


def _lora_name(user_id: str, assistant_id: str) -> str:
    return f"{user_id}--{assistant_id}"


def _stable_lora_integer_id(lora_name: str) -> int:
    if lora_name not in _lora_integer_ids_by_name:
        _lora_integer_ids_by_name[lora_name] = len(_lora_integer_ids_by_name) + 1
    return _lora_integer_ids_by_name[lora_name]


async def load_basemodel() -> Any:
    """Lazily create (once) and return the shared ``AsyncLLMEngine``."""
    global _engine
    if _engine is not None:
        return _engine
    async with _engine_lock:
        if _engine is None:
            # vLLM is a heavy import and allocates GPU memory at construction;
            # both are deferred to the first real inference call.
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            logger.info(
                "Loading base model %s (gpu_memory_utilization=%s, max_loras=%s)",
                BASE_MODEL,
                GPU_MEMORY_UTILIZATION,
                MAX_LORAS,
            )
            engine_args = AsyncEngineArgs(
                model=BASE_MODEL,
                enable_lora=True,
                max_loras=MAX_LORAS,
                max_lora_rank=MAX_LORA_RANK,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_model_len=MAX_MODEL_LEN,
            )
            _engine = AsyncLLMEngine.from_engine_args(engine_args)
    return _engine


async def attach_adapter(
    s3_client: Any, user_id: str, assistant_id: str, model: str
) -> Optional[Any]:
    """Resolve the caller's adapter to a ``LoRARequest``.

    Downloads the adapter from S3 on first use (skipping files already
    present). Returns ``None`` when no adapter exists for the
    ``(user_id, assistant_id, model)`` triple — the caller then serves
    unadapted base-model inference (the demo's "before training" path).
    """
    lora_name = _lora_name(user_id, assistant_id)
    existing_lora_request = _lora_requests_by_name.get(lora_name)
    if existing_lora_request is not None:
        return existing_lora_request

    local_directory = await asyncio.to_thread(
        download_adapter, s3_client, user_id, assistant_id, model
    )
    if local_directory is None:
        return None

    from vllm.lora.request import LoRARequest

    lora_request = LoRARequest(
        lora_name=lora_name,
        lora_int_id=_stable_lora_integer_id(lora_name),
        lora_path=local_directory,
    )
    _lora_requests_by_name[lora_name] = lora_request
    logger.info(
        "Attached adapter %s (lora_int_id=%s) from %s",
        lora_name,
        lora_request.lora_int_id,
        local_directory,
    )
    return lora_request


def remove_adapter(user_id: str, assistant_id: str, model: str) -> bool:
    """Detach an adapter: drop its ``LoRARequest`` and delete the local files.

    Returns ``True`` when an attachment existed. The S3 copy is untouched (use
    :func:`src.adapters.storage.delete_adapter` for that).
    """
    lora_name = _lora_name(user_id, assistant_id)
    removed_lora_request = _lora_requests_by_name.pop(lora_name, None)
    shutil.rmtree(
        adapter_local_directory(user_id, assistant_id, model), ignore_errors=True
    )
    return removed_lora_request is not None


async def generate_stream(
    messages: List[Dict[str, str]],
    lora_request: Optional[Any] = None,
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> AsyncIterator[str]:
    """Stream token-text deltas for a chat ``messages`` list.

    Applies the base model's chat template, then iterates vLLM's async
    generator, yielding only the newly generated text of each engine step. The
    optional ``lora_request`` runs the request through that user's adapter.
    """
    engine = await load_basemodel()
    tokenizer = await engine.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    request_id = str(uuid4())

    previous_text = ""
    async for request_output in engine.generate(
        prompt, sampling_params, request_id, lora_request=lora_request
    ):
        generated_text = request_output.outputs[0].text
        text_delta = generated_text[len(previous_text) :]
        previous_text = generated_text
        if text_delta:
            yield text_delta
