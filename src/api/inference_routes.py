"""SSE ``/message`` endpoints: per-user, per-assistant adapter inference.

Mirrors the main Anubis API's message surface (``scaffold/webapp.py``): form
fields, SSE ``assistant_token`` events per token, and the terminal ``done``
event carrying ``content`` / ``thread_id`` / ``request_id`` /
``total_response_time_ms``. The graph behind the scaffold is replaced by
direct vLLM generation with the caller's LoRA adapter attached when one exists
in S3 (unadapted base-model inference otherwise — the demo's "before
training" path).

Threads are held in an in-process store (``app.state.threads``) so multi-turn
context works for the demo; production would persist them in Postgres/Redis.
"""

from __future__ import annotations

import json
import logging
from time import time_ns
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import BASE_MODEL
from src.inference.engine import attach_adapter, generate_stream
from src.metrics.prometheus import ADAPTER_INFERENCE_TOKENS_TOTAL
from src.security.auth import get_current_user

logger = logging.getLogger(__name__)

inference_route = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _thread_store(request: Request) -> Dict[str, Dict[str, Any]]:
    return request.app.state.threads


def _new_thread_state(
    user_id: str,
    assistant_id: Optional[str],
    conversation_title: Optional[str],
    your_name: Optional[str],
    your_description: Optional[str],
) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    if your_name or your_description:
        user_introduction = (
            f"You are speaking with {your_name or 'a user'}."
            + (f" About them: {your_description}" if your_description else "")
        )
        messages.append({"role": "system", "content": user_introduction})
    return {
        "user_id": user_id,
        "assistant_id": assistant_id,
        "conversation_title": conversation_title,
        "messages": messages,
        "closed": False,
    }


def _get_owned_thread(
    request: Request, thread_id: str, user_id: str
) -> Dict[str, Any]:
    thread_state = _thread_store(request).get(thread_id)
    if thread_state is None:
        raise HTTPException(status_code=404, detail="Unknown thread_id.")
    if thread_state["user_id"] != user_id:
        raise HTTPException(
            status_code=403, detail="This conversation belongs to another user."
        )
    return thread_state


async def _resolve_lora_request(
    request: Request, user_id: str, assistant_id: Optional[str]
) -> Optional[Any]:
    """The caller's adapter as a ``LoRARequest``, or ``None`` (base model)."""
    if assistant_id is None:
        return None
    lora_request = await attach_adapter(
        request.app.state.s3_client, user_id, assistant_id, BASE_MODEL
    )
    if lora_request is not None:
        logger.info(
            "Serving with LoRARequest %s for assistant %s",
            lora_request.lora_name,
            assistant_id,
        )
    return lora_request


async def _generate_and_record(
    thread_state: Dict[str, Any], lora_request: Optional[Any]
) -> AsyncIterator[str]:
    """Stream token deltas, recording per-token metrics as they are produced."""
    adapter_label = lora_request.lora_name if lora_request is not None else "base"
    async for token_text in generate_stream(
        thread_state["messages"], lora_request
    ):
        ADAPTER_INFERENCE_TOKENS_TOTAL.labels(
            model=BASE_MODEL, adapter=adapter_label
        ).inc()
        yield token_text


async def _message_sse(
    thread_state: Dict[str, Any],
    lora_request: Optional[Any],
    *,
    thread_id: str,
    request_id: str,
    start_time_ns: int,
    include_metrics: bool,
) -> AsyncIterator[str]:
    """SSE stream: ``assistant_token`` per token, then the terminal ``done``."""
    accumulated_token_texts: List[str] = []
    async for token_text in _generate_and_record(thread_state, lora_request):
        accumulated_token_texts.append(token_text)
        yield f"data: {json.dumps({'type': 'assistant_token', 'text': token_text})}\n\n"

    content = "".join(accumulated_token_texts)
    thread_state["messages"].append({"role": "assistant", "content": content})

    done_event: Dict[str, Any] = {
        "type": "done",
        "content": content,
        "thread_id": thread_id,
        "request_id": request_id,
        "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
    }
    if include_metrics:
        done_event["response_metadata"] = {
            "model": BASE_MODEL,
            "adapter": lora_request.lora_name if lora_request is not None else None,
            "completion_token_estimate": len(accumulated_token_texts),
        }
    yield f"data: {json.dumps(done_event, default=str)}\n\n"


async def _handle_message(
    request: Request,
    assistant_id: Optional[str],
    *,
    message: str,
    your_name: Optional[str],
    your_description: Optional[str],
    conversation_title: Optional[str],
    thread_id: Optional[str],
    stream: bool,
    include_metrics: bool,
    current_user: dict,
):
    start_time_ns = time_ns()
    user_id = current_user["identities"][0]["user_id"]

    if not (message or "").strip():
        raise HTTPException(status_code=400, detail="message must not be empty.")

    if thread_id:
        thread_state = _get_owned_thread(request, thread_id, user_id)
        if thread_state["closed"]:
            raise HTTPException(
                status_code=409, detail="This conversation has been closed."
            )
    else:
        thread_id = str(uuid4())
        thread_state = _new_thread_state(
            user_id,
            assistant_id,
            conversation_title if conversation_title else thread_id,
            your_name,
            your_description,
        )
        _thread_store(request)[thread_id] = thread_state

    thread_state["messages"].append({"role": "user", "content": message})

    lora_request = await _resolve_lora_request(request, user_id, assistant_id)
    request_id = request.state.request_id

    if stream:
        return StreamingResponse(
            _message_sse(
                thread_state,
                lora_request,
                thread_id=thread_id,
                request_id=request_id,
                start_time_ns=start_time_ns,
                include_metrics=include_metrics,
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    accumulated_token_texts: List[str] = []
    async for token_text in _generate_and_record(thread_state, lora_request):
        accumulated_token_texts.append(token_text)
    content = "".join(accumulated_token_texts)
    thread_state["messages"].append({"role": "assistant", "content": content})

    response_data: Dict[str, Any] = {
        "content": content,
        "thread_id": thread_id,
        "request_id": request_id,
        "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
    }
    if include_metrics:
        response_data["response_metadata"] = {
            "model": BASE_MODEL,
            "adapter": lora_request.lora_name if lora_request is not None else None,
            "completion_token_estimate": len(accumulated_token_texts),
        }
    return JSONResponse(response_data, status_code=200)


@inference_route.post("/message")
async def message(
    request: Request,
    message: str = Form(""),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    conversation_title: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    stream: bool = Form(True),
    include_metrics: bool = Form(True),
    current_user: dict = Depends(get_current_user),
):
    """Unadapted base-model inference (no assistant selected, no adapter)."""
    return await _handle_message(
        request,
        None,
        message=message,
        your_name=your_name,
        your_description=your_description,
        conversation_title=conversation_title,
        thread_id=thread_id,
        stream=stream,
        include_metrics=include_metrics,
        current_user=current_user,
    )


@inference_route.post("/message/{assistant_id}")
async def message_avatar(
    request: Request,
    assistant_id: str,
    message: str = Form(""),
    your_name: Optional[str] = Form(None),
    your_description: Optional[str] = Form(None),
    conversation_title: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    stream: bool = Form(True),
    include_metrics: bool = Form(True),
    current_user: dict = Depends(get_current_user),
):
    """Message one assistant; runs with the caller's trained LoRA adapter when
    one exists in S3, and unadapted base-model inference otherwise."""
    return await _handle_message(
        request,
        assistant_id,
        message=message,
        your_name=your_name,
        your_description=your_description,
        conversation_title=conversation_title,
        thread_id=thread_id,
        stream=stream,
        include_metrics=include_metrics,
        current_user=current_user,
    )


@inference_route.post("/message/{assistant_id}/resume")
async def resume_avatar_message(
    request: Request,
    assistant_id: str,
    thread_id: str = Form(...),
    decision: str = Form("apply"),
    include_metrics: bool = Form(True),
    current_user: dict = Depends(get_current_user),
):
    """Resume a paused conversation thread.

    ``decision`` is ``apply`` | ``cancel`` (older clients' ``approve`` /
    ``reject`` are accepted as aliases — scaffold parity). ``apply``
    regenerates the assistant's latest reply and streams the continuation over
    SSE; ``cancel`` marks the thread closed and emits a terminal ``done``.

    NOTE: this service has no LangGraph interrupt loop — resume here is plain
    thread continuation over the in-process thread store, shaped to match the
    main Anubis API's ``/message/{assistant_id}/resume`` contract.
    """
    start_time_ns = time_ns()
    user_id = current_user["identities"][0]["user_id"]

    decision_aliases = {"approve": "apply", "reject": "cancel"}
    raw_decision = (decision or "apply").strip().lower()
    decision_value = decision_aliases.get(raw_decision, raw_decision)
    if decision_value not in ("apply", "cancel"):
        raise HTTPException(status_code=400, detail="decision must be apply or cancel.")

    thread_state = _get_owned_thread(request, thread_id, user_id)
    request_id = request.state.request_id

    if decision_value == "cancel":
        thread_state["closed"] = True

        async def cancelled_sse() -> AsyncIterator[str]:
            done_event = {
                "type": "done",
                "content": "",
                "thread_id": thread_id,
                "request_id": request_id,
                "total_response_time_ms": (time_ns() - start_time_ns) // 1_000_000,
                "message": "Conversation closed.",
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        return StreamingResponse(
            cancelled_sse(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    if thread_state["closed"]:
        raise HTTPException(status_code=409, detail="This conversation has been closed.")
    if not thread_state["messages"]:
        raise HTTPException(
            status_code=400, detail="Nothing to resume: the thread has no messages."
        )

    # ``apply`` regenerates the latest assistant reply: drop it (when present)
    # and generate a fresh continuation from the remaining context.
    if thread_state["messages"][-1]["role"] == "assistant":
        thread_state["messages"].pop()

    lora_request = await _resolve_lora_request(request, user_id, assistant_id)
    return StreamingResponse(
        _message_sse(
            thread_state,
            lora_request,
            thread_id=thread_id,
            request_id=request_id,
            start_time_ns=start_time_ns,
            include_metrics=include_metrics,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
