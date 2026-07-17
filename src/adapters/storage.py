"""S3 storage for trained LoRA adapters and base-model weights.

Single bucket (``BUCKET``), two prefixes:

* ``MODELS_PREFIX`` (``models/``) — base-model weights, uploaded from the
  Hugging Face Hub via ``snapshot_download`` then a directory walk.
* ``ADAPTERS_PREFIX`` (``adapters/{user_id}/{assistant_id}/{model}/``) —
  trained adapters, namespaced per user, per assistant, per base model. This
  layout is what makes multi-user / multi-assistant adapter selection at
  inference time work; keep it consistent with the main Anubis repo.

Every function here does blocking boto3/filesystem work; callers on async
paths wrap them in ``asyncio.to_thread``. The boto3 client is created once in
the FastAPI lifespan (``app.state.s3_client``) and passed in.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, List, Optional, Tuple

from src.config import (
    ADAPTER_LOCAL_DIRECTORY,
    ADAPTERS_PREFIX,
    BUCKET,
    MODELS_PREFIX,
)

logger = logging.getLogger(__name__)


def sanitize_model_name(model: str) -> str:
    """Model identifier as a clean single S3 key segment (slashes -> ``--``)."""
    return model.replace("/", "--")


def adapter_s3_prefix(user_id: str, assistant_id: str, model: str) -> str:
    """The S3 prefix one adapter's files live under."""
    return ADAPTERS_PREFIX.format(
        user_id=user_id,
        assistant_id=assistant_id,
        model=sanitize_model_name(model),
    )


def adapter_local_directory(user_id: str, assistant_id: str, model: str) -> str:
    """The local directory a downloaded adapter is materialized into."""
    return os.path.join(
        ADAPTER_LOCAL_DIRECTORY, user_id, assistant_id, sanitize_model_name(model)
    )


def directory_size_bytes(directory_path: str) -> int:
    """Total size in bytes of every file under ``directory_path``."""
    total_bytes = 0
    for parent_directory, _subdirectories, filenames in os.walk(directory_path):
        for filename in filenames:
            total_bytes += os.path.getsize(os.path.join(parent_directory, filename))
    return total_bytes


def save_adapter(
    s3_client: Any,
    local_directory: str,
    user_id: str,
    assistant_id: str,
    model: str,
) -> Tuple[str, int]:
    """Upload a trained adapter directory to S3, preserving relative paths.

    Returns ``(s3_prefix, total_uploaded_bytes)``.
    """
    s3_prefix = adapter_s3_prefix(user_id, assistant_id, model)
    total_uploaded_bytes = 0
    for parent_directory, _subdirectories, filenames in os.walk(local_directory):
        for filename in filenames:
            local_file_path = os.path.join(parent_directory, filename)
            relative_path = os.path.relpath(local_file_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            logger.info("Uploading %s -> s3://%s/%s", relative_path, BUCKET, s3_key)
            s3_client.upload_file(local_file_path, BUCKET, s3_key)
            total_uploaded_bytes += os.path.getsize(local_file_path)
    return s3_prefix, total_uploaded_bytes


def download_adapter(
    s3_client: Any, user_id: str, assistant_id: str, model: str
) -> Optional[str]:
    """Download one adapter's files from S3 into the local adapter directory.

    Files already present locally are not re-downloaded. Returns the local
    directory path, or ``None`` when no adapter exists under the prefix (the
    caller then serves unadapted base-model inference).
    """
    s3_prefix = adapter_s3_prefix(user_id, assistant_id, model)
    local_directory = adapter_local_directory(user_id, assistant_id, model)

    paginator = s3_client.get_paginator("list_objects_v2")
    object_keys: List[str] = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for s3_object in page.get("Contents", []):
            object_keys.append(s3_object["Key"])
    if not object_keys:
        return None

    os.makedirs(local_directory, exist_ok=True)
    for s3_key in object_keys:
        relative_path = os.path.relpath(s3_key, s3_prefix)
        local_file_path = os.path.join(local_directory, relative_path)
        if os.path.exists(local_file_path):
            continue
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        logger.info("Downloading s3://%s/%s -> %s", BUCKET, s3_key, local_file_path)
        s3_client.download_file(BUCKET, s3_key, local_file_path)
    return local_directory


def list_adapters(s3_client: Any, user_id: str) -> List[str]:
    """Every adapter S3 prefix stored for ``user_id``.

    Lists all object keys under the user's adapter root and collapses them to
    the distinct ``adapters/{user_id}/{assistant_id}/{model}/`` prefixes.
    """
    user_root_prefix = ADAPTERS_PREFIX.format(
        user_id=user_id, assistant_id="", model=""
    ).rstrip("/")
    user_root_prefix = user_root_prefix + "/" if user_root_prefix else user_root_prefix

    paginator = s3_client.get_paginator("list_objects_v2")
    adapter_prefixes: List[str] = []
    seen_prefixes: set = set()
    for page in paginator.paginate(Bucket=BUCKET, Prefix=user_root_prefix):
        for s3_object in page.get("Contents", []):
            key_parts = s3_object["Key"].split("/")
            # adapters/{user_id}/{assistant_id}/{model}/<file...>
            if len(key_parts) < 5:
                continue
            prefix = "/".join(key_parts[:4]) + "/"
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                adapter_prefixes.append(prefix)
    return adapter_prefixes


def delete_adapter(
    s3_client: Any, user_id: str, assistant_id: str, model: str
) -> int:
    """Delete one adapter from S3 and its local copy. Returns objects deleted."""
    s3_prefix = adapter_s3_prefix(user_id, assistant_id, model)
    paginator = s3_client.get_paginator("list_objects_v2")
    deleted_count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        object_identifiers = [
            {"Key": s3_object["Key"]} for s3_object in page.get("Contents", [])
        ]
        if object_identifiers:
            s3_client.delete_objects(
                Bucket=BUCKET, Delete={"Objects": object_identifiers}
            )
            deleted_count += len(object_identifiers)

    local_directory = adapter_local_directory(user_id, assistant_id, model)
    shutil.rmtree(local_directory, ignore_errors=True)
    return deleted_count


def upload_basemodel_to_s3(s3_client: Any, model_id: str) -> int:
    """Download a base model from the Hugging Face Hub and mirror it to S3.

    Files land under ``MODELS_PREFIX + sanitize_model_name(model_id) + "/"``,
    preserving the snapshot's directory structure as S3 keys (the pattern
    prototyped in ``_.ipynb``). Returns the number of files uploaded.
    """
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s from the Hugging Face Hub...", model_id)
    local_snapshot_directory = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt", "*.model"],
    )
    logger.info("Model downloaded locally to: %s", local_snapshot_directory)

    model_s3_prefix = os.path.join(MODELS_PREFIX, sanitize_model_name(model_id))
    uploaded_count = 0
    for parent_directory, _subdirectories, filenames in os.walk(
        local_snapshot_directory
    ):
        for filename in filenames:
            local_file_path = os.path.join(parent_directory, filename)
            relative_path = os.path.relpath(local_file_path, local_snapshot_directory)
            s3_key = os.path.join(model_s3_prefix, relative_path).replace("\\", "/")
            logger.info("Uploading %s -> s3://%s/%s", relative_path, BUCKET, s3_key)
            s3_client.upload_file(local_file_path, BUCKET, s3_key)
            uploaded_count += 1
    return uploaded_count
