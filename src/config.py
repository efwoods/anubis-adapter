"""Module-level configuration constants read from the local ``.env`` file.

There is deliberately no configuration-object abstraction in this repository
(unlike the main Anubis API's ``GlobalContext``): configuration is read once at
import time via ``python-dotenv`` and exposed as plain module constants. Every
key listed here must also appear in ``.env.example`` with its value left blank.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _environment_value(key: str, default: str | None = None) -> str | None:
    """Read one environment variable, treating an EMPTY value as unset.

    ``.env.example`` ships every key with a blank value, so a copied-but-not-
    yet-filled ``.env`` must behave as if the key were absent rather than
    feeding ``""`` into ``int()``/``float()`` conversions.
    """
    value = os.getenv(key)
    return value if value not in (None, "") else default


# boto3/botocore read the AWS keys straight from ``os.environ`` and treat an
# EMPTY string as a real value (an empty region yields the invalid endpoint
# ``https://s3..amazonaws.com``). A ``.env`` copied from ``.env.example`` with
# blanks must therefore have those blanks removed from the process
# environment, not just from this module's constants.
for _aws_environment_key in (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
):
    if os.environ.get(_aws_environment_key) == "":
        del os.environ[_aws_environment_key]

# ── AWS / S3 ────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID = _environment_value("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _environment_value("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = _environment_value("AWS_DEFAULT_REGION")
BUCKET = _environment_value("BUCKET", "maverick-4-4-bit-quantized")
MODELS_PREFIX = _environment_value("MODELS_PREFIX", "models/")
# A Python format string filled via
# ``ADAPTERS_PREFIX.format(user_id=..., assistant_id=..., model=...)``.
ADAPTERS_PREFIX = _environment_value(
    "ADAPTERS_PREFIX", "adapters/{user_id}/{assistant_id}/{model}/"
)

# ── Hugging Face ────────────────────────────────────────────────────────────
HF_TOKEN = _environment_value("HF_TOKEN")

# ── Base model & vLLM serving ───────────────────────────────────────────────
# The current resource-limited base model; Llama-4 Scout/Maverick stay
# selectable by pointing this at the larger model identifier.
BASE_MODEL = _environment_value("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
MAX_LORAS = int(_environment_value("MAX_LORAS", "8"))
MAX_LORA_RANK = int(_environment_value("MAX_LORA_RANK", "16"))
# 0.35 leaves GPU head-room for a concurrent training job on the same GPU.
GPU_MEMORY_UTILIZATION = float(_environment_value("GPU_MEMORY_UTILIZATION", "0.35"))
MAX_MODEL_LEN = int(_environment_value("MAX_MODEL_LEN", "4096"))

# ── Training ────────────────────────────────────────────────────────────────
TRAINING_CONCURRENCY = int(_environment_value("TRAINING_CONCURRENCY", "1"))
ADAPTER_LOCAL_DIRECTORY = _environment_value("ADAPTER_LOCAL_DIRECTORY", "/workspace/adapters")

# ── Anubis LangGraph store (PostgreSQL) ─────────────────────────────────────
# Read-only source of the per-user/per-assistant prompt-completion datasets
# written by the Anubis API's media -> identity pipeline.
ASYNC_POSTGRES_STORE_URI = _environment_value("ASYNC_POSTGRES_STORE_URI")

# ── Auth0 ───────────────────────────────────────────────────────────────────
AUTH0_DOMAIN = _environment_value("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = _environment_value("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = _environment_value("AUTH0_CLIENT_SECRET")
AUTH0_AUDIENCE = _environment_value("AUTH0_AUDIENCE")
AUTH0_CONNECTION = _environment_value("AUTH0_CONNECTION", "Username-Password-Authentication")

# ── HTTP server ─────────────────────────────────────────────────────────────
PORT = int(_environment_value("PORT", "8000"))
