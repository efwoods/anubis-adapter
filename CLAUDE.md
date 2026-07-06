# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`anubis-adapter` is the **adapter-training and inference service for the Neural Nexus** (the product built on the main Anubis API at `github.com/efwoods/anubis`). Its job: take training data preprocessed by the Anubis API, fine-tune per-user/per-assistant LoRA adapters on a Llama base model, store them, and serve inference with the adapter attached — for many concurrent users and assistants.

- **Training**: TRL **GRPO** with reward functions that mirror the quality metrics used in the Anubis API.
- **Serving**: **vLLM** on **RunPod.io** GPUs; adapters attached at inference time.
- **Base models** (scaling smallest → largest as demand and demonstrated quality justify the cost): `meta-llama/Llama-3.2-11B-Vision-Instruct` (current, resource-limited) → `Llama-4-Scout-17B-16E-Instruct` → `Llama-4-Maverick-17B-128E-Instruct` (target). Quantized 4-bit (bitsandbytes nf4) in the notebook experiments.

This is the **scaffold/test/demo** stage. The intended demo path is: unadapted inference → adapter training → adapter storage → adapter attachment → multi-assistant, concurrent multi-user adapter inference.

## Current state (important)

There is **no package or module structure yet**. All work lives in a single exploratory Jupyter notebook, **`_.ipynb`**, containing scratch cells (model download, quantized load, S3 upload, smoke-test generation) — much of it commented out. `README.md`'s "Resource Requirements" section is a set of empty headings to be filled in with measured GPU/memory/storage/cost numbers. When adding real code, expect to introduce the actual service structure; do not assume one exists.

Note: `install.sh` only installs the Claude Code CLI — it is **not** a project setup script.

## Environment & config

Config is read from a local `.env` via `python-dotenv` and accessed directly through `os.environ` / `os.getenv` (there is no config-object abstraction). Keys currently in use:

- **AWS / S3**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `BUCKET` (`maverick-4-4-bit-quantized`), `MODELS_PREFIX` (`models/`), `ADAPTERS_PREFIX` (`adapters/{user_id}/{assistant_id}/{model}/` — a Python format string, filled via `.format(user_id=..., assistant_id=..., model=...)`).
- **Hugging Face**: `HF_TOKEN`, plus cache locations pinned to the RunPod workspace volume — `HF_HOME`, `CACHE_DIR`, `HF_HUB_CACHE` all under `/workspace/.cache/huggingface`, and `HF_HUB_OFFLINE`.

`.env` is gitignored (never commit it). There is no `.env.example`; when you add a new variable, create/update one with the key present and its value left blank.

## S3 storage layout

Single bucket (`BUCKET`), two prefixes:
- `models/` — base-model weights (uploaded from HF via `snapshot_download` → walk the local snapshot dir → `s3_client.upload_file`, preserving the directory structure as S3 keys).
- `adapters/{user_id}/{assistant_id}/{model}/` — trained adapters, namespaced per user, per assistant, per base model. This layout is what makes multi-user / multi-assistant adapter selection at inference time work; keep it consistent.

## Dependencies

Runtime stack (see `requirements.txt`): `vllm`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `huggingface_hub`, `boto3`, `dotenv`, `ipython`. Torch is installed separately against a CUDA wheel index (`--index-url https://download.pytorch.org/whl/cu124`, i.e. CUDA 12.4), not pinned in `requirements.txt`. This runs on GPU hosts (RunPod) — CUDA availability (`torch.cuda.is_available()`, `nvidia-smi`) is assumed.

## Relationship to the main Anubis repo

Training data comes **from** the Anubis API's media→identity preprocessing pipeline; the GRPO reward functions here are meant to **match** the quality dimensions defined there (avatar authenticity across relationships, knowledge, behavior, emotions, sentence-structure). Adapters produced here are the "adapter" lever of that system. When reward/quality logic is ambiguous, the Anubis API is the source of truth to align against.
