# [Initialize the anubis-adapter service (FastAPI + vLLM + TRL GRPO)](/home/user/.claude/plans/please-initialize-this-scalar-polished-horizon.md)

## Context

`anubis-adapter` is the adapter-training and inference service for Neural Nexus. Today the
repo has **no service code** — only an exploratory notebook (`_.ipynb`), a build spec
(`FEATURE.md`), reference implementations copied from the main Anubis API (`scaffold/`), and
`requirements.txt` (ML stack only; no web/auth deps). `FEATURE.md` asks for an async FastAPI
service that serves per-`user_id`/per-`assistant_id` base-model inference with LoRA adapters
attached, trains adapters via TRL GRPO, stores them in S3, and reports cost/capacity metrics —
for many concurrent users.

This plan **initializes that service**, mimicking the scaffold patterns (SSE `/message`, the
`media_job.py` background-job registry, `get_current_user`) rather than importing the scaffold
(its `from src.anubis…` imports don't exist here). The scaffold stays as reference; `_.ipynb`'s
working snippets (HF `snapshot_download` → S3 upload, BitsAndBytes nf4 load) are reused.

**Deployment target (per user): a RunPod VM with full resources** — capable GPU, network
volume, and AWS/HF/Auth0 credentials. So training runs **in-process on the VM** and inference
uses the real target base model. `BASE_MODEL` stays env-configurable so a small model can be
substituted for smoke tests.

### Decisions taken (flagged for override at approval)
- **Training execution**: in-process background job on the VM (the `scaffold/media_job.py`
  pattern), not a separate remote worker.
- **Auth**: lean Auth0 `get_current_user` (API-KEY → Auth0 Management-API lookup → hashed-key
  TTL cache), keyed callers only. No Supabase/Stripe/anonymous.
- **`/message` HITL**: SSE inference + a `/resume` endpoint mirroring the scaffold shape
  (`decision=apply|cancel`) — no true LangGraph interrupt loop.

## Package structure (new `src/` package, mirrors scaffold's `src.` import roots)

```
src/
  config.py                 # module-level constants read directly from os.environ (per CLAUDE.md: no config object)
  api/
    webapp.py               # FastAPI app, lifespan (engine + registries + S3 + Prometheus), metrics middleware, /metrics, / -> /docs
    inference_routes.py     # POST /message, /message/{assistant_id}, /message/{assistant_id}/resume  (SSE)
    training_routes.py      # POST /train_adapter, /cancel_adapter_training_job; GET /adapter_training_status, /adapter_training_progress (SSE)
    training_jobs.py        # TrainingJob dataclass + in-process registry (adapted from scaffold/media_job.py)
    cost_routes.py          # GET /cost/estimate
  security/
    auth.py                 # lean get_current_user + Auth0 mgmt-token plumbing + security_route (/signup, /rotate_api_key)
  inference/
    engine.py               # vLLM AsyncLLMEngine wrapper: load_basemodel, generate_stream, attach_adapter, remove_adapter (multi-LoRA via LoRARequest)
  training/
    grpo.py                 # run_training(job, request): QLoRA + TRL GRPOTrainer; save adapter -> S3
    rewards.py              # reward fns aligned to the 5 avatar quality dimensions (scaffold impls, swappable)
    dataset.py              # format Anubis-preprocessed training data into a GRPO dataset
  adapters/
    storage.py              # S3 save/download/list/delete adapter under adapters/{user_id}/{assistant_id}/{model}/ ; model up/download (from _.ipynb)
  metrics/
    prometheus.py           # CollectorRegistry + request/inference/training metrics (mirror scaffold names)
    cost.py                 # MODEL_PRICING (from FEATURE.md/README) + calculators: training cost, adapter storage cost, concurrency, break-even
```

Entrypoint: containerized (see **Containerization** below); the container `CMD` is
`uvicorn src.api.webapp:app --host 0.0.0.0 --port ${PORT:-8000}`.

## What each piece does

### Inference — `inference/engine.py`
- `load_basemodel()` builds a vLLM `AsyncLLMEngine` from `AsyncEngineArgs(model=BASE_MODEL,
  enable_lora=True, max_loras=MAX_LORAS, max_lora_rank=MAX_LORA_RANK,
  gpu_memory_utilization=…, max_model_len=…)`. Lazy singleton created in `lifespan`.
- `generate_stream(prompt, adapter_handle=None, sampling_params)` — async generator yielding
  token deltas; when an adapter is given, passes vLLM `LoRARequest(name, id, local_path)`.
- `attach_adapter(user_id, assistant_id, model)` — resolves the S3 prefix
  `adapters/{user_id}/{assistant_id}/{model}/`, ensures it's downloaded to a local dir
  (`adapters/storage.download_adapter`), returns a `LoRARequest` handle (vLLM loads on first
  use). `remove_adapter(...)` evicts the local copy + mapping.

### Inference endpoints — `api/inference_routes.py`
- `POST /message` and `POST /message/{assistant_id}` (`Depends(get_current_user)`): form fields
  mirror scaffold (`message`, `thread_id`, `stream`, `your_name`…). Resolve+attach the
  caller's adapter, then stream SSE `{"type":"assistant_token","text":…}` events followed by a
  terminal `{"type":"done", …, "total_response_time_ms":…}` — reusing the scaffold's
  `message_graph_sse` event shape so existing clients work unchanged.
- `POST /message/{assistant_id}/resume`: parity shape (`thread_id`, `decision=apply|cancel`).
  `apply` continues/regenerates the thread; `cancel` ends it. Accepts legacy
  `approve`/`reject` aliases like the scaffold. A short docstring notes there is no interrupt
  graph — this is thread continuation, not tool-approval HITL.
- Lightweight **in-process thread store** (dict `thread_id -> messages`) so multi-turn +
  resume work; documented as demo-only (production: Postgres/Redis).

### Training job registry — `api/training_jobs.py`  (adapted from `scaffold/media_job.py`)
- `TrainingJob` dataclass: `job_id, user_id, assistant_id, model, status`
  (`queued|running|completed|error|cancelled`), timing, progress
  (`current_step, total_steps, latest_loss, latest_reward`), append-only `events` buffer +
  `asyncio.Event` for subscribers, `done` event, `task`, cooperative `cancelled` flag.
- Registry lives on `app.state.training_jobs`. Reuses the scaffold's `add_event` /
  `finish_job` / `request_cancel` / TTL `_cleanup` design (late subscribers replay from index 0).

### Training endpoints — `api/training_routes.py`
- `POST /train_adapter` (`Depends(get_current_user)`): body = `assistant_id`, optional `model`
  (default target), GRPO/LoRA config (epochs, lr, lora rank/alpha), and a training-data
  reference (inline or S3). Creates a `TrainingJob`, launches it, returns `job_id` + `202`.
- `GET /adapter_training_status?job_id=` — snapshot (status, progress metrics, duration, and
  the adapter S3 prefix when complete).
- `GET /adapter_training_progress?job_id=` — SSE replay of `job.events` then live updates
  (same pattern as scaffold media progress).
- `POST /cancel_adapter_training_job?job_id=` — `request_cancel`.

### Training run — `training/grpo.py`
- `run_training(job, request)`: load base + tokenizer with QLoRA (BitsAndBytes nf4 config from
  `_.ipynb`) + PEFT `LoraConfig`; build `GRPOConfig` + `GRPOTrainer` with reward fns from
  `rewards.py`; dataset from `training/dataset.py`. A trainer callback pushes per-step
  `{step, loss, reward}` into the job buffer and checks `job.cancelled`. On completion, save the
  adapter locally then `adapters/storage.save_adapter(...)` to S3. Runs via
  `loop.run_in_executor` so the blocking training loop doesn't stall the event loop (mirrors the
  media_job runner structure).
- `rewards.py`: one reward function per **avatar quality dimension** (relationships, knowledge,
  behavior, emotions, sentence-structure), with scaffold implementations (e.g. embedding /
  heuristic similarity to reference quotes) and a clear TODO that the Anubis API reward logic is
  the source of truth to align against.

### Adapters + models — `adapters/storage.py`
- `save_adapter` / `download_adapter` / `list_adapters` / `delete_adapter` over the S3 layout
  `adapters/{user_id}/{assistant_id}/{model}/` (walk local dir → `upload_file`, preserving keys;
  reverse for download) — the walk/upload loop is lifted from `_.ipynb`'s
  `upload_hf_model_to_s3`. Also `upload_basemodel_to_s3` (HF `snapshot_download` →
  `models/` prefix) reused from the notebook.

### Auth — `security/auth.py`
- Port the **core** of `scaffold/security/auth.py`: `generate_api_key`, `_hash_key`,
  `_get_mgmt_token`/`_mgmt_headers` (cached Auth0 Management token), `retry_async_httpx_request`,
  `get_user_with_api_key` (hashed-key `TTLCache` + Auth0 `/api/v2/users` lookup, clean 503 on
  transport error), and `get_current_user`. Include a small `security_route` with `/signup` and
  `/rotate_api_key`. Drop Supabase, Stripe, and the anonymous-user path.
- Env: `AUTH0_DOMAIN`, `AUTH0_CLIENT_ID`, `AUTH0_CLIENT_SECRET`, `AUTH0_AUDIENCE`,
  `AUTH0_CONNECTION`.

### Metrics + cost — `metrics/`
- `prometheus.py`: `CollectorRegistry` with request count/latency + active-requests (scaffold
  names), inference token counters, and training gauges (active training jobs, training
  duration, adapter size bytes). Exposed at `GET /metrics`; `metrics_middleware` mirrors scaffold.
- `cost.py`: `MODEL_PRICING` dict transcribed from `FEATURE.md`/`README.md` (per-model size,
  volume $/mo, GPU $/hr; S3 tiers $0.023/0.022/0.021 per GB). Calculators:
  `adapter_storage_cost_per_month(size_bytes)`, `training_cost(model, duration_hours)`,
  `basemodel_storage_cost(model)`, `concurrency_capacity(...)`, `break_even(...)`. Surfaced via
  `GET /cost/estimate` and used to fill `README.md`'s empty **Resource Requirements** headings.

### App wiring — `api/webapp.py`
- `FastAPI(title="Neural Nexus Adapter API", lifespan=…)`. `lifespan` initializes the boto3 S3
  client, the vLLM engine (lazy — first inference call, so the app boots before weights load),
  the `training_jobs` + adapter registries, and the Prometheus registry. `metrics_middleware`,
  `GET /metrics`, `GET /health` (liveness for the compose healthcheck — returns
  `{"status":"ok"}` without touching the GPU/engine), `GET /` → redirect `/docs`, and
  `include_router` for security + inference + training + cost routes.

## Containerization (RunPod VM, GPU runtime)

The main Anubis repo's Docker files build on `langchain/langgraph-api:3.11-wolfi` — the wrong
base here; this service needs CUDA + a GPU runtime for vLLM/TRL. Mirror the Anubis *conventions*
(single `env_file`, HF-cache volume mount, healthcheck, tight `.dockerignore`) on a CUDA base.

- **`Dockerfile`** (single stage, cache-friendly): base
  `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`; install Python 3.11 + build basics; **copy
  `requirements.txt` first**, then `pip install torch torchvision torchaudio --index-url
  https://download.pytorch.org/whl/cu124` followed by `pip install -r requirements.txt` (layer
  caches so source edits don't reinstall vLLM); then `COPY src/ ./src/`. Set the HF cache env
  (`HF_HOME`/`HF_HUB_CACHE` under `/workspace/.cache/huggingface`, matching CLAUDE.md).
  `EXPOSE 8000`; `CMD ["uvicorn","src.api.webapp:app","--host","0.0.0.0","--port","8000"]`.
- **`docker-compose.yml`**: one `adapter-api` service — `build: .`, `env_file: .env`,
  `ports: ["${PORT:-8000}:8000"]`, `restart: unless-stopped`, a `healthcheck` hitting
  `GET /health`, and **GPU reservation**:
  ```yaml
  deploy:
    resources:
      reservations:
        devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
  ```
  Volumes: `~/.cache/huggingface:/workspace/.cache/huggingface` (survive rebuilds; matches the
  Anubis HF-cache mount) and a named volume for the local adapter/model download dir.
- **`.dockerignore`**: exclude `.venv/`, `.git/`, `_.ipynb`, `scaffold/` (reference only —
  not shipped in the image), `__pycache__/`, `.env*`.
- **`Makefile`** (small): `build` (`docker compose build`), `up` (`docker compose up`),
  `down`, `logs`. Optional follow-up: split a deps base image like Anubis's two-stage build if
  vLLM rebuilds get slow.

## Dependency + config changes
- **`requirements.txt`**: add `fastapi`, `uvicorn[standard]`, `python-multipart`,
  `prometheus-client`, `httpx`, `python-jose[cryptography]`, `cachetools`. Torch stays a
  separate cu124 install (unchanged convention).
- **`.env.example`** (new, per CLAUDE.md convention — keys present, values blank): all existing
  AWS/HF keys + `AUTH0_DOMAIN`/`AUTH0_CLIENT_ID`/`AUTH0_CLIENT_SECRET`/`AUTH0_AUDIENCE`/
  `AUTH0_CONNECTION` + `BASE_MODEL` + inference knobs (`MAX_LORAS`, `MAX_LORA_RANK`,
  `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`). Add the same keys (blank) to `.env`… (values are
  user-supplied; `.env` is gitignored).
- **`config.py`**: thin module of `os.getenv(...)` constants (not a config object) so the rest
  of the code imports names instead of scattering env reads.

## Out of scope (noted follow-ups)
Two-stage deps/base Docker split (single Dockerfile now), persistent thread/checkpoint store,
real Anubis-aligned reward math, Stripe metering.

## Verification (on the RunPod VM)
1. Populate `.env` (AWS, HF, AUTH0, `BASE_MODEL`); `docker compose build` (installs cu124 torch
   + `requirements.txt`).
2. `docker compose up` — the container boots and `GET /health`, `GET /docs`, `GET /metrics`,
   `GET /` → `/docs` all respond (engine lazy — not yet loaded); confirm the GPU is visible in
   the container (`nvidia-smi` inside, or first inference call succeeds).
3. **Inference**: `POST /message/{assistant_id}` with `API-KEY` → SSE `assistant_token…done`
   (use a small `BASE_MODEL` for a fast smoke test); multi-turn via `thread_id`; `/resume`
   with `decision=apply` continues.
4. **Training**: `POST /train_adapter` → `job_id`; `GET /adapter_training_progress` streams
   `{step,loss,reward}`; on completion the adapter appears in S3 under
   `adapters/{user}/{assistant}/{model}/` and `GET /adapter_training_status` shows `completed`.
   Then `POST /message` reflects the attached adapter.
5. **Cancel**: `POST /cancel_adapter_training_job` mid-run → status `cancelled`.
6. **Cost**: `GET /cost/estimate` returns numbers; `README.md` Resource Requirements sections
   filled from `metrics/cost.py`.



https://huggingface.co/docs/trl/en/grpo_trainer given the features established here: /home/user/gh/anubis-project/anubis/src/anubis/utils/dataset/style_features.py, the prompt completion format datasets per user_id and avatar_id located in the vectorstor from this script: /home/user/gh/anubis-project/anubis/sql/adapter_dataset_store_search.sql I need async reward functions per feature and to be able to pull that data to train a
  model on runpod.io with GRPO using the media job implementation. The model choice will begin with meta-llama/Llama-3.2-1B-Instruct for now. The endpoint accepts a user_id (Depends(get_current_user)) and the avatar_id, searches for prompt completion format data in the vectorstore, and starts the media job of training a LoRA adapter for that model /train_adapter. I am not merging the weights into the basemodel.

I am not quantizing the base model using bitsandbytes for the Llama-3.2-1B-Instruct LoRA adapter: from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
	torch_dtype=torch.bfloat16,
	quantization_config=bnb_config,
	trust_remote_code=True
).to(model.device)


model.config.use_cache = False
model.config.pretraining_tp = 1

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


# Use the following reference material as guidance for training and serving:
https://www.datacamp.com/tutorial/llama-4-vllm
https://www.datacamp.com/tutorial/fine-tuning-llama-4
https://huggingface.co/docs/trl/en/grpo_trainer

Instead of scout or qwen or any bnb quantizations, use the following models:
NOW FOR SCAFFOLDING/DEMO/TEST
meta-llama/Llama-3.2-1B-Instruct

FUTURE:
meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 (FOR LORA TRAINING)
unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit (FOR LORA TRAINING)

meta-llama/Llama-4-Scout-17B-16E-Instruct (for inference)
meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 (FOR INFERENCE WITH ADAPTERS)
