# Neural Nexus Adapter API — CUDA runtime image for RunPod GPU hosts.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        build-essential git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Torch against the CUDA 12.4 wheel index, then the pinned application deps —
# requirements.txt copied first so dependency layers cache across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

# Pre-fetch the nltk data the stylometric reward features need so the first
# training step never pays a download.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data \
        punkt punkt_tab averaged_perceptron_tagger_eng stopwords

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api.webapp:app", "--host", "0.0.0.0", "--port", "8000"]
