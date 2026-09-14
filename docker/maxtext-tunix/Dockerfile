FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONPATH=/workspace/tmx

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip uv

# MaxText docs recommend maxtext[runner] for Docker image build/workload
# scheduling support. Tunix provides the JAX RL layer and rollout adapters.
RUN uv pip install --system --resolution=lowest \
    "maxtext[runner]==0.2.1" \
    "tunix" \
    "huggingface_hub[hf_transfer]" \
    "wandb"

WORKDIR /workspace/tmx
COPY . /workspace/tmx

ENTRYPOINT ["python", "-m", "tmx.orchestration.workload"]
