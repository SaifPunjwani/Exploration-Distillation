#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
TPU_NAME="${TPU_NAME:?TPU_NAME is required}"
ZONE="${ZONE:?ZONE is required}"
WORKER="${WORKER:-0}"
TPU_SSH_MODE="${TPU_SSH_MODE:-iap}"
REMOTE_HF_HUB="${REMOTE_HF_HUB:?set REMOTE_HF_HUB to the Hugging Face hub cache directory on the TPU VM}"

if [ -z "$MODEL_CACHE_DIR" ]; then
  echo "MODEL_CACHE_DIR is required" >&2
  exit 1
fi

if [ ! -d "$MODEL_CACHE_DIR" ]; then
  echo "MODEL_CACHE_DIR not found: $MODEL_CACHE_DIR" >&2
  exit 1
fi

transport_flag=()
if [ "$TPU_SSH_MODE" = "iap" ]; then
  transport_flag+=(--tunnel-through-iap)
fi

tar_path="$ROOT_DIR/runs/$(basename "$MODEL_CACHE_DIR")_$(date +%Y%m%d_%H%M%S).tar"
tar -cf "$tar_path" -C "$(dirname "$MODEL_CACHE_DIR")" "$(basename "$MODEL_CACHE_DIR")"
tar_base="$(basename "$tar_path")"

gcloud alpha compute tpus tpu-vm scp \
  "$tar_path" \
  "${TPU_NAME}:/tmp/" \
  --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
  --zone="$ZONE" \
  --worker="$WORKER" \
  "${transport_flag[@]}"

gcloud alpha compute tpus tpu-vm ssh \
  "$TPU_NAME" \
  --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
  --zone="$ZONE" \
  --worker="$WORKER" \
  "${transport_flag[@]}" \
  --command="mkdir -p '$REMOTE_HF_HUB' && tar -xf '/tmp/$tar_base' -C '$REMOTE_HF_HUB' && find '$REMOTE_HF_HUB' -maxdepth 2 -type d -name '$(basename "$MODEL_CACHE_DIR")'"
