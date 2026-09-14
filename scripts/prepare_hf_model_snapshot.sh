#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-$ROOT_DIR/.models}"
VLLM_LOCAL_MODEL_DIR="${VLLM_LOCAL_MODEL_DIR:-}"
VLLM_PREFETCH_RETRIES="${VLLM_PREFETCH_RETRIES:-3}"

log() {
  printf '[prefetch-model] %s\n' "$*" >&2
}

snapshot_is_complete() {
  local candidate_dir="$1"
  SNAPSHOT_DIR="$candidate_dir" python - <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(os.environ["SNAPSHOT_DIR"])

def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)

if not root.is_dir():
    fail(f"missing directory: {root}")

if not (root / "config.json").is_file():
    fail(f"missing config.json: {root}")

weight_index_candidates = [
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
]

for index_name in weight_index_candidates:
    index_path = root / index_name
    if not index_path.is_file():
        continue
    try:
        payload = json.loads(index_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        fail(f"failed to parse {index_path.name}: {exc}")
    weight_map = payload.get("weight_map") or {}
    referenced = sorted(set(weight_map.values()))
    if not referenced:
        fail(f"empty weight map in {index_path.name}")
    missing = [name for name in referenced if not (root / name).is_file()]
    if missing:
        fail(f"missing weight shards in {root}: {', '.join(missing)}")
    print(root)
    raise SystemExit(0)

weight_file_candidates = [
    "model.safetensors",
    "pytorch_model.bin",
    "consolidated.safetensors",
]

for filename in weight_file_candidates:
    if (root / filename).is_file():
        print(root)
        raise SystemExit(0)

fail(f"no complete weight files found in {root}")
PY
}

if [ -d "$MODEL_NAME" ]; then
  if snapshot_is_complete "$MODEL_NAME" >/dev/null; then
    printf '%s\n' "$MODEL_NAME"
    exit 0
  fi
  log "local model path is incomplete; will attempt repair: $MODEL_NAME"
fi

HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$LOCAL_MODEL_ROOT"

if [ -z "$VLLM_LOCAL_MODEL_DIR" ]; then
  model_tag="$(printf '%s' "$MODEL_NAME" | sed 's#[^A-Za-z0-9._-]#_#g')"
  VLLM_LOCAL_MODEL_DIR="$LOCAL_MODEL_ROOT/$model_tag"
fi
mkdir -p "$VLLM_LOCAL_MODEL_DIR"

if snapshot_is_complete "$VLLM_LOCAL_MODEL_DIR" >/dev/null 2>&1; then
  printf '%s\n' "$VLLM_LOCAL_MODEL_DIR"
  exit 0
fi

if [ -f "$VLLM_LOCAL_MODEL_DIR/config.json" ]; then
  log "existing snapshot is incomplete; continuing download repair for $VLLM_LOCAL_MODEL_DIR"
fi

source "$VLLM_VENV_DIR/bin/activate"

attempt=1
while [ "$attempt" -le "$VLLM_PREFETCH_RETRIES" ]; do
  log "snapshot_download attempt ${attempt}/${VLLM_PREFETCH_RETRIES}: $MODEL_NAME -> $VLLM_LOCAL_MODEL_DIR"
  if MODEL_NAME="$MODEL_NAME" VLLM_LOCAL_MODEL_DIR="$VLLM_LOCAL_MODEL_DIR" python - <<'PY'
import os

from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_NAME"]
local_dir = os.environ["VLLM_LOCAL_MODEL_DIR"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_files_only=False,
    resume_download=True,
    token=token,
)
PY
  then
    break
  fi
  if [ "$attempt" -eq "$VLLM_PREFETCH_RETRIES" ]; then
    log "ERROR: failed to prefetch $MODEL_NAME after ${VLLM_PREFETCH_RETRIES} attempts"
    exit 1
  fi
  sleep $((attempt * 5))
  attempt=$((attempt + 1))
done

if ! snapshot_is_complete "$VLLM_LOCAL_MODEL_DIR" >/dev/null 2>&1; then
  log "ERROR: prefetched directory is incomplete: $VLLM_LOCAL_MODEL_DIR"
  exit 1
fi

printf '%s\n' "$VLLM_LOCAL_MODEL_DIR"
