#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
SOURCE_MODEL_NAME="${SOURCE_MODEL_NAME:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-bfloat16}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HF_HUB_CACHE}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-$ROOT_DIR/.models}"
VLLM_LOCAL_MODEL_DIR="${VLLM_LOCAL_MODEL_DIR:-}"
VLLM_PREFETCH_MODEL="${VLLM_PREFETCH_MODEL:-0}"
VLLM_PREFETCH_RETRIES="${VLLM_PREFETCH_RETRIES:-3}"
HF_HUB_OFFLINE_AFTER_PREFETCH="${HF_HUB_OFFLINE_AFTER_PREFETCH:-0}"

source "$VLLM_VENV_DIR/bin/activate"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$LOCAL_MODEL_ROOT"

if [ -d "$MODEL_NAME" ]; then
  if ! MODEL_NAME="$MODEL_NAME" \
    VLLM_VENV_DIR="$VLLM_VENV_DIR" \
    LOCAL_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
    VLLM_LOCAL_MODEL_DIR="$MODEL_NAME" \
    VLLM_PREFETCH_RETRIES="$VLLM_PREFETCH_RETRIES" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    bash scripts/prepare_hf_model_snapshot.sh >/dev/null
  then
    if [ -z "$SOURCE_MODEL_NAME" ] || [ "$SOURCE_MODEL_NAME" = "$MODEL_NAME" ]; then
      echo "[run-vllm] ERROR: local model snapshot is incomplete and no source repo is available: $MODEL_NAME" >&2
      exit 1
    fi
    MODEL_NAME="$(
      MODEL_NAME="$SOURCE_MODEL_NAME" \
      VLLM_VENV_DIR="$VLLM_VENV_DIR" \
      LOCAL_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
      VLLM_LOCAL_MODEL_DIR="$MODEL_NAME" \
      VLLM_PREFETCH_RETRIES="$VLLM_PREFETCH_RETRIES" \
      HF_HOME="$HF_HOME" \
      HF_HUB_CACHE="$HF_HUB_CACHE" \
      HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
      TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
      bash scripts/prepare_hf_model_snapshot.sh
    )"
  fi
elif [ "$VLLM_PREFETCH_MODEL" = "1" ]; then
  MODEL_NAME="$(
    MODEL_NAME="$MODEL_NAME" \
    VLLM_VENV_DIR="$VLLM_VENV_DIR" \
    LOCAL_MODEL_ROOT="$LOCAL_MODEL_ROOT" \
    VLLM_LOCAL_MODEL_DIR="$VLLM_LOCAL_MODEL_DIR" \
    VLLM_PREFETCH_RETRIES="$VLLM_PREFETCH_RETRIES" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    bash scripts/prepare_hf_model_snapshot.sh
  )"
fi

if [ "$HF_HUB_OFFLINE_AFTER_PREFETCH" = "1" ] && [ -d "$MODEL_NAME" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

pkill -u "$USER" -f "vllm serve .*--port ${PORT}" >/dev/null 2>&1 || true
pkill -u "$USER" -f "${VLLM_VENV_DIR}/bin/vllm serve .*--port ${PORT}" >/dev/null 2>&1 || true
sleep 2

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0,1,2,3}"
export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS:-2,2,1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
# Persist XLA compile cache across vLLM restarts (saves 60-90s per reload).
# vLLM-TPU uses torch_xla's XLA_PERSISTENT_CACHE_PATH for compile artifacts.
export XLA_PERSISTENT_CACHE_PATH="${XLA_PERSISTENT_CACHE_PATH:-/home/${USER}/vllm_xla_cache}"
mkdir -p "$XLA_PERSISTENT_CACHE_PATH" 2>/dev/null || true

ray stop -f >/dev/null 2>&1 || true
ray start --head --disable-usage-stats --port=6379 >/dev/null

ARGS=(
  "$MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --tensor-parallel-size "$TP_SIZE"
  --distributed-executor-backend ray
  --dtype "$DTYPE"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --download-dir "$DOWNLOAD_DIR"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)

case "$ENABLE_PREFIX_CACHING" in
  1|true|TRUE|yes|YES)
    ARGS+=(--enable-prefix-caching)
    ;;
esac

if [ -n "$VLLM_API_KEY" ]; then
  ARGS+=(--api-key "$VLLM_API_KEY")
fi

# Launch vLLM. With EXPDIS_VLLM_HOT_RELOAD=1, try patched server with /update_weights.
# Safety: if patched server exits within 60s (crash on import/startup), fall back
# to `vllm serve` so training pipeline doesn't break.
if [ "${EXPDIS_VLLM_HOT_RELOAD:-0}" = "1" ] && [ -f "$ROOT_DIR/scripts/vllm_tpu_hot_server.py" ]; then
  echo "[run-vllm] attempting patched server with /update_weights endpoint"
  "$VLLM_VENV_DIR/bin/python" "$ROOT_DIR/scripts/vllm_tpu_hot_server.py" "${ARGS[@]}" &
  HOT_PID=$!
  # Watch for 60s: if process dies, fall back. If process is still running OR
  # port opens, assume it's starting normally and wait for it.
  for i in $(seq 1 60); do
    if ! kill -0 "$HOT_PID" 2>/dev/null; then
      echo "[run-vllm] patched server exited early; falling back to 'vllm serve'"
      HOT_PID=""
      break
    fi
    if curl -s --connect-timeout 1 "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "[run-vllm] patched server is up; handing off"
      wait "$HOT_PID"
      exit $?
    fi
    sleep 1
  done
  if [ -n "$HOT_PID" ]; then
    # Still running after 60s but port not ready — probably still initializing.
    # Keep it, don't fall back.
    wait "$HOT_PID"
    exit $?
  fi
fi

exec vllm serve "${ARGS[@]}"
