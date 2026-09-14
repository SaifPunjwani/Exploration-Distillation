#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [ -n "${HF_ACCESS_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HF_ACCESS_TOKEN"
fi
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv_tpu/bin/python" ]; then
    PYTHON_BIN=".venv_tpu/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

SESSION_NAME="${SESSION_NAME:-novelty_tpu_full}"
RUN_TAG="${RUN_TAG:-novelty_full_tpu}"
LOG_DIR="${LOG_DIR:-runs/${RUN_TAG}_logs}"
LOG_FILE="${LOG_DIR}/tmux_run_$(date +%Y%m%d_%H%M%S).log"
TPU_PREFLIGHT_TIMEOUT_SEC="${TPU_PREFLIGHT_TIMEOUT_SEC:-180}"
TPU_MIN_VISIBLE_DEVICES="${TPU_MIN_VISIBLE_DEVICES:-2}"

SMOKE_FIRST="${SMOKE_FIRST:-1}"
SMOKE_SEED="${SMOKE_SEED:-13}"

SEEDS="${SEEDS:-13 17 23}"
DATASETS="${DATASETS:-deepscaler_preview dapo_math_17k}"
MAX_TRAIN="${MAX_TRAIN:-4096}"
MAX_EVAL="${MAX_EVAL:-1024}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-120}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
ID_OOD_HOLDOUT_TOPICS="${ID_OOD_HOLDOUT_TOPICS:-geometry,number_theory}"

export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TMX_REQUIRE_TPU=1
export DEVICE=tpu
export TPU_MIN_VISIBLE_DEVICES
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export USE_TORCH_XLA=1
export TOKENIZERS_PARALLELISM=false
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
if [ -z "${PJRT_LOCAL_PROCESS_COUNT:-}" ]; then
  unset PJRT_LOCAL_PROCESS_COUNT || true
else
  export PJRT_LOCAL_PROCESS_COUNT
fi
if [ -z "${PJRT_LOCAL_PROCESS_RANK:-}" ]; then
  unset PJRT_LOCAL_PROCESS_RANK || true
else
  export PJRT_LOCAL_PROCESS_RANK
fi
if [ -n "${TPU_VISIBLE_DEVICES:-}" ]; then
  export TPU_VISIBLE_DEVICES
fi
if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
  TPU_LIBRARY_PATH="$("$PYTHON_BIN" - <<'PY'
import os
try:
    import libtpu
    base = os.path.dirname(libtpu.__file__)
    print(os.path.join(base, "libtpu.so"))
except Exception:
    print("")
PY
)"
  if [ -n "$TPU_LIBRARY_PATH" ]; then
    export TPU_LIBRARY_PATH
  fi
fi

mkdir -p "$LOG_DIR"

if [ "${INSTALL_EXTRA_DEPS:-1}" = "1" ]; then
  "$PYTHON_BIN" -m pip install --quiet --disable-pip-version-check scikit-learn umap-learn
fi

echo "[preflight] checking TPU device access + torch_xla..."
if command -v timeout >/dev/null 2>&1; then
  timeout "$TPU_PREFLIGHT_TIMEOUT_SEC" "$PYTHON_BIN" - <<'PY'
import os

paths = [f"/dev/accel{i}" for i in range(8)]
ok = False
for p in paths:
    if not os.path.exists(p):
        continue
    try:
        fd = os.open(p, os.O_RDWR)
        os.close(fd)
        print(f"{p}: OK")
        ok = True
    except OSError as e:
        print(f"{p}: FAIL ({e})")
if not ok:
    raise SystemExit("No openable /dev/accel* nodes. Aborting due to strict TPU requirement.")

os.environ.setdefault("PJRT_DEVICE", "TPU")
try:
    import torch_xla.core.xla_model as xm  # type: ignore
    dev = xm.xla_device()
    supported = xm.get_xla_supported_devices()
    min_visible = int(os.environ.get("TPU_MIN_VISIBLE_DEVICES", "2"))
    if len(supported) < min_visible:
        raise SystemExit(
            f"Only {len(supported)} XLA devices visible, expected at least {min_visible}."
        )
    print(f"XLA device: {dev}")
    print(f"XLA supported devices ({len(supported)}): {supported}")
except Exception as e:
    raise SystemExit(f"torch_xla preflight failed: {e}")
PY
else
  "$PYTHON_BIN" - <<'PY'
import os

paths = [f"/dev/accel{i}" for i in range(8)]
ok = False
for p in paths:
    if not os.path.exists(p):
        continue
    try:
        fd = os.open(p, os.O_RDWR)
        os.close(fd)
        print(f"{p}: OK")
        ok = True
    except OSError as e:
        print(f"{p}: FAIL ({e})")
if not ok:
    raise SystemExit("No openable /dev/accel* nodes. Aborting due to strict TPU requirement.")

os.environ.setdefault("PJRT_DEVICE", "TPU")
try:
    import torch_xla.core.xla_model as xm  # type: ignore
    dev = xm.xla_device()
    supported = xm.get_xla_supported_devices()
    min_visible = int(os.environ.get("TPU_MIN_VISIBLE_DEVICES", "2"))
    if len(supported) < min_visible:
        raise SystemExit(
            f"Only {len(supported)} XLA devices visible, expected at least {min_visible}."
        )
    print(f"XLA device: {dev}")
    print(f"XLA supported devices ({len(supported)}): {supported}")
except Exception as e:
    raise SystemExit(f"torch_xla preflight failed: {e}")
PY
fi

if [ "$SMOKE_FIRST" = "1" ]; then
  echo "[preflight] running strict TPU smoke run..."
  WANDB_GROUP="${RUN_TAG}_smoke_$(date +%Y%m%d_%H%M%S)" \
  DEVICE=tpu \
  TMX_REQUIRE_TPU=1 \
  MODEL_NAME="$MODEL_NAME" \
  DATASETS="$DATASETS" \
  SEEDS="$SMOKE_SEED" \
  MAX_TRAIN=24 \
  MAX_EVAL=24 \
  GRPO_MAX_STEPS=1 \
  RUN_TAG="${RUN_TAG}_smoke" \
  EMBEDDING_MODEL_NAME="$EMBEDDING_MODEL_NAME" \
  EMBEDDING_MAX_TEXTS=96 \
  EMBEDDING_PERM_ITERS=100 \
  ID_OOD_HOLDOUT_TOPICS="$ID_OOD_HOLDOUT_TOPICS" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_novelty_study_small.sh
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

WANDB_GROUP="${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
export WANDB_GROUP

RUN_CMD="cd '$ROOT_DIR' && \
  WANDB_GROUP='$WANDB_GROUP' \
  TMX_REQUIRE_TPU=1 \
  DEVICE=tpu \
  MODEL_NAME='$MODEL_NAME' \
  EMBEDDING_MODEL_NAME='$EMBEDDING_MODEL_NAME' \
  ID_OOD_HOLDOUT_TOPICS='$ID_OOD_HOLDOUT_TOPICS' \
  SEEDS='$SEEDS' \
  DATASETS='$DATASETS' \
  MAX_TRAIN='$MAX_TRAIN' \
  MAX_EVAL='$MAX_EVAL' \
  GRPO_MAX_STEPS='$GRPO_MAX_STEPS' \
  RUN_TAG='$RUN_TAG' \
  PYTHON_BIN='$PYTHON_BIN' \
  bash scripts/run_novelty_study_full.sh && \
  '$PYTHON_BIN' scripts/plot_novelty_study.py --root runs --tag '$RUN_TAG' --out-dir runs/${RUN_TAG}_aggregate"

tmux new-session -d -s "$SESSION_NAME" "bash -lc \"$RUN_CMD\" 2>&1 | tee '$LOG_FILE'"

echo "tmux session started: $SESSION_NAME"
echo "log file: $LOG_FILE"
echo "attach: tmux attach -t $SESSION_NAME"
echo "watch log: tail -f $LOG_FILE"
