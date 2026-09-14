#!/usr/bin/env bash
set -euo pipefail

# Full decoupled run: Explorer GRPO + novelty, offline GRPO Actual, eval + plots.
# Uses .env if present (WANDB/HF tokens + project/entity).
#
# Usage:
#   bash scripts/run_full_tpu_decoupled.sh
#   SKIP_INSTALL=1 bash scripts/run_full_tpu_decoupled.sh
#   USE_TPU=0 bash scripts/run_full_tpu_decoupled.sh
#   DETACH=1 USE_TPU=0 bash scripts/run_full_tpu_decoupled.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

ROOT_AVAIL_MB="$(df -Pm / | tail -1 | awk '{print $4}')"
if [ "${ROOT_AVAIL_MB:-0}" -lt 2048 ]; then
  export HF_HOME="${HF_HOME:-/dev/shm/hf_home}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-/dev/shm/hf_hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/dev/shm/hf_datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/dev/shm/hf_transformers}"
  export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/xdg_cache}"
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$WANDB_DIR" "$XDG_CACHE_HOME"
fi

OUTPUT_ROOT="${EXPDIS_OUTPUT_DIR:-}"
if [ -z "$OUTPUT_ROOT" ] && [ "${ROOT_AVAIL_MB:-0}" -lt 2048 ]; then
  OUTPUT_ROOT="/dev/shm/expdis_outputs"
fi
if [ -n "$OUTPUT_ROOT" ]; then
  mkdir -p "$OUTPUT_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_ROOT/mpl_cache}"
  mkdir -p "$MPLCONFIGDIR"
fi

USE_TPU="${USE_TPU:-1}"
VENV_DIR="${VENV_DIR:-}"
TMPDIR="${TMPDIR:-/dev/shm}"

if [ -z "$VENV_DIR" ]; then
  if [ "${ROOT_AVAIL_MB:-0}" -lt 2048 ] && [ -d "/dev/shm" ]; then
    VENV_DIR="/dev/shm/rl_venv"
  else
    VENV_DIR=".venv"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv "$VENV_DIR"
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export TMPDIR
export PIP_NO_CACHE_DIR=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/dev/shm/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"

pip install --upgrade pip

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  if [ "$USE_TPU" = "1" ]; then
    TORCH_VERSION="${TORCH_VERSION:-2.2.0}"
    TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.2.0}"
    TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    XLA_WHEEL_URL="${XLA_WHEEL_URL:-https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.2/index.html}"
    LIBTPU_WHEEL_URL="${LIBTPU_WHEEL_URL:-https://storage.googleapis.com/libtpu-releases/index.html}"
    LIBTPU_VERSION="${LIBTPU_VERSION:-0.1.dev20231130}"
    pip install --index-url "${TORCH_CPU_INDEX_URL}" "torch==${TORCH_VERSION}" "torchvision==0.17.0" "torchaudio==2.2.0"
    pip install "torch_xla==${TORCH_XLA_VERSION}" -f "${XLA_WHEEL_URL}"
    if ! pip install "libtpu-nightly==${LIBTPU_VERSION}" -f "${LIBTPU_WHEEL_URL}"; then
      echo "[WARN] libtpu-nightly not found in wheel index; assuming system libtpu is present."
    fi
  else
    pip install torch torchvision torchaudio
  fi
  pip install "transformers<5" tokenizers trl datasets matplotlib tqdm accelerate scipy "wandb==0.17.9" sentencepiece tiktoken
  pip install "numpy<2"
  if [ "$USE_TPU" = "1" ]; then
    pip install "protobuf<4"
  fi
fi

export WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-full_run_$(date +%Y%m%d_%H%M%S)}"
export TOKENIZERS_PARALLELISM=false

if [ "$USE_TPU" = "1" ]; then
  export PJRT_DEVICE=TPU
  export USE_TORCH_XLA=1
  export EXPDIS_REQUIRE_TPU=1
  export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v4-8}"
  export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
  export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
  export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
  if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
    TPU_LIBRARY_PATH="$("$PYTHON_BIN" - <<'PY'
import os
try:
    import libtpu
    base = os.path.dirname(libtpu.__file__)
    print(os.path.join(base, 'libtpu.so'))
except Exception:
    print('')
PY
)"
    if [ -n "$TPU_LIBRARY_PATH" ]; then
      export TPU_LIBRARY_PATH
    fi
  fi
else
  unset PJRT_DEVICE USE_TORCH_XLA TPU_LIBRARY_PATH TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
MAX_TRAIN="${MAX_TRAIN:-7473}"
MAX_EVAL="${MAX_EVAL:-1319}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-200}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-192}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-96}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-288}"
GRPO_LOSS_TYPE="${GRPO_LOSS_TYPE:-dr_grpo}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-4}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.1}"
NOVELTY_METRIC="${NOVELTY_METRIC:-sqrt_mse}"
NOVELTY_CLIP="${NOVELTY_CLIP:-5.0}"
NOVELTY_FEATURE_SOURCE="${NOVELTY_FEATURE_SOURCE:-multilayer}"
NOVELTY_LAYERS="${NOVELTY_LAYERS:-}"
NOVELTY_LAYER_POOL="${NOVELTY_LAYER_POOL:-mean}"
NOVELTY_LAYER_AGG="${NOVELTY_LAYER_AGG:-mean}"
NOVELTY_ZSCORE="${NOVELTY_ZSCORE:-off}"
CORRECTNESS_ZSCORE="${CORRECTNESS_ZSCORE:-off}"
FORMAT_PENALTY_WEIGHT="${FORMAT_PENALTY_WEIGHT:-0.2}"
GATE_NOVELTY_BY_QUALITY="${GATE_NOVELTY_BY_QUALITY:-1}"
TEMP_SCHEDULE="${TEMP_SCHEDULE:-1.2,1.0,0.8}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
REWARD_INCORRECT="${REWARD_INCORRECT:--1.0}"
REWARD_SHAPE="${REWARD_SHAPE:-linear}"
REWARD_PIECEWISE_BOUNDS="${REWARD_PIECEWISE_BOUNDS:--1.0,-0.5,0.0,0.5,1.0}"
ACTUAL_TRAIN_MODE="${ACTUAL_TRAIN_MODE:-sft_grpo}"
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-2}"
ACTUAL_GRPO_BATCH="${ACTUAL_GRPO_BATCH:-2}"
ACTUAL_GRPO_KL="${ACTUAL_GRPO_KL:-0.05}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
DISTILL_BATCH="${DISTILL_BATCH:-4}"
BASE_EVAL_MODE="${BASE_EVAL_MODE:-grpo}"

EXPLORER_OUTPUT_DIR="${EXPLORER_OUTPUT_DIR:-}"
PLOTS_DIR="${PLOTS_DIR:-}"
EXPLORER_TRAJ_PATH="${EXPLORER_TRAJ_PATH:-}"
if [ -n "$OUTPUT_ROOT" ]; then
  EXPLORER_OUTPUT_DIR="${EXPLORER_OUTPUT_DIR:-$OUTPUT_ROOT/explorer_grpo_${GRPO_LOSS_TYPE}_lambda${LAMBDA_NOVELTY}}"
  PLOTS_DIR="${PLOTS_DIR:-$OUTPUT_ROOT/plots}"
  EXPLORER_TRAJ_PATH="${EXPLORER_TRAJ_PATH:-$OUTPUT_ROOT/explorer_trajectories.jsonl}"
fi

DEVICE_FLAG="cpu"
if [ "$USE_TPU" = "1" ]; then
  DEVICE_FLAG="tpu"
fi

EXTRA_ARGS=()
if [ -n "$EXPLORER_OUTPUT_DIR" ]; then
  EXTRA_ARGS+=(--explorer-output-dir "$EXPLORER_OUTPUT_DIR")
fi
if [ -n "$PLOTS_DIR" ]; then
  EXTRA_ARGS+=(--plots-dir "$PLOTS_DIR")
fi
if [ -n "$EXPLORER_TRAJ_PATH" ]; then
  EXTRA_ARGS+=(--explorer-traj-path "$EXPLORER_TRAJ_PATH")
fi

RUN_CMD=(
  "$PYTHON_BIN" run_expdis_torch.py --device "$DEVICE_FLAG" --use-wandb
  --model-name "$MODEL_NAME"
  --dataset-name "$DATASET_NAME"
  --decoupling-mode decoupled
  --actual-train-mode "$ACTUAL_TRAIN_MODE"
  --grpo-loss-type "$GRPO_LOSS_TYPE"
  --lambda-novelty "$LAMBDA_NOVELTY"
  --novelty-metric "$NOVELTY_METRIC"
  --novelty-clip "$NOVELTY_CLIP"
  --novelty-feature-source "$NOVELTY_FEATURE_SOURCE"
  --novelty-layer-pool "$NOVELTY_LAYER_POOL"
  --novelty-layer-agg "$NOVELTY_LAYER_AGG"
  --novelty-zscore "$NOVELTY_ZSCORE"
  --correctness-zscore "$CORRECTNESS_ZSCORE"
  --format-penalty-weight "$FORMAT_PENALTY_WEIGHT"
  --reward-incorrect "$REWARD_INCORRECT"
  --reward-shape "$REWARD_SHAPE"
  --reward-piecewise-bounds="$REWARD_PIECEWISE_BOUNDS"
  --temperature-schedule "$TEMP_SCHEDULE"
  --num-samples-per-prompt "$NUM_SAMPLES"
  --base-eval-mode "$BASE_EVAL_MODE"
  --max-train-examples "$MAX_TRAIN"
  --max-eval-examples "$MAX_EVAL"
  --grpo-max-steps "$GRPO_MAX_STEPS"
  --grpo-num-generations "$GRPO_NUM_GENERATIONS"
  --grpo-generation-batch-size "$GRPO_GEN_BATCH"
  --grpo-grad-accum "$GRPO_GRAD_ACCUM"
  --max-prompt-len "$MAX_PROMPT_LEN"
  --max-completion-len "$MAX_COMPLETION_LEN"
  --max-total-len "$MAX_TOTAL_LEN"
  --actual-grpo-epochs "$ACTUAL_GRPO_EPOCHS"
  --actual-grpo-batch-size "$ACTUAL_GRPO_BATCH"
  --actual-grpo-kl-coef "$ACTUAL_GRPO_KL"
  --distill-epochs "$DISTILL_EPOCHS"
  --distill-batch-size "$DISTILL_BATCH"
  "${EXTRA_ARGS[@]}"
)

if [ -n "$NOVELTY_LAYERS" ]; then
  RUN_CMD+=(--novelty-layers "$NOVELTY_LAYERS")
fi

if [ "$GATE_NOVELTY_BY_QUALITY" = "0" ]; then
  RUN_CMD+=(--no-gate-novelty-by-quality)
fi

if [ "${DETACH:-0}" = "1" ]; then
  LOG_PATH="${LOG_PATH:-/dev/shm/full_run_${DEVICE_FLAG}_$(date +%Y%m%d_%H%M%S).log}"
  nohup "${RUN_CMD[@]}" >"$LOG_PATH" 2>&1 &
  echo "Detached run started (PID $!). Log: $LOG_PATH"
  exit 0
fi

"${RUN_CMD[@]}"
