#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load .env if present (tokens/project/entity)
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

OUTPUT_ROOT="${TMX_OUTPUT_DIR:-}"
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$ROOT_DIR/runs"
fi
RUN_ID="${RUN_ID:-tpu_compare_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$OUTPUT_ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_ROOT/mpl_cache}"
mkdir -p "$MPLCONFIGDIR"

USE_TPU="${USE_TPU:-1}"
VENV_DIR="${VENV_DIR:-}"
if [ -z "$VENV_DIR" ]; then
  VENV_DIR="$ROOT_DIR/.venv"
fi
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export TMPDIR="${TMPDIR:-/dev/shm}"
export PIP_NO_CACHE_DIR=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/dev/shm/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  pip install --upgrade pip
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

export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false

if [ "$USE_TPU" = "1" ]; then
  export PJRT_DEVICE=TPU
  export USE_TORCH_XLA=1
  export TMX_REQUIRE_TPU=1
  export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v4-8}"
  export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
  export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
  export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
  if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
    TPU_LIBRARY_PATH="$($PYTHON_BIN - <<'PY'
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
MAX_TRAIN="${MAX_TRAIN:-256}"
MAX_EVAL="${MAX_EVAL:-128}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-50}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-8}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
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
NUM_SAMPLES="${NUM_SAMPLES:-2}"
REWARD_INCORRECT="${REWARD_INCORRECT:--1.0}"
REWARD_SHAPE="${REWARD_SHAPE:-linear}"
REWARD_PIECEWISE_BOUNDS="${REWARD_PIECEWISE_BOUNDS:--1.0,-0.5,0.0,0.5,1.0}"
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-1}"
ACTUAL_GRPO_BATCH="${ACTUAL_GRPO_BATCH:-2}"
ACTUAL_GRPO_KL="${ACTUAL_GRPO_KL:-0.05}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
DISTILL_BATCH="${DISTILL_BATCH:-4}"
BASE_EVAL_MODE="${BASE_EVAL_MODE:-grpo}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-192}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-96}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-288}"

DEVICE_FLAG="cpu"
if [ "$USE_TPU" = "1" ]; then
  DEVICE_FLAG="tpu"
fi

run_case() {
  local name="$1"
  local decoupling="$2"
  local actual_mode="$3"

  local run_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$run_dir"
  local plots_dir="$run_dir/plots"
  local traj_path="$run_dir/explorer_trajectories.jsonl"

  export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${name}_$(date +%H%M%S)}"
  echo "\n===== RUN: $name (mode=$decoupling, actual=$actual_mode) ====="

  CMD=(
    "$PYTHON_BIN" main.py --device "$DEVICE_FLAG" --use-wandb
    --model-name "$MODEL_NAME"
    --dataset-name "$DATASET_NAME"
    --decoupling-mode "$decoupling"
    --actual-train-mode "$actual_mode"
    --grpo-loss-type dr_grpo
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
    --grpo-batch-size "$GRPO_BATCH_SIZE"
    --grpo-grad-accum "$GRPO_GRAD_ACCUM"
    --max-prompt-len "$MAX_PROMPT_LEN"
    --max-completion-len "$MAX_COMPLETION_LEN"
    --max-total-len "$MAX_TOTAL_LEN"
    --actual-grpo-epochs "$ACTUAL_GRPO_EPOCHS"
    --actual-grpo-batch-size "$ACTUAL_GRPO_BATCH"
    --actual-grpo-kl-coef "$ACTUAL_GRPO_KL"
    --distill-epochs "$DISTILL_EPOCHS"
    --distill-batch-size "$DISTILL_BATCH"
    --plots-dir "$plots_dir"
    --explorer-traj-path "$traj_path"
  )
  if [ -n "$NOVELTY_LAYERS" ]; then
    CMD+=(--novelty-layers "$NOVELTY_LAYERS")
  fi
  if [ "$GATE_NOVELTY_BY_QUALITY" = "0" ]; then
    CMD+=(--no-gate-novelty-by-quality)
  fi
  "${CMD[@]}"
}

# 1) Coupled GRPO (Explorer==Actual): "GRPO on base" baseline (optional)
INCLUDE_COUPLED="${INCLUDE_COUPLED:-0}"
if [ "$INCLUDE_COUPLED" = "1" ]; then
  run_case "coupled_grpo" "coupled" "grpo"
fi

# 2) Decoupled with offline GRPO for Actual
run_case "decoupled_actual_grpo" "decoupled" "grpo"

# 3) Decoupled with SFT warm-start + offline GRPO for Actual
run_case "decoupled_actual_sft_grpo" "decoupled" "sft_grpo"

echo "\nAll runs complete. Outputs in: $OUTPUT_ROOT"
