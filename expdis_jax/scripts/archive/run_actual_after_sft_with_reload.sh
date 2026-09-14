#!/usr/bin/env bash
# Continue a pipeline after SFT has completed:
# 1. reload all configured vLLM workers to the SFT HF bundle, retrying forever;
# 2. run Actual correctness-only DR-GRPO from the SFT Orbax checkpoint;
# 3. export/upload the final Actual HF bundle.
set -euo pipefail

RUN_NAME="${RUN_NAME:?Must set RUN_NAME}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/expdis/runs}"
REPO_ROOT="${REPO_ROOT:-$HOME/expdis}"
ROOT="$RUNS_ROOT/$RUN_NAME"
ACTUAL_DIR="$ROOT/actual"
SFT_CKPT="${SFT_CKPT:-$ACTUAL_DIR/actual_sft_final}"
SFT_CKPT_GCS="${SFT_CKPT_GCS:-}"
SFT_GCS="${SFT_GCS:-}"
ACTUAL_OUT="${ACTUAL_OUT:-$ACTUAL_DIR/grpo}"
EXPDIS_ARTIFACT_BACKEND="${EXPDIS_ARTIFACT_BACKEND:-hf}"
EXPDIS_ALLOW_GCS_ARTIFACTS="${EXPDIS_ALLOW_GCS_ARTIFACTS:-0}"
EXPDIS_HF_CHECKPOINT_REPO="${EXPDIS_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}"
EXPDIS_HF_REPO_TYPE="${EXPDIS_HF_REPO_TYPE:-dataset}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
VLLM_URLS="${EXPDIS_VLLM_SERVER_URLS:-${VLLM_SERVER_URLS:-}}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-8192}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-$((MAX_PROMPT_LEN + MAX_COMPLETION_LEN))}"
NOVELTY_FEATURE_MAX_LENGTH="${NOVELTY_FEATURE_MAX_LENGTH:-$MAX_TOTAL_LEN}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-8}"
GRPO_CLIP_EPSILON="${GRPO_CLIP_EPSILON:-0.2}"
GRPO_CLIP_EPSILON_HIGH="${GRPO_CLIP_EPSILON_HIGH:-$GRPO_CLIP_EPSILON}"
GRPO_ADVANTAGE_NORMALIZATION="${GRPO_ADVANTAGE_NORMALIZATION:-none}"
DYNAMIC_SAMPLING="${DYNAMIC_SAMPLING:-false}"
DYNAMIC_SAMPLING_MAX_ATTEMPTS="${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-8}"
MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-false}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:-0.0}"
NONTERMINATION_PENALTY="${NONTERMINATION_PENALTY:-0.0}"
INVALID_ANSWER_PENALTY="${INVALID_ANSWER_PENALTY:-0.0}"
SOFT_OVERLONG_EXPECTED_LEN="${SOFT_OVERLONG_EXPECTED_LEN:-0}"
SOFT_OVERLONG_CACHE_LEN="${SOFT_OVERLONG_CACHE_LEN:-0}"

if [ -z "$VLLM_URLS" ]; then
  echo "[actual-resume] ERROR: set EXPDIS_VLLM_SERVER_URLS or VLLM_SERVER_URLS" >&2
  exit 2
fi
if [ ! -d "$SFT_CKPT" ]; then
  if [ "${EXPDIS_ALLOW_GCS_ARTIFACTS:-0}" != "1" ] || [ -z "$SFT_CKPT_GCS" ]; then
    echo "[actual-resume] ERROR: missing local SFT checkpoint and GCS fallback is disabled: $SFT_CKPT" >&2
    exit 2
  fi
  echo "[actual-resume] SFT checkpoint missing locally; downloading $SFT_CKPT_GCS -> $SFT_CKPT" >&2
  sft_parent="$(dirname "$SFT_CKPT")"
  mkdir -p "$sft_parent"
  rm -rf "$SFT_CKPT"
  gsutil -m cp -r "$SFT_CKPT_GCS" "$sft_parent/"
fi
if [ ! -d "$SFT_CKPT" ]; then
  echo "[actual-resume] ERROR: missing SFT checkpoint after download: $SFT_CKPT" >&2
  exit 2
fi

cd "$REPO_ROOT"
source "${ENV_DIR:-$HOME/miniforge3}/bin/activate" "${ENV_NAME:-expdis_jax}"
mkdir -p "$ACTUAL_OUT"
INIT_CKPT="${INIT_CKPT:-}"
if [ -z "$INIT_CKPT" ]; then
  INIT_CKPT="$(find "$ACTUAL_OUT" -maxdepth 1 -type d -name 'step_*' 2>/dev/null | sort | tail -1 || true)"
fi
if [ -z "$INIT_CKPT" ]; then
  INIT_CKPT="$SFT_CKPT"
fi
echo "[actual-resume] init checkpoint: $INIT_CKPT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [ "${EXPDIS_ENABLE_JAX_COMPILATION_CACHE:-1}" != "0" ]; then
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/expdis_jax_compilation_cache}"
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-5}"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
fi
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0,1,2,3}"
export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
export EXPDIS_ARTIFACT_BACKEND="$EXPDIS_ARTIFACT_BACKEND"
export EXPDIS_ALLOW_GCS_ARTIFACTS="$EXPDIS_ALLOW_GCS_ARTIFACTS"
export EXPDIS_HF_CHECKPOINT_REPO="$EXPDIS_HF_CHECKPOINT_REPO"
export EXPDIS_HF_REPO_TYPE="$EXPDIS_HF_REPO_TYPE"
export EXPDIS_GCS_CHECKPOINT_BASE="${EXPDIS_GCS_CHECKPOINT_BASE:-}"
export EXPDIS_SAVE_OPT_STATE="${EXPDIS_SAVE_OPT_STATE:-0}"
export EXPDIS_RESTORE_OPT_STATE="${EXPDIS_RESTORE_OPT_STATE:-0}"
export EXPDIS_CKPT_DELETE_OLD_RUNS="${EXPDIS_CKPT_DELETE_OLD_RUNS:-1}"
export EXPDIS_LOCAL_CHECKPOINT_EVERY_STEPS="${EXPDIS_LOCAL_CHECKPOINT_EVERY_STEPS:-3}"
export EXPDIS_GCS_CHECKPOINT_EVERY_STEPS="${EXPDIS_GCS_CHECKPOINT_EVERY_STEPS:-${SAVE_EVERY_STEPS:-50}}"
export EXPDIS_LOCAL_CHECKPOINT_KEEP_LAST="${EXPDIS_LOCAL_CHECKPOINT_KEEP_LAST:-1}"
export EXPDIS_LOCAL_VLLM_RELOAD_KEEP_LAST="${EXPDIS_LOCAL_VLLM_RELOAD_KEEP_LAST:-0}"
export EXPDIS_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD="${EXPDIS_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD:-1}"
export EXPDIS_CKPT_MIN_FREE_GB="${EXPDIS_CKPT_MIN_FREE_GB:-28}"
export EXPDIS_VLLM_EXPORT_MIN_FREE_GB="${EXPDIS_VLLM_EXPORT_MIN_FREE_GB:-28}"
export EXPDIS_WANDB_SINGLE_RUN="${EXPDIS_WANDB_SINGLE_RUN:-1}"
export EXPDIS_WANDB_COMPACT="${EXPDIS_WANDB_COMPACT:-1}"
export EXPDIS_WANDB_MINIMAL="${EXPDIS_WANDB_MINIMAL:-1}"
export EXPDIS_VLLM_RETURN_LOGPROBS="${EXPDIS_VLLM_RETURN_LOGPROBS:-0}"
export EXPDIS_TRACK_EXTERNAL_OLD_GAP="${EXPDIS_TRACK_EXTERNAL_OLD_GAP:-0}"
export EXPDIS_PREFETCH_ROLLOUTS="${EXPDIS_PREFETCH_ROLLOUTS:-1}"
export EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS="${EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS:-1}"
export EXPDIS_TRAIN_VLLM_RELOAD_ON_RESUME="${EXPDIS_TRAIN_VLLM_RELOAD_ON_RESUME:-1}"
export EXPDIS_TRAIN_VLLM_RELOAD_MODE="${EXPDIS_TRAIN_VLLM_RELOAD_MODE:-external}"
export EXPDIS_EXTERNAL_VLLM_RELOAD_WAIT="${EXPDIS_EXTERNAL_VLLM_RELOAD_WAIT:-1}"
export EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED="${EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED:-1}"
export EXPDIS_EXTERNAL_VLLM_RELOAD_TIMEOUT="${EXPDIS_EXTERNAL_VLLM_RELOAD_TIMEOUT:-0}"
export EXPDIS_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL="${EXPDIS_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL:-1}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_key" ]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_key")"
fi

reload_workers="${EXPDIS_VLLM_RELOAD_WORKERS:-}"
if [ -z "$reload_workers" ]; then
  echo "[actual-resume] ERROR: set EXPDIS_VLLM_RELOAD_WORKERS" >&2
  exit 2
fi

attempt=0
if [ "${EXPDIS_ACTUAL_ASSUME_VLLM_RELOADED:-0}" = "1" ]; then
  echo "[actual-resume] assuming vLLM already reloaded to $SFT_GCS $(date -Is)"
else
  while true; do
    attempt=$((attempt + 1))
    echo "[actual-resume] vLLM SFT reload attempt $attempt $(date -Is)"
    if env \
      GCS_MODEL_PATH="$SFT_GCS" \
      TPU_NAME="${EXPDIS_VLLM_RELOAD_TPU_NAME:?Must set EXPDIS_VLLM_RELOAD_TPU_NAME}" \
      ZONE="${EXPDIS_VLLM_RELOAD_ZONE:?Must set EXPDIS_VLLM_RELOAD_ZONE}" \
      WORKERS="$reload_workers" \
      SERVED_MODEL_NAME="$MODEL_NAME" \
      LOCAL_MODEL_DIR_BASENAME="${EXPDIS_ACTUAL_SFT_LOCAL_MODEL_BASENAME:-expdis_actual_sft_model}" \
      MAX_MODEL_LEN="${EXPDIS_ACTUAL_GRPO_MAX_MODEL_LEN:-$MAX_TOTAL_LEN}" \
      MAX_NUM_SEQS="${EXPDIS_ACTUAL_GRPO_VLLM_MAX_NUM_SEQS:-32}" \
      MAX_NUM_BATCHED_TOKENS="${EXPDIS_VLLM_RELOAD_MAX_BATCHED_TOKENS:-65536}" \
      PARALLELISM="${EXPDIS_VLLM_RELOAD_PARALLELISM:-4}" \
      RELOAD_TIMEOUT_SECONDS="${EXPDIS_VLLM_RELOAD_TIMEOUT_SECONDS:-1800}" \
      bash expdis_jax/scripts/reload_vllm_direct_slice.sh; then
      echo "[actual-resume] vLLM SFT reload succeeded $(date -Is)"
      break
    fi
    echo "[actual-resume] WARN vLLM SFT reload failed; retrying in ${EXPDIS_EXTERNAL_VLLM_RELOAD_RETRY_SECONDS:-60}s"
    sleep "${EXPDIS_EXTERNAL_VLLM_RELOAD_RETRY_SECONDS:-60}"
  done
fi

python -m expdis_jax.train \
  --model-name "$MODEL_NAME" \
  --dataset-name dapo_math_17k \
  --max-train-examples 20000 \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --max-total-len "$MAX_TOTAL_LEN" \
  --grpo-batch-size 1 \
  --grpo-grad-accum 2 \
  --grpo-num-generations "$GRPO_NUM_GENERATIONS" \
  --grpo-max-steps "${ACTUAL_ONLINE_GRPO_MAX_STEPS:-96}" \
  --grpo-loss-type dr_grpo \
  --optimizer adamw \
  --grpo-lr "${ACTUAL_GRPO_LR:-1e-6}" \
  --grpo-clip-epsilon "$GRPO_CLIP_EPSILON" \
  --grpo-clip-epsilon-high "$GRPO_CLIP_EPSILON_HIGH" \
  --grpo-kl-beta "${ACTUAL_GRPO_KL_COEF:-${GRPO_KL_BETA:-0.0}}" \
  --grpo-updates-per-rollout "${ACTUAL_GRPO_UPDATES_PER_ROLLOUT:-1}" \
  --grpo-advantage-normalization "$GRPO_ADVANTAGE_NORMALIZATION" \
  --trajectory-batch-size 2 \
  --vllm-server-urls "$VLLM_URLS" \
  --vllm-concurrency "${VLLM_CONCURRENCY:-16}" \
  --vllm-fanout-per-prompt "${VLLM_FANOUT_PER_PROMPT:-8}" \
  --dynamic-sampling "$DYNAMIC_SAMPLING" \
  --dynamic-sampling-max-attempts "$DYNAMIC_SAMPLING_MAX_ATTEMPTS" \
  --mask-truncated-completions "$MASK_TRUNCATED_COMPLETIONS" \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --enable-thinking auto \
  --reward-correct "$REWARD_CORRECT" \
  --reward-incorrect "$REWARD_INCORRECT" \
  --format-penalty-weight 0.0 \
  --nontermination-penalty "$NONTERMINATION_PENALTY" \
  --invalid-answer-penalty "$INVALID_ANSWER_PENALTY" \
  --soft-overlong-expected-len "$SOFT_OVERLONG_EXPECTED_LEN" \
  --soft-overlong-cache-len "$SOFT_OVERLONG_CACHE_LEN" \
  --lambda-novelty 0.0 \
  --novelty-metric sqrt_mse \
  --novelty-clip 0.0 \
  --novelty-feature-source multilayer \
  --novelty-layers 7,14,21 \
  --novelty-layer-pool mean \
  --novelty-layer-agg mean \
  --novelty-text-contract full_raw \
  --novelty-feature-max-length "$NOVELTY_FEATURE_MAX_LENGTH" \
  --novelty-zscore off \
  --correctness-zscore off \
  --rnd-hidden 512 \
  --rnd-lr 1e-4 \
  --incorrect-novelty-scale 0.0 \
  --gate-novelty-by-quality false \
  --benchmark-num-rollouts 32 \
  --benchmark-rollout-max-completion-len 32768 \
  --rescore-chunk-size 1 \
  --logprob-chunk-tokens "${LOGPROB_CHUNK_TOKENS:-64}" \
  --init-checkpoint "$INIT_CKPT" \
  --output-dir "$ACTUAL_OUT" \
  --wandb-run-name "$RUN_NAME" \
  --save-every-steps "${SAVE_EVERY_STEPS:-50}" \
  --seed 0

FINAL_CKPT="$(find "$ACTUAL_OUT" -maxdepth 1 -type d -name 'step_*' | sort | tail -1)"
if [ -z "$FINAL_CKPT" ]; then
  echo "[actual-export] ERROR no actual checkpoint found in $ACTUAL_OUT" >&2
  exit 3
fi
export FINAL_CKPT
export ACTUAL_HF_OUT="${ACTUAL_HF_OUT:-$ACTUAL_DIR/actual_hf_final}"
export MODEL_NAME
export RUN_NAME

python - <<'PY'
import os
import jax
import numpy as np
import orbax.checkpoint as ocp
from huggingface_hub import HfApi

from expdis_jax.weights import export_flax_params_to_hf_dir, load_hf_config

ckpt = os.environ["FINAL_CKPT"]
out = os.environ["ACTUAL_HF_OUT"]
model_name = os.environ["MODEL_NAME"]
run = os.environ["RUN_NAME"]
print(f"[actual-export] restoring {ckpt}", flush=True)
payload = ocp.PyTreeCheckpointer().restore(os.path.abspath(ckpt))
params = jax.tree_util.tree_map(lambda x: np.asarray(x), payload["params"])
hf_cfg = load_hf_config(model_name)
export_flax_params_to_hf_dir(
    params,
    model_name,
    int(hf_cfg["num_hidden_layers"]),
    bool(hf_cfg.get("tie_word_embeddings", False)),
    out,
)
print(f"[actual-export] exported {out}", flush=True)
repo_id = os.environ.get("EXPDIS_HF_CHECKPOINT_REPO", "SaifPunjwani/two-model-exploration-checkpoints")
repo_type = os.environ.get("EXPDIS_HF_REPO_TYPE", "dataset")
path_in_repo = f"{run}/actual_hf_final"
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=out,
    path_in_repo=path_in_repo,
    commit_message=f"actual export {path_in_repo}",
)
print(f"[actual-export] uploaded hf://{repo_id}/{path_in_repo}", flush=True)
PY
