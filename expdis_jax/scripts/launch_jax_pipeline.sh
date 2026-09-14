#!/usr/bin/env bash
# Launch the full JAX pipeline: Explorer -> Main SFT -> Main RL -> AIME eval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-expdis_jax}"
if [ -n "${ENV_DIR:-}" ]; then
  if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "ERROR: ENV_DIR must contain bin/activate: $ENV_DIR" >&2
    exit 1
  fi
  source "$ENV_DIR/bin/activate" "$ENV_NAME"
elif [ -n "${VIRTUAL_ENV:-${CONDA_PREFIX:-}}" ] && [ -x "${VIRTUAL_ENV:-${CONDA_PREFIX:-}}/bin/python" ]; then
  export PATH="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}/bin:$PATH"
elif [ -f "$HOME/miniforge3/bin/activate" ]; then
  source "$HOME/miniforge3/bin/activate" "$ENV_NAME"
elif [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  source "$REPO_ROOT/.venv/bin/activate"
elif [ -f "$HOME/vllm_tpu_env/bin/activate" ]; then
  source "$HOME/vllm_tpu_env/bin/activate"
elif [ -x "$HOME/vllm_tpu_env/bin/python" ]; then
  export PATH="$HOME/vllm_tpu_env/bin:$PATH"
else
  echo "ERROR: activate a Python environment or set ENV_DIR to its directory" >&2
  exit 1
fi

LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.5}"
SCOUTS_PER_ROUND="${EXPLORERS_PER_ROUND:-${SCOUTS_PER_ROUND:-1}}"
RUN_NAME="${RUN_NAME:-jax_pipeline_lambda${LAMBDA_NOVELTY}_$(date +%Y%m%d_%H%M%S)}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/expdis/runs}"
OUT="$RUNS_ROOT/$RUN_NAME"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATASET_NAME="${DATASET_NAME:-${DATASET:-dapo_math_17k}}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-200}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-16}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-4}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-4}"
ACTUAL_ONLINE_GRPO_MAX_STEPS="${MAIN_RL_MAX_STEPS:-${ACTUAL_ONLINE_GRPO_MAX_STEPS:-100}}"
ACTUAL_GRPO_LR="${MAIN_RL_LR:-${ACTUAL_GRPO_LR:-1e-6}}"
ACTUAL_GRPO_KL_COEF="${ACTUAL_GRPO_KL_COEF:-0.0}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
MIN_ACCEPTED_TRAJECTORIES="${MIN_ACCEPTED_TRAJECTORIES:-1}"
MAX_ACCEPTED_TRAJECTORIES="${MAX_ACCEPTED_TRAJECTORIES:-500}"
ACCEPTED_SELECTION_POLICY="${ACCEPTED_SELECTION_POLICY:-coverage_pool_c8}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-16}"
VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
LOGPROB_CHUNK_TOKENS="${LOGPROB_CHUNK_TOKENS:-256}"
GRPO_LR="${GRPO_LR:-5e-6}"
DR_GRPO_LOSS_NORMALIZER="${DR_GRPO_LOSS_NORMALIZER:-32768}"
GRPO_CLIP_EPSILON="${GRPO_CLIP_EPSILON:-0.2}"
GRPO_CLIP_EPSILON_HIGH="${GRPO_CLIP_EPSILON_HIGH:-0.28}"  # DAPO asymmetric clip; validate_contract requires 0.28
GRPO_ADVANTAGE_NORMALIZATION="${GRPO_ADVANTAGE_NORMALIZATION:-none}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:--1.0}"
SOFT_OVERLONG_EXPECTED_LEN="${SOFT_OVERLONG_EXPECTED_LEN:-26214}"
SOFT_OVERLONG_CACHE_LEN="${SOFT_OVERLONG_CACHE_LEN:-6554}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
export EXPDIS_ARTIFACT_BACKEND="${EXPDIS_ARTIFACT_BACKEND:-hf}"
export EXPDIS_ALLOW_GCS_ARTIFACTS="${EXPDIS_ALLOW_GCS_ARTIFACTS:-0}"
export EXPDIS_HF_CHECKPOINT_REPO="${EXPDIS_HF_CHECKPOINT_REPO:-}"
export EXPDIS_HF_REPO_TYPE="${EXPDIS_HF_REPO_TYPE:-dataset}"
EXPDIS_GCS_CHECKPOINT_BASE="${EXPDIS_GCS_CHECKPOINT_BASE:-}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-2}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-1}"
MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-500}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-64}"
# Contract: 32k completions in both training and evaluation.
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-32768}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-32768}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-$((MAX_PROMPT_LEN + MAX_COMPLETION_LEN))}"
NOVELTY_FEATURE_MAX_LENGTH="${NOVELTY_FEATURE_MAX_LENGTH:-$MAX_TOTAL_LEN}"

VLLM_URLS="${EXPDIS_VLLM_SERVER_URLS:-${VLLM_SERVER_URLS:-}}"
if [ -z "$VLLM_URLS" ]; then
  echo "ERROR: set EXPDIS_VLLM_SERVER_URLS=http://ip:8000/v1,..." >&2
  exit 1
fi

if [ "${EXPDIS_VLLM_RELOAD_TRANSPORT:-hf}" = "hf" ] && [ -z "$EXPDIS_HF_CHECKPOINT_REPO" ] && [ -z "$EXPDIS_GCS_CHECKPOINT_BASE" ]; then
  echo "ERROR: set EXPDIS_HF_CHECKPOINT_REPO to a dataset repo you can write, or configure another reload transport" >&2
  exit 1
fi
for setting in EXPDIS_VLLM_RELOAD_TPU_NAME EXPDIS_VLLM_RELOAD_ZONE EXPDIS_VLLM_RELOAD_WORKERS; do
  if [ -z "${!setting:-}" ]; then
    echo "ERROR: set $setting for serving-weight synchronization; see expdis_jax/README.md" >&2
    exit 1
  fi
done
mkdir -p "$OUT"
printf '%s\n' "$RUN_NAME" > "$HOME/active_jax_run_name"

export WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
if [ -n "${WANDB_ENTITY:-}" ]; then export WANDB_ENTITY; fi  # unset = the W&B default entity
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_key" ]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_key")"
fi
if [ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -z "${EXPDIS_WANDB_RUN_ID:-}" ] && [ -f "$OUT/.wandb_run_id" ]; then
  export EXPDIS_WANDB_RUN_ID="$(tr -d '[:space:]' < "$OUT/.wandb_run_id")"
fi
if [ -z "${EXPDIS_WANDB_RUN_ID:-}" ] && [ -f "$OUT/launcher.log" ]; then
  _existing_wandb_id="$(
    grep -oE 'https://wandb.ai/[^ ]+/runs/[^ ]+' "$OUT/launcher.log" 2>/dev/null \
      | sed -E 's#.*/runs/##' \
      | grep -v '^jaxpipe-' \
      | tail -1 || true
  )"
  if [ -n "$_existing_wandb_id" ]; then
    export EXPDIS_WANDB_RUN_ID="$_existing_wandb_id"
  fi
fi
if [ -z "${EXPDIS_WANDB_RUN_ID:-}" ]; then
  export EXPDIS_WANDB_RUN_ID="$(
    python - "$RUN_NAME" <<'PY'
import hashlib
import sys
run_name = sys.argv[1]
print("jaxe2e-" + hashlib.sha1(f"{run_name}:e2e".encode("utf-8")).hexdigest()[:24])
PY
  )"
fi
printf '%s\n' "$EXPDIS_WANDB_RUN_ID" > "$OUT/.wandb_run_id"
export EXPDIS_WANDB_EXPLORER_RUN_ID="${EXPDIS_WANDB_EXPLORER_RUN_ID:-$EXPDIS_WANDB_RUN_ID}"
export EXPDIS_WANDB_DISTILL_RUN_ID="${EXPDIS_WANDB_DISTILL_RUN_ID:-$EXPDIS_WANDB_RUN_ID}"
export EXPDIS_WANDB_ACTUAL_RUN_ID="${EXPDIS_WANDB_ACTUAL_RUN_ID:-$EXPDIS_WANDB_RUN_ID}"
export EXPDIS_WANDB_ACTUAL_GRPO_RUN_ID="${EXPDIS_WANDB_ACTUAL_GRPO_RUN_ID:-$EXPDIS_WANDB_RUN_ID}"
export EXPDIS_WANDB_GROUP="${EXPDIS_WANDB_GROUP:-$RUN_NAME}"
export EXPDIS_WANDB_COMPACT="${EXPDIS_WANDB_COMPACT:-1}"
export EXPDIS_WANDB_MINIMAL="${EXPDIS_WANDB_MINIMAL:-1}"
export EXPDIS_WANDB_EVAL_PROGRESS="${EXPDIS_WANDB_EVAL_PROGRESS:-0}"
export EXPDIS_HF_ARTIFACT_PREFIX="${EXPDIS_HF_ARTIFACT_PREFIX:-$RUN_NAME}"
export EXPDIS_HF_MIRROR_CHECKPOINTS="${EXPDIS_HF_MIRROR_CHECKPOINTS:-1}"
export EXPDIS_HF_MIRROR_TRAJECTORIES="${EXPDIS_HF_MIRROR_TRAJECTORIES:-1}"
export EXPDIS_HF_MIRROR_METRICS="${EXPDIS_HF_MIRROR_METRICS:-1}"
export EXPDIS_HF_MIRROR_VLLM_EXPORTS="${EXPDIS_HF_MIRROR_VLLM_EXPORTS:-1}"
export EXPDIS_HF_MIRROR_EVAL_PROBES="${EXPDIS_HF_MIRROR_EVAL_PROBES:-1}"
export EXPDIS_HF_MIRROR_RUN_SUMMARY="${EXPDIS_HF_MIRROR_RUN_SUMMARY:-1}"
export EXPDIS_HF_MIRROR_FINAL_EXPORTS="${EXPDIS_HF_MIRROR_FINAL_EXPORTS:-1}"
export EXPDIS_HF_MIRROR_FINAL_EVAL="${EXPDIS_HF_MIRROR_FINAL_EVAL:-1}"
export EXPDIS_HF_MIRROR_ASYNC="${EXPDIS_HF_MIRROR_ASYNC:-1}"
export EXPDIS_REQUIRE_HF_ARTIFACT_UPLOAD="${EXPDIS_REQUIRE_HF_ARTIFACT_UPLOAD:-0}"
# Generate candidate prompt-groups incrementally. This preserves the same
# max_attempts budget while avoiding over-generating long completions up front.
export EXPDIS_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-1}"
case "${EXPDIS_REQUIRE_WANDB:-1}" in
  0|false|False|no|off) ;;
  *)
    python - <<'PY'
import wandb
wandb.Api(timeout=10)
print("[wandb] credentials OK", flush=True)
PY
    ;;
esac
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [ "${EXPDIS_ENABLE_JAX_COMPILATION_CACHE:-1}" != "0" ]; then
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/expdis_jax_compilation_cache}"
  # Lowered from 5s to 1s so smaller kernels (RND step, novelty feature_step,
  # rescore_jit) get persisted across resumes too — saves 60-180s on cold start.
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-1}"
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
fi
# NOTE: tried adding XLA_FLAGS for async collective fusion but the flag names
# in this JAX/XLA build are not recognized (causes SIGABRT). Leaving unset;
# JAX defaults handle async collectives on v5lite/v6e.
# Optimizer state is huge for Qwen3-1.7B AdamW and is not needed for the
# research artifacts we consume downstream (Explorer params/RND state/trajectories,
# SFT, Actual GRPO). Saving it filled v5 root disks at step-50 checkpoints.
export EXPDIS_SAVE_OPT_STATE="${EXPDIS_SAVE_OPT_STATE:-0}"
export EXPDIS_CKPT_DELETE_OLD_RUNS="${EXPDIS_CKPT_DELETE_OLD_RUNS:-0}"
export EXPDIS_LOCAL_CHECKPOINT_EVERY_STEPS="${EXPDIS_LOCAL_CHECKPOINT_EVERY_STEPS:-3}"
export EXPDIS_GCS_CHECKPOINT_EVERY_STEPS="${EXPDIS_GCS_CHECKPOINT_EVERY_STEPS:-$SAVE_EVERY_STEPS}"
export EXPDIS_LOCAL_CHECKPOINT_KEEP_LAST="${EXPDIS_LOCAL_CHECKPOINT_KEEP_LAST:-1}"
export EXPDIS_LOCAL_VLLM_RELOAD_KEEP_LAST="${EXPDIS_LOCAL_VLLM_RELOAD_KEEP_LAST:-0}"
export EXPDIS_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD="${EXPDIS_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD:-1}"
export EXPDIS_CKPT_MIN_FREE_GB="${EXPDIS_CKPT_MIN_FREE_GB:-28}"
export EXPDIS_VLLM_EXPORT_MIN_FREE_GB="${EXPDIS_VLLM_EXPORT_MIN_FREE_GB:-28}"
export EXPDIS_PREFETCH_ROLLOUTS="${EXPDIS_PREFETCH_ROLLOUTS:-1}"
# Paper protocol: synchronize the sampler before every rollout batch (independent of SAVE_EVERY_STEPS).
export EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS="${EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS:-1}"
export EXPDIS_TRAIN_VLLM_RELOAD_ON_RESUME="${EXPDIS_TRAIN_VLLM_RELOAD_ON_RESUME:-1}"
export EXPDIS_TRAIN_VLLM_RELOAD_MODE="${EXPDIS_TRAIN_VLLM_RELOAD_MODE:-external}"
export EXPDIS_EXTERNAL_VLLM_RELOAD_WAIT="${EXPDIS_EXTERNAL_VLLM_RELOAD_WAIT:-1}"
export EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED="${EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED:-1}"
export EXPDIS_EXTERNAL_VLLM_RELOAD_TIMEOUT="${EXPDIS_EXTERNAL_VLLM_RELOAD_TIMEOUT:-0}"
export EXPDIS_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL="${EXPDIS_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL:-1}"
export EXPDIS_VLLM_EXPORT_FROM_LIVE_ALLGATHER="${EXPDIS_VLLM_EXPORT_FROM_LIVE_ALLGATHER:-1}"
export EXPDIS_VLLM_EXPORT_FROM_GCS_CHECKPOINT="${EXPDIS_VLLM_EXPORT_FROM_GCS_CHECKPOINT:-0}"
# Lightweight live eval trend. This runs only after a successful current-policy
# vLLM reload, so it measures the checkpoint being used for subsequent rollouts.
# The final AIME24 benchmark defaults to 30 problems x 64 rollouts x 32k.
export EXPDIS_AIME_PROBE_EVERY_STEPS="${EXPDIS_AIME_PROBE_EVERY_STEPS:-$SAVE_EVERY_STEPS}"
export EXPDIS_AIME_PROBE_NUM_PROBLEMS="${EXPDIS_AIME_PROBE_NUM_PROBLEMS:-6}"
export EXPDIS_AIME_PROBE_NUM_ROLLOUTS="${EXPDIS_AIME_PROBE_NUM_ROLLOUTS:-4}"
export EXPDIS_AIME_PROBE_MAX_TOKENS="${EXPDIS_AIME_PROBE_MAX_TOKENS:-$MAX_COMPLETION_LEN}"
export EXPDIS_AIME_PROBE_CONCURRENCY="${EXPDIS_AIME_PROBE_CONCURRENCY:-8}"
export EXPDIS_WANDB_SINGLE_RUN="${EXPDIS_WANDB_SINGLE_RUN:-1}"
export EXPDIS_VLLM_RETURN_LOGPROBS="${EXPDIS_VLLM_RETURN_LOGPROBS:-0}"
export EXPDIS_TRACK_EXTERNAL_OLD_GAP="${EXPDIS_TRACK_EXTERNAL_OLD_GAP:-0}"
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu}"
if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
  tpu_library_path="$(python - <<'PY' 2>/dev/null || true
import os
try:
    import libtpu
    path = os.path.join(os.path.dirname(libtpu.__file__), "libtpu.so")
    if os.path.exists(path):
        print(path)
except Exception:
    pass
PY
)"
  if [ -n "$tpu_library_path" ] && [ -f "$tpu_library_path" ]; then
    export TPU_LIBRARY_PATH="$tpu_library_path"
  fi
fi
if [ "${JAX_NUM_PROCESSES:-1}" != "1" ]; then
  if [ -n "${TPU_WORKER_HOSTNAMES:-}" ] && [ -n "${TPU_WORKER_ID:-}" ]; then
    export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
  else
    unset TPU_VISIBLE_CHIPS TPU_PROCESS_BOUNDS TPU_HOST_BOUNDS TPU_WORKER_ID TPU_WORKER_HOSTNAMES
    unset TPU_CHIPS_PER_HOST_BOUNDS TPU_CHIPS_PER_PROCESS_BOUNDS JAX_LOCAL_DEVICE_IDS
  fi
  if [ -n "${EXPDIS_JAX_CLUSTER_DETECTION_METHOD:-}" ]; then
    export EXPDIS_JAX_CLUSTER_DETECTION_METHOD
  else
    unset EXPDIS_JAX_CLUSTER_DETECTION_METHOD
  fi
else
  export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0,1,2,3}"
  export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
  export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
  export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
  export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
fi
export EXPDIS_GCS_CHECKPOINT_BASE
export EXPDIS_PIPELINE_STAGE="${EXPDIS_PIPELINE_STAGE:-explorer_novelty}"
export EXPDIS_PIPELINE_STOP_AFTER_EXPLORER="${EXPDIS_PIPELINE_STOP_AFTER_EXPLORER:-0}"
export EXPDIS_PIPELINE_STOP_AFTER_SFT="${EXPDIS_PIPELINE_STOP_AFTER_SFT:-0}"
export EXPDIS_PIPELINE_SKIP_EXPLORER="${EXPDIS_PIPELINE_SKIP_EXPLORER:-0}"
export EXPDIS_EXISTING_TRAJECTORIES="${EXPDIS_EXISTING_TRAJECTORIES:-}"
export EXPDIS_VLLM_RANDOMIZE_SEEDS="${EXPDIS_VLLM_RANDOMIZE_SEEDS:-0}"
export EXPDIS_VLLM_SEED_MODE="${EXPDIS_VLLM_SEED_MODE:-api}"
export EXPDIS_VLLM_SEED_API_FALLBACK="${EXPDIS_VLLM_SEED_API_FALLBACK:-1}"
export EXPDIS_VLLM_SEED_BASE="${EXPDIS_VLLM_SEED_BASE:-}"
export EXPDIS_VLLM_FILTER_DEAD_SERVERS="${EXPDIS_VLLM_FILTER_DEAD_SERVERS:-1}"
export EXPDIS_VLLM_HEALTH_FILTER_TIMEOUT="${EXPDIS_VLLM_HEALTH_FILTER_TIMEOUT:-2}"
export EXPDIS_VLLM_HEALTH_FILTER_TTL_SECONDS="${EXPDIS_VLLM_HEALTH_FILTER_TTL_SECONDS:-30}"
export EXPDIS_VLLM_HEALTH_FILTER_CONCURRENCY="${EXPDIS_VLLM_HEALTH_FILTER_CONCURRENCY:-64}"
export EXPDIS_VLLM_FAILOVER_ROUNDS="${EXPDIS_VLLM_FAILOVER_ROUNDS:-2}"
export EXPDIS_VLLM_FAILOVER_ROUND_SLEEP="${EXPDIS_VLLM_FAILOVER_ROUND_SLEEP:-10}"

# Align validate_contract's expected geometry with this launcher's actual geometry
# (otherwise the contract check expects 16 generations / grad_accum 1 and aborts).
export EXPDIS_EXPECTED_GRPO_NUM_GENERATIONS="${EXPDIS_EXPECTED_GRPO_NUM_GENERATIONS:-$GRPO_NUM_GENERATIONS}"
export EXPDIS_EXPECTED_GRPO_BATCH_SIZE="${EXPDIS_EXPECTED_GRPO_BATCH_SIZE:-$GRPO_BATCH_SIZE}"
export EXPDIS_EXPECTED_GRPO_GRAD_ACCUM="${EXPDIS_EXPECTED_GRPO_GRAD_ACCUM:-$GRPO_GRAD_ACCUM}"
export EXPDIS_EXPECTED_GRPO_UPDATES_PER_ROLLOUT="${EXPDIS_EXPECTED_GRPO_UPDATES_PER_ROLLOUT:-${GRPO_UPDATES_PER_ROLLOUT:-1}}"
export EXPDIS_DAPO_DRGRPO_FULL_CONTRACT="${EXPDIS_DAPO_DRGRPO_FULL_CONTRACT:-1}"
export EXPDIS_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN="${EXPDIS_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN:-$MAX_COMPLETION_LEN}"
export EXPDIS_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN="${EXPDIS_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN:-$SOFT_OVERLONG_EXPECTED_LEN}"
export EXPDIS_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN="${EXPDIS_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN:-$SOFT_OVERLONG_CACHE_LEN}"
export EXPDIS_DAPO_DRGRPO_CONTRACT_LOSS_NORMALIZER="${EXPDIS_DAPO_DRGRPO_CONTRACT_LOSS_NORMALIZER:-$DR_GRPO_LOSS_NORMALIZER}"

PIPELINE_MODE="${PIPELINE_MODE:-two_model}"
if [ "${NUM_ROUNDS:-1}" != "1" ] || [ "${SCOUTS_PER_ROUND:-1}" != "1" ] || [ -n "${ROUND_NOVELTY_SCHEDULE:-}" ]; then
  PIPELINE_MODE=multi_round
fi
python -m expdis_jax.pipeline \
  --pipeline-mode "$PIPELINE_MODE" \
  --num-rounds "${NUM_ROUNDS:-1}" \
  --explorers-per-round "$SCOUTS_PER_ROUND" \
  --round-novelty-schedule "${ROUND_NOVELTY_SCHEDULE:-}" \
  --model-name "$MODEL_NAME" \
  --dataset-name "$DATASET_NAME" \
  --max-train-examples "$MAX_TRAIN_EXAMPLES" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --max-total-len "$MAX_TOTAL_LEN" \
  --grpo-batch-size "$GRPO_BATCH_SIZE" \
  --grpo-grad-accum "$GRPO_GRAD_ACCUM" \
  --grpo-num-generations "$GRPO_NUM_GENERATIONS" \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --grpo-loss-type dr_grpo \
  --dr-grpo-loss-normalizer "$DR_GRPO_LOSS_NORMALIZER" \
  --optimizer adamw \
  --grpo-lr "$GRPO_LR" \
  --grpo-clip-epsilon "$GRPO_CLIP_EPSILON" \
  --grpo-clip-epsilon-high "$GRPO_CLIP_EPSILON_HIGH" \
  --grpo-kl-beta "${GRPO_KL_BETA:-0.0}" \
  --grpo-updates-per-rollout "${GRPO_UPDATES_PER_ROLLOUT:-1}" \
  --grpo-advantage-normalization "$GRPO_ADVANTAGE_NORMALIZATION" \
  --trajectory-batch-size "$TRAJECTORY_BATCH_SIZE" \
  --vllm-server-urls "$VLLM_URLS" \
  --vllm-concurrency "$VLLM_CONCURRENCY" \
  --vllm-fanout-per-prompt "$VLLM_FANOUT_PER_PROMPT" \
  --dynamic-sampling "${DYNAMIC_SAMPLING:-true}" \
  --dynamic-sampling-max-attempts "${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-8}" \
  --mask-truncated-completions "${MASK_TRUNCATED_COMPLETIONS:-true}" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --enable-thinking auto \
  --reward-correct "$REWARD_CORRECT" \
  --reward-incorrect "$REWARD_INCORRECT" \
  --format-penalty-weight 0.0 \
  --nontermination-penalty "${NONTERMINATION_PENALTY:-0.0}" \
  --invalid-answer-penalty "${INVALID_ANSWER_PENALTY:-0.0}" \
  --soft-overlong-expected-len "$SOFT_OVERLONG_EXPECTED_LEN" \
  --soft-overlong-cache-len "$SOFT_OVERLONG_CACHE_LEN" \
  --lambda-novelty "$LAMBDA_NOVELTY" \
  --novelty-metric "${NOVELTY_METRIC:-sqrt_mse}" \
  --novelty-clip "${NOVELTY_CLIP:-0.0}" \
  --novelty-feature-source "${NOVELTY_FEATURE_SOURCE:-multilayer}" \
  --novelty-layers "${NOVELTY_LAYERS:-auto}" \
  --novelty-layer-pool "${NOVELTY_LAYER_POOL:-mean}" \
  --novelty-layer-agg "${NOVELTY_LAYER_AGG:-mean}" \
  --novelty-text-contract "${NOVELTY_TEXT_CONTRACT:-full_raw}" \
  --novelty-feature-max-length "$NOVELTY_FEATURE_MAX_LENGTH" \
  --novelty-zscore "${NOVELTY_ZSCORE:-off}" \
  --correctness-zscore "${CORRECTNESS_ZSCORE:-off}" \
  --rnd-hidden "${RND_HIDDEN:-512}" \
  --rnd-lr "${RND_LR:-1e-4}" \
  --incorrect-novelty-scale "${INCORRECT_NOVELTY_SCALE:-0.0}" \
  --gate-novelty-by-quality "${GATE_NOVELTY_BY_QUALITY:-false}" \
  --distill-epochs "$DISTILL_EPOCHS" \
  --distill-batch-size "$DISTILL_BATCH_SIZE" \
  --max-distill-examples "$MAX_DISTILL_EXAMPLES" \
  --min-accepted-trajectories "$MIN_ACCEPTED_TRAJECTORIES" \
  --max-accepted-trajectories "$MAX_ACCEPTED_TRAJECTORIES" \
  --accepted-selection-policy "$ACCEPTED_SELECTION_POLICY" \
  --main-rl-max-steps "$ACTUAL_ONLINE_GRPO_MAX_STEPS" \
  --main-rl-lr "$ACTUAL_GRPO_LR" \
  --main-rl-kl-coef "$ACTUAL_GRPO_KL_COEF" \
  --benchmark-num-rollouts "$BENCHMARK_NUM_ROLLOUTS" \
  --benchmark-rollout-max-completion-len "$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN" \
  --rescore-chunk-size 1 \
  --logprob-chunk-tokens "$LOGPROB_CHUNK_TOKENS" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output-dir "$OUT" \
  --checkpoint-root "${CHECKPOINT_ROOT:-}" \
  --wandb-run-name "$RUN_NAME" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --seed "${SEED:-0}" \
  2>&1 | tee -a "$OUT/console.log"
