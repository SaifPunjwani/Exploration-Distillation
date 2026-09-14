#!/usr/bin/env bash
# Launch the full JAX pipeline: Explorer -> SFT -> Actual GRPO -> AIME eval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-tmx_jax}"
if [ -n "${ENV_DIR:-}" ] && [ -f "$ENV_DIR/bin/activate" ]; then
  source "$ENV_DIR/bin/activate" "$ENV_NAME"
elif [ -f "$HOME/miniforge3/bin/activate" ]; then
  source "$HOME/miniforge3/bin/activate" "$ENV_NAME"
elif [ -f "$HOME/vllm_tpu_env/bin/activate" ]; then
  source "$HOME/vllm_tpu_env/bin/activate"
elif [ -x "$HOME/vllm_tpu_env/bin/python" ]; then
  export PATH="$HOME/vllm_tpu_env/bin:$PATH"
else
  echo "ERROR: no usable Python env found; set ENV_DIR or install ~/miniforge3 or ~/vllm_tpu_env" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-jax_pipeline_lambda0_$(date +%Y%m%d_%H%M%S)}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/two-model-exploration/runs}"
OUT="$RUNS_ROOT/$RUN_NAME"
mkdir -p "$OUT"
printf '%s\n' "$RUN_NAME" > "$HOME/active_jax_run_name"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATASET_NAME="${DATASET_NAME:-${DATASET:-dapo_math_17k}}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-500}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-8}"
TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-2}"
ACTUAL_ONLINE_GRPO_MAX_STEPS="${ACTUAL_ONLINE_GRPO_MAX_STEPS:-250}"
ACTUAL_GRPO_LR="${ACTUAL_GRPO_LR:-1e-6}"
ACTUAL_GRPO_KL_COEF="${ACTUAL_GRPO_KL_COEF:-0.10}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
MIN_ACCEPTED_TRAJECTORIES="${MIN_ACCEPTED_TRAJECTORIES:-2048}"
MAX_ACCEPTED_TRAJECTORIES="${MAX_ACCEPTED_TRAJECTORIES:-0}"
ACCEPTED_SELECTION_POLICY="${ACCEPTED_SELECTION_POLICY:-quality_first}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-16}"
VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
LOGPROB_CHUNK_TOKENS="${LOGPROB_CHUNK_TOKENS:-256}"
GRPO_LR="${GRPO_LR:-5e-6}"
GRPO_CLIP_EPSILON="${GRPO_CLIP_EPSILON:-0.2}"
GRPO_CLIP_EPSILON_HIGH="${GRPO_CLIP_EPSILON_HIGH:-$GRPO_CLIP_EPSILON}"
GRPO_ADVANTAGE_NORMALIZATION="${GRPO_ADVANTAGE_NORMALIZATION:-none}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:-0.0}"
SOFT_OVERLONG_EXPECTED_LEN="${SOFT_OVERLONG_EXPECTED_LEN:-0}"
SOFT_OVERLONG_CACHE_LEN="${SOFT_OVERLONG_CACHE_LEN:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
TMX_ARTIFACT_BACKEND="${TMX_ARTIFACT_BACKEND:-hf}"
TMX_ALLOW_GCS_ARTIFACTS="${TMX_ALLOW_GCS_ARTIFACTS:-0}"
TMX_HF_CHECKPOINT_REPO="${TMX_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}"
TMX_HF_REPO_TYPE="${TMX_HF_REPO_TYPE:-dataset}"
TMX_GCS_CHECKPOINT_BASE="${TMX_GCS_CHECKPOINT_BASE:-}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-2}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-1}"
MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-20000}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
# Contract: 16k completion for both training and mid-training eval probes so
# train/eval distributions match (no clipping shift). Standalone Qwen3 paper
# baselines run at 32k via baseline_runs/ — they are NOT this rollout probe.
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-32768}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-16384}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-$((MAX_PROMPT_LEN + MAX_COMPLETION_LEN))}"
NOVELTY_FEATURE_MAX_LENGTH="${NOVELTY_FEATURE_MAX_LENGTH:-$MAX_TOTAL_LEN}"

VLLM_URLS="${TMX_VLLM_SERVER_URLS:-${VLLM_SERVER_URLS:-}}"
if [ -z "$VLLM_URLS" ]; then
  echo "ERROR: set TMX_VLLM_SERVER_URLS=http://ip:8000/v1,..." >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_key" ]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_key")"
fi
if [ -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -z "${TMX_WANDB_RUN_ID:-}" ] && [ -f "$OUT/.wandb_run_id" ]; then
  export TMX_WANDB_RUN_ID="$(tr -d '[:space:]' < "$OUT/.wandb_run_id")"
fi
if [ -z "${TMX_WANDB_RUN_ID:-}" ] && [ -f "$OUT/launcher.log" ]; then
  _existing_wandb_id="$(
    grep -oE 'https://wandb.ai/[^ ]+/runs/[^ ]+' "$OUT/launcher.log" 2>/dev/null \
      | sed -E 's#.*/runs/##' \
      | grep -v '^jaxpipe-' \
      | tail -1 || true
  )"
  if [ -n "$_existing_wandb_id" ]; then
    export TMX_WANDB_RUN_ID="$_existing_wandb_id"
  fi
fi
if [ -z "${TMX_WANDB_RUN_ID:-}" ]; then
  export TMX_WANDB_RUN_ID="$(
    python - "$RUN_NAME" <<'PY'
import hashlib
import sys
run_name = sys.argv[1]
print("jaxe2e-" + hashlib.sha1(f"{run_name}:e2e".encode("utf-8")).hexdigest()[:24])
PY
  )"
fi
printf '%s\n' "$TMX_WANDB_RUN_ID" > "$OUT/.wandb_run_id"
export TMX_WANDB_EXPLORER_RUN_ID="${TMX_WANDB_EXPLORER_RUN_ID:-$TMX_WANDB_RUN_ID}"
export TMX_WANDB_DISTILL_RUN_ID="${TMX_WANDB_DISTILL_RUN_ID:-$TMX_WANDB_RUN_ID}"
export TMX_WANDB_ACTUAL_RUN_ID="${TMX_WANDB_ACTUAL_RUN_ID:-$TMX_WANDB_RUN_ID}"
export TMX_WANDB_ACTUAL_GRPO_RUN_ID="${TMX_WANDB_ACTUAL_GRPO_RUN_ID:-$TMX_WANDB_RUN_ID}"
export TMX_WANDB_GROUP="${TMX_WANDB_GROUP:-$RUN_NAME}"
export TMX_WANDB_COMPACT="${TMX_WANDB_COMPACT:-1}"
export TMX_WANDB_MINIMAL="${TMX_WANDB_MINIMAL:-1}"
export TMX_WANDB_EVAL_PROGRESS="${TMX_WANDB_EVAL_PROGRESS:-0}"
export TMX_HF_CHECKPOINT_REPO="${TMX_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}"
export TMX_HF_ARTIFACT_PREFIX="${TMX_HF_ARTIFACT_PREFIX:-$RUN_NAME}"
export TMX_HF_MIRROR_CHECKPOINTS="${TMX_HF_MIRROR_CHECKPOINTS:-1}"
export TMX_HF_MIRROR_TRAJECTORIES="${TMX_HF_MIRROR_TRAJECTORIES:-1}"
export TMX_HF_MIRROR_METRICS="${TMX_HF_MIRROR_METRICS:-1}"
export TMX_HF_MIRROR_VLLM_EXPORTS="${TMX_HF_MIRROR_VLLM_EXPORTS:-1}"
export TMX_HF_MIRROR_EVAL_PROBES="${TMX_HF_MIRROR_EVAL_PROBES:-1}"
export TMX_HF_MIRROR_RUN_SUMMARY="${TMX_HF_MIRROR_RUN_SUMMARY:-1}"
export TMX_HF_MIRROR_FINAL_EXPORTS="${TMX_HF_MIRROR_FINAL_EXPORTS:-1}"
export TMX_HF_MIRROR_FINAL_EVAL="${TMX_HF_MIRROR_FINAL_EVAL:-1}"
export TMX_HF_MIRROR_ASYNC="${TMX_HF_MIRROR_ASYNC:-1}"
export TMX_REQUIRE_HF_ARTIFACT_UPLOAD="${TMX_REQUIRE_HF_ARTIFACT_UPLOAD:-0}"
# Generate candidate prompt-groups incrementally. This preserves the same
# max_attempts budget while avoiding over-generating long completions up front.
export TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-1}"
case "${TMX_REQUIRE_WANDB:-1}" in
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
if [ "${TMX_ENABLE_JAX_COMPILATION_CACHE:-1}" != "0" ]; then
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/tmx_jax_compilation_cache}"
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
export TMX_SAVE_OPT_STATE="${TMX_SAVE_OPT_STATE:-0}"
export TMX_CKPT_DELETE_OLD_RUNS="${TMX_CKPT_DELETE_OLD_RUNS:-1}"
export TMX_LOCAL_CHECKPOINT_EVERY_STEPS="${TMX_LOCAL_CHECKPOINT_EVERY_STEPS:-3}"
export TMX_GCS_CHECKPOINT_EVERY_STEPS="${TMX_GCS_CHECKPOINT_EVERY_STEPS:-$SAVE_EVERY_STEPS}"
export TMX_LOCAL_CHECKPOINT_KEEP_LAST="${TMX_LOCAL_CHECKPOINT_KEEP_LAST:-1}"
export TMX_LOCAL_VLLM_RELOAD_KEEP_LAST="${TMX_LOCAL_VLLM_RELOAD_KEEP_LAST:-0}"
export TMX_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD="${TMX_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD:-1}"
export TMX_CKPT_MIN_FREE_GB="${TMX_CKPT_MIN_FREE_GB:-28}"
export TMX_VLLM_EXPORT_MIN_FREE_GB="${TMX_VLLM_EXPORT_MIN_FREE_GB:-28}"
export TMX_PREFETCH_ROLLOUTS="${TMX_PREFETCH_ROLLOUTS:-1}"
export TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS="${TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS:-$SAVE_EVERY_STEPS}"
export TMX_TRAIN_VLLM_RELOAD_ON_RESUME="${TMX_TRAIN_VLLM_RELOAD_ON_RESUME:-1}"
export TMX_TRAIN_VLLM_RELOAD_MODE="${TMX_TRAIN_VLLM_RELOAD_MODE:-external}"
export TMX_EXTERNAL_VLLM_RELOAD_WAIT="${TMX_EXTERNAL_VLLM_RELOAD_WAIT:-1}"
export TMX_EXTERNAL_VLLM_RELOAD_REQUIRED="${TMX_EXTERNAL_VLLM_RELOAD_REQUIRED:-1}"
export TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT="${TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT:-0}"
export TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL="${TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL:-1}"
export TMX_VLLM_EXPORT_FROM_LIVE_ALLGATHER="${TMX_VLLM_EXPORT_FROM_LIVE_ALLGATHER:-1}"
export TMX_VLLM_EXPORT_FROM_GCS_CHECKPOINT="${TMX_VLLM_EXPORT_FROM_GCS_CHECKPOINT:-0}"
# Lightweight live eval trend. This runs only after a successful current-policy
# vLLM reload, so it measures the checkpoint being used for subsequent rollouts.
# The official final benchmark remains 30 AIME24 problems x 32 rollouts x 32k.
export TMX_AIME_PROBE_EVERY_STEPS="${TMX_AIME_PROBE_EVERY_STEPS:-$SAVE_EVERY_STEPS}"
export TMX_AIME_PROBE_NUM_PROBLEMS="${TMX_AIME_PROBE_NUM_PROBLEMS:-6}"
export TMX_AIME_PROBE_NUM_ROLLOUTS="${TMX_AIME_PROBE_NUM_ROLLOUTS:-4}"
export TMX_AIME_PROBE_MAX_TOKENS="${TMX_AIME_PROBE_MAX_TOKENS:-$MAX_COMPLETION_LEN}"
export TMX_AIME_PROBE_CONCURRENCY="${TMX_AIME_PROBE_CONCURRENCY:-8}"
export TMX_WANDB_SINGLE_RUN="${TMX_WANDB_SINGLE_RUN:-1}"
export TMX_VLLM_RETURN_LOGPROBS="${TMX_VLLM_RETURN_LOGPROBS:-0}"
export TMX_TRACK_EXTERNAL_OLD_GAP="${TMX_TRACK_EXTERNAL_OLD_GAP:-0}"
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
  if [ -n "${TMX_JAX_CLUSTER_DETECTION_METHOD:-}" ]; then
    export TMX_JAX_CLUSTER_DETECTION_METHOD
  else
    unset TMX_JAX_CLUSTER_DETECTION_METHOD
  fi
else
  export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0,1,2,3}"
  export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
  export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
  export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
  export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
fi
export TMX_GCS_CHECKPOINT_BASE
export TMX_PIPELINE_STAGE="${TMX_PIPELINE_STAGE:-explorer_novelty}"
export TMX_PIPELINE_STOP_AFTER_EXPLORER="${TMX_PIPELINE_STOP_AFTER_EXPLORER:-0}"
export TMX_PIPELINE_STOP_AFTER_SFT="${TMX_PIPELINE_STOP_AFTER_SFT:-0}"
export TMX_PIPELINE_SKIP_EXPLORER="${TMX_PIPELINE_SKIP_EXPLORER:-0}"
export TMX_EXISTING_TRAJECTORIES="${TMX_EXISTING_TRAJECTORIES:-}"
export TMX_VLLM_RANDOMIZE_SEEDS="${TMX_VLLM_RANDOMIZE_SEEDS:-0}"
export TMX_VLLM_SEED_MODE="${TMX_VLLM_SEED_MODE:-api}"
export TMX_VLLM_SEED_API_FALLBACK="${TMX_VLLM_SEED_API_FALLBACK:-1}"
export TMX_VLLM_SEED_BASE="${TMX_VLLM_SEED_BASE:-}"
export TMX_VLLM_FILTER_DEAD_SERVERS="${TMX_VLLM_FILTER_DEAD_SERVERS:-1}"
export TMX_VLLM_HEALTH_FILTER_TIMEOUT="${TMX_VLLM_HEALTH_FILTER_TIMEOUT:-2}"
export TMX_VLLM_HEALTH_FILTER_TTL_SECONDS="${TMX_VLLM_HEALTH_FILTER_TTL_SECONDS:-30}"
export TMX_VLLM_HEALTH_FILTER_CONCURRENCY="${TMX_VLLM_HEALTH_FILTER_CONCURRENCY:-64}"
export TMX_VLLM_FAILOVER_ROUNDS="${TMX_VLLM_FAILOVER_ROUNDS:-2}"
export TMX_VLLM_FAILOVER_ROUND_SLEEP="${TMX_VLLM_FAILOVER_ROUND_SLEEP:-10}"
export TMX_VLLM_FANOUT_DESYNC="${TMX_VLLM_FANOUT_DESYNC:-0}"
export TMX_VLLM_FANOUT_NONCE="${TMX_VLLM_FANOUT_NONCE:-0}"
export TMX_VLLM_FANOUT_NONCE_MODE="${TMX_VLLM_FANOUT_NONCE_MODE:-instruction}"
export TMX_VLLM_FANOUT_NONCE_TEMPLATE="${TMX_VLLM_FANOUT_NONCE_TEMPLATE:-}"

python -m tmx_jax.pipeline \
  --model-name "$MODEL_NAME" \
  --dataset-name "$DATASET_NAME" \
  --max-train-examples "$MAX_TRAIN_EXAMPLES" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --max-total-len "$MAX_TOTAL_LEN" \
  --grpo-batch-size "${GRPO_BATCH_SIZE:-1}" \
  --grpo-grad-accum "${GRPO_GRAD_ACCUM:-2}" \
  --grpo-num-generations "$GRPO_NUM_GENERATIONS" \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --grpo-loss-type dr_grpo \
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
  --dynamic-sampling "${DYNAMIC_SAMPLING:-false}" \
  --dynamic-sampling-max-attempts "${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-8}" \
  --mask-truncated-completions "${MASK_TRUNCATED_COMPLETIONS:-false}" \
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
  --lambda-novelty "${LAMBDA_NOVELTY:-0.0}" \
  --novelty-metric "${NOVELTY_METRIC:-sqrt_mse}" \
  --novelty-clip "${NOVELTY_CLIP:-0.0}" \
  --novelty-feature-source "${NOVELTY_FEATURE_SOURCE:-multilayer}" \
  --novelty-layers "${NOVELTY_LAYERS:-7,14,21}" \
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
  --actual-online-grpo-max-steps "$ACTUAL_ONLINE_GRPO_MAX_STEPS" \
  --actual-grpo-lr "$ACTUAL_GRPO_LR" \
  --actual-grpo-kl-coef "$ACTUAL_GRPO_KL_COEF" \
  --benchmark-num-rollouts "$BENCHMARK_NUM_ROLLOUTS" \
  --benchmark-rollout-max-completion-len "$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN" \
  --rescore-chunk-size 1 \
  --logprob-chunk-tokens "$LOGPROB_CHUNK_TOKENS" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --output-dir "$OUT" \
  --wandb-run-name "$RUN_NAME" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --seed 0 \
  2>&1 | tee -a "$OUT/console.log"
