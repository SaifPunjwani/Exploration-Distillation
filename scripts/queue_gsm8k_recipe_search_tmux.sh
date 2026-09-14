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

. scripts/tpu_target_lib.sh
resolve_tpu_target_defaults
resolve_tpu_worker_env

TPU_NAME="${TPU_NAME}"
ZONE="${ZONE}"
TPU_TRANSPORT_FLAG="$(tmx_tpu_transport_flag)"
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-$(tmx_resolve_tpu_accelerator_type 2>/dev/null || true)}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VENV_DIR="${VENV_DIR:-.venv_tpu_slice}"
SESSION_PREFIX="${SESSION_PREFIX:-gsm8k_recipe}"
QUEUE_TAG="${QUEUE_TAG:-gsm8k_recipe_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/${QUEUE_TAG}}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
LOCAL_CHIPS="${LOCAL_CHIPS:-4}"
CHIPS="${CHIPS:-}"
WORKERS=(${WORKERS})
SEED="${SEED:-0}"
TRAIN_SELECTION_POLICY="${TRAIN_SELECTION_POLICY:-random}"
TRAIN_SELECTION_MANIFEST_PATH="${TRAIN_SELECTION_MANIFEST_PATH:-}"
RECIPE_MODE="${RECIPE_MODE:-recipe_search}"
RUN_PROFILE="${RUN_PROFILE:-smoke}"
RETRY_ON_FAILURE="${RETRY_ON_FAILURE:-1}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-15}"
RUN_OFFSET="${RUN_OFFSET:-0}"
RUN_LIMIT="${RUN_LIMIT:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-32}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-64}"
MAX_PROMPT_LEN_WAS_SET="${MAX_PROMPT_LEN+x}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-192}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-24}"
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-24}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-320}"
TRAIN_PROBE_POOL_SIZE="${TRAIN_PROBE_POOL_SIZE:-128}"
TRAIN_PROBE_SAMPLES_PER_PROMPT="${TRAIN_PROBE_SAMPLES_PER_PROMPT:-4}"
TRAIN_PROBE_TEMPERATURE="${TRAIN_PROBE_TEMPERATURE:-0.8}"
TRAIN_PROBE_BATCH_SIZE="${TRAIN_PROBE_BATCH_SIZE:-8}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-8}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-4}"
GRPO_LOSS_TYPE="${GRPO_LOSS_TYPE:-dr_grpo}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-1}"
MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-128}"
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-1}"
ACTUAL_GRPO_BATCH_SIZE="${ACTUAL_GRPO_BATCH_SIZE:-4}"
ACTUAL_GRPO_GRAD_ACCUM="${ACTUAL_GRPO_GRAD_ACCUM:-1}"
ACTUAL_MIN_ACCEPTED_TRAJECTORIES="${ACTUAL_MIN_ACCEPTED_TRAJECTORIES:-8}"
ACTUAL_DATASET_POLICY="${ACTUAL_DATASET_POLICY:-accepted_balanced}"
ACTUAL_MAX_INCORRECT_RATIO="${ACTUAL_MAX_INCORRECT_RATIO:-1.0}"
MAX_ACTUAL_EXAMPLES="${MAX_ACTUAL_EXAMPLES:-256}"
ANSWER_CONSTRAINT_MODE="${ANSWER_CONSTRAINT_MODE:-integer}"
EXPLORATION_RESPONSE_STYLE="${EXPLORATION_RESPONSE_STYLE:-answer_only}"
EVAL_RESPONSE_STYLE="${EVAL_RESPONSE_STYLE:-answer_only}"
EXPLORATION_ANSWER_CONSTRAINT_MODE="${EXPLORATION_ANSWER_CONSTRAINT_MODE:-auto}"
EVAL_ANSWER_CONSTRAINT_MODE="${EVAL_ANSWER_CONSTRAINT_MODE:-auto}"
EXPLORATION_THINKING_MODE="${EXPLORATION_THINKING_MODE:-auto}"
EVAL_THINKING_MODE="${EVAL_THINKING_MODE:-auto}"
ACTUAL_THINKING_MODE="${ACTUAL_THINKING_MODE:-no_think}"
ACTUAL_TARGET_CONTRACT="${ACTUAL_TARGET_CONTRACT:-canonical}"
BASE_EVAL_MODE="${BASE_EVAL_MODE:-grpo}"
ROUND_BUDGET_MODE="${ROUND_BUDGET_MODE:-constant_total}"
SAVE_BASE_MODEL="${SAVE_BASE_MODEL:-0}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:-}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
BENCHMARK_DATASET_SPLIT_POLICY="${BENCHMARK_DATASET_SPLIT_POLICY:-auto}"
BENCHMARK_EVAL_MODE="${BENCHMARK_EVAL_MODE:-sampled}"
MAX_BENCHMARK="${MAX_BENCHMARK:-}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_PASS_K="${BENCHMARK_ROLLOUT_PASS_K:-1,4,8,16,32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-16384}"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-1}"
NOVELTY_FEATURE_SOURCE="${NOVELTY_FEATURE_SOURCE:-multilayer}"
NOVELTY_METRIC="${NOVELTY_METRIC:-sqrt_mse}"
NOVELTY_LAYERS="${NOVELTY_LAYERS:-}"
NOVELTY_LAYER_POOL="${NOVELTY_LAYER_POOL:-mean}"
NOVELTY_LAYER_AGG="${NOVELTY_LAYER_AGG:-mean}"
NOVELTY_FEATURE_MAX_LENGTH="${NOVELTY_FEATURE_MAX_LENGTH:-}"
RND_HIDDEN="${RND_HIDDEN:-512}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:-0.0}"
REWARD_SHAPE="${REWARD_SHAPE:-linear}"
INCORRECT_NOVELTY_SCALE="${INCORRECT_NOVELTY_SCALE:-0.25}"
FORMAT_PENALTY_WEIGHT="${FORMAT_PENALTY_WEIGHT:-0.2}"
TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
TPU_HOST_THREADS="${TPU_HOST_THREADS:-4}"
TPU_HOST_INTEROP_THREADS="${TPU_HOST_INTEROP_THREADS:-1}"
KILL_ALL_TPU_PROCS="${KILL_ALL_TPU_PROCS:-0}"

export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$TPU_HOST_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$TPU_HOST_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$TPU_HOST_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$TPU_HOST_THREADS}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$TPU_HOST_THREADS}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-$TPU_HOST_INTEROP_THREADS}"
export TMX_TPU_HOST_TORCH_THREADS="${TMX_TPU_HOST_TORCH_THREADS:-$TPU_HOST_THREADS}"
export TMX_TPU_HOST_TORCH_INTEROP_THREADS="${TMX_TPU_HOST_TORCH_INTEROP_THREADS:-$TPU_HOST_INTEROP_THREADS}"
export HF_HOME="${HF_HOME:-$REMOTE_ROOT/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-}"

case "$RUN_PROFILE" in
  smoke)
    ;;
  stable_answer)
    [ "$MAX_TRAIN_EXAMPLES" = "32" ] && MAX_TRAIN_EXAMPLES="512"
    [ "$MAX_EVAL_EXAMPLES" = "64" ] && MAX_EVAL_EXAMPLES="256"
    [ -z "${MAX_PROMPT_LEN_WAS_SET}" ] && [ "$MAX_PROMPT_LEN" = "192" ] && MAX_PROMPT_LEN="256"
    [ "$MAX_COMPLETION_LEN" = "128" ] && MAX_COMPLETION_LEN="128"
    [ "$EXPLORATION_MAX_COMPLETION_LEN" = "24" ] && EXPLORATION_MAX_COMPLETION_LEN="48"
    [ "$EVAL_MAX_COMPLETION_LEN" = "24" ] && EVAL_MAX_COMPLETION_LEN="64"
    [ "$MAX_TOTAL_LEN" = "320" ] && MAX_TOTAL_LEN="384"
    [ "$TRAIN_PROBE_POOL_SIZE" = "128" ] && TRAIN_PROBE_POOL_SIZE="512"
    [ "$TRAIN_PROBE_SAMPLES_PER_PROMPT" = "4" ] && TRAIN_PROBE_SAMPLES_PER_PROMPT="2"
    [ "$TRAIN_PROBE_BATCH_SIZE" = "8" ] && TRAIN_PROBE_BATCH_SIZE="16"
    [ "$GRPO_MAX_STEPS" = "8" ] && GRPO_MAX_STEPS="48"
    [ "$NUM_SAMPLES_PER_PROMPT" = "4" ] && NUM_SAMPLES_PER_PROMPT="4"
    [ "$DISTILL_EPOCHS" = "1" ] && DISTILL_EPOCHS="2"
    [ "$MAX_DISTILL_EXAMPLES" = "128" ] && MAX_DISTILL_EXAMPLES="512"
    [ "$ACTUAL_GRPO_EPOCHS" = "1" ] && ACTUAL_GRPO_EPOCHS="2"
    [ "$MAX_ACTUAL_EXAMPLES" = "256" ] && MAX_ACTUAL_EXAMPLES="512"
    ANSWER_CONSTRAINT_MODE="integer"
    EXPLORATION_RESPONSE_STYLE="answer_only"
    EVAL_RESPONSE_STYLE="answer_only"
    EXPLORATION_ANSWER_CONSTRAINT_MODE="auto"
    EVAL_ANSWER_CONSTRAINT_MODE="auto"
    [ "$EXPLORATION_THINKING_MODE" = "auto" ] && EXPLORATION_THINKING_MODE="no_think"
    [ "$EVAL_THINKING_MODE" = "auto" ] && EVAL_THINKING_MODE="no_think"
    ACTUAL_TARGET_CONTRACT="canonical"
    BASE_EVAL_MODE="grpo"
    ;;
  stable_trace)
    [ "$MAX_TRAIN_EXAMPLES" = "32" ] && MAX_TRAIN_EXAMPLES="512"
    [ "$MAX_EVAL_EXAMPLES" = "64" ] && MAX_EVAL_EXAMPLES="256"
    [ -z "${MAX_PROMPT_LEN_WAS_SET}" ] && [ "$MAX_PROMPT_LEN" = "192" ] && MAX_PROMPT_LEN="256"
    [ "$MAX_COMPLETION_LEN" = "128" ] && MAX_COMPLETION_LEN="192"
    [ "$EXPLORATION_MAX_COMPLETION_LEN" = "24" ] && EXPLORATION_MAX_COMPLETION_LEN="96"
    [ "$EVAL_MAX_COMPLETION_LEN" = "24" ] && EVAL_MAX_COMPLETION_LEN="128"
    [ "$MAX_TOTAL_LEN" = "320" ] && MAX_TOTAL_LEN="448"
    [ "$TRAIN_PROBE_POOL_SIZE" = "128" ] && TRAIN_PROBE_POOL_SIZE="512"
    [ "$TRAIN_PROBE_SAMPLES_PER_PROMPT" = "4" ] && TRAIN_PROBE_SAMPLES_PER_PROMPT="2"
    [ "$TRAIN_PROBE_BATCH_SIZE" = "8" ] && TRAIN_PROBE_BATCH_SIZE="16"
    [ "$GRPO_MAX_STEPS" = "8" ] && GRPO_MAX_STEPS="48"
    [ "$NUM_SAMPLES_PER_PROMPT" = "4" ] && NUM_SAMPLES_PER_PROMPT="4"
    [ "$DISTILL_EPOCHS" = "1" ] && DISTILL_EPOCHS="2"
    [ "$MAX_DISTILL_EXAMPLES" = "128" ] && MAX_DISTILL_EXAMPLES="512"
    [ "$ACTUAL_GRPO_EPOCHS" = "1" ] && ACTUAL_GRPO_EPOCHS="2"
    [ "$MAX_ACTUAL_EXAMPLES" = "256" ] && MAX_ACTUAL_EXAMPLES="512"
    ANSWER_CONSTRAINT_MODE="none"
    EXPLORATION_RESPONSE_STYLE="reasoning_final"
    EVAL_RESPONSE_STYLE="reasoning_final"
    EXPLORATION_ANSWER_CONSTRAINT_MODE="none"
    EVAL_ANSWER_CONSTRAINT_MODE="none"
    ACTUAL_TARGET_CONTRACT="full_trace"
    BASE_EVAL_MODE="grpo"
    ;;
  dapo_standard)
    # DAPO-think canary: use standard boxed formatting, but keep the
    # Explorer envelope moderate enough to survive single-chip TPU runs.
    [ "$MAX_TRAIN_EXAMPLES" = "32" ] && MAX_TRAIN_EXAMPLES="512"
    [ "$MAX_EVAL_EXAMPLES" = "64" ] && MAX_EVAL_EXAMPLES="256"
    [ -z "${MAX_PROMPT_LEN_WAS_SET}" ] && [ "$MAX_PROMPT_LEN" = "192" ] && MAX_PROMPT_LEN="256"
    [ "$MAX_COMPLETION_LEN" = "128" ] && MAX_COMPLETION_LEN="192"
    [ "$EXPLORATION_MAX_COMPLETION_LEN" = "24" ] && EXPLORATION_MAX_COMPLETION_LEN="128"
    [ "$EVAL_MAX_COMPLETION_LEN" = "24" ] && EVAL_MAX_COMPLETION_LEN="192"
    [ "$MAX_TOTAL_LEN" = "320" ] && MAX_TOTAL_LEN="448"
    [ "$TRAIN_PROBE_POOL_SIZE" = "128" ] && TRAIN_PROBE_POOL_SIZE="512"
    [ "$TRAIN_PROBE_SAMPLES_PER_PROMPT" = "4" ] && TRAIN_PROBE_SAMPLES_PER_PROMPT="4"
    [ "$TRAIN_PROBE_BATCH_SIZE" = "8" ] && TRAIN_PROBE_BATCH_SIZE="4"
    [ "$GRPO_MAX_STEPS" = "8" ] && GRPO_MAX_STEPS="12"
    [ "$GRPO_NUM_GENERATIONS" = "2" ] && GRPO_NUM_GENERATIONS="2"
    [ "$NUM_SAMPLES_PER_PROMPT" = "4" ] && NUM_SAMPLES_PER_PROMPT="2"
    [ "$DISTILL_EPOCHS" = "1" ] && DISTILL_EPOCHS="2"
    [ "$MAX_DISTILL_EXAMPLES" = "128" ] && MAX_DISTILL_EXAMPLES="512"
    [ "$ACTUAL_GRPO_EPOCHS" = "1" ] && ACTUAL_GRPO_EPOCHS="3"
    [ "$MAX_ACTUAL_EXAMPLES" = "256" ] && MAX_ACTUAL_EXAMPLES="512"
    # Random selection — do NOT filter to solvable-only.
    TRAIN_SELECTION_POLICY="random"
    # Standard \boxed{} format via reasoning_final style.
    ANSWER_CONSTRAINT_MODE="none"
    EXPLORATION_RESPONSE_STYLE="reasoning_final"
    EVAL_RESPONSE_STYLE="reasoning_final"
    EXPLORATION_ANSWER_CONSTRAINT_MODE="none"
    EVAL_ANSWER_CONSTRAINT_MODE="none"
    [ "$EXPLORATION_THINKING_MODE" = "auto" ] && EXPLORATION_THINKING_MODE="think"
    [ "$EVAL_THINKING_MODE" = "auto" ] && EVAL_THINKING_MODE="think"
    [ "$INCORRECT_NOVELTY_SCALE" = "0.25" ] && INCORRECT_NOVELTY_SCALE="0.0"
    ACTUAL_TARGET_CONTRACT="full_trace"
    BASE_EVAL_MODE="raw"
    ;;
  *)
    echo "Unknown RUN_PROFILE=$RUN_PROFILE"
    exit 1
    ;;
esac

ensure_gcloud_ssh_key() {
  local key_path="${HOME}/.ssh/google_compute_engine"
  if [ ! -f "$key_path" ]; then
    return
  fi
  if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l >/dev/null 2>&1; then
    eval "$(ssh-agent -s)" >/dev/null
  fi
  ssh-add "$key_path" >/dev/null 2>&1 || true
}

ensure_gcloud_ssh_key

kill_existing_sessions() {
  tmux ls 2>/dev/null | sed 's/:.*//' | grep '^'"${SESSION_PREFIX}"'_' | xargs -r -n1 tmux kill-session -t || true
  local worker
  for worker in "${WORKERS[@]}"; do
    local remote_cmd="tmux ls 2>/dev/null | sed 's/:.*//' | grep '^${SESSION_PREFIX}_' | xargs -r -n1 tmux kill-session -t || true"
    if [ "$KILL_ALL_TPU_PROCS" = "1" ]; then
      pkill -f 'python -u main.py --device [t]pu' || true
      pkill -f 'python main.py --device [t]pu' || true
      remote_cmd="pkill -f 'python -u main.py --device [t]pu' || true; pkill -f 'python main.py --device [t]pu' || true; ${remote_cmd}"
    fi
    tmx_gcloud alpha compute tpus tpu-vm ssh \
      "$TPU_NAME" \
      --worker="$worker" \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG" \
      --command="$remote_cmd"
  done
}

if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  kill_existing_sessions
  bash scripts/bootstrap_tpu_slice.sh
fi

mkdir -p "$RUN_ROOT"

COMMON_ARGS=(
  --device tpu
  --use-wandb
  --seed "$SEED"
  --model-name "$MODEL_NAME"
  --dataset-name "$DATASET_NAME"
  --train-selection-policy "$TRAIN_SELECTION_POLICY"
  --train-probe-pool-size "$TRAIN_PROBE_POOL_SIZE"
  --train-probe-samples-per-prompt "$TRAIN_PROBE_SAMPLES_PER_PROMPT"
  --train-probe-temperature "$TRAIN_PROBE_TEMPERATURE"
  --train-probe-batch-size "$TRAIN_PROBE_BATCH_SIZE"
  --dataset-split-policy auto
  --topic-split-mode heuristic_math_topics
  --topic-partition-policy explicit_or_heuristic
  --decoupling-mode decoupled
  --base-eval-mode "$BASE_EVAL_MODE"
  --actual-target-contract "$ACTUAL_TARGET_CONTRACT"
  --max-train-examples "$MAX_TRAIN_EXAMPLES"
  --max-eval-examples "$MAX_EVAL_EXAMPLES"
  --grpo-max-steps "$GRPO_MAX_STEPS"
  --grpo-batch-size "$GRPO_BATCH_SIZE"
  --grpo-grad-accum "$GRPO_GRAD_ACCUM"
  --grpo-loss-type "$GRPO_LOSS_TYPE"
  --grpo-num-generations "$GRPO_NUM_GENERATIONS"
  --grpo-generation-batch-size "$GRPO_GENERATION_BATCH_SIZE"
  --num-samples-per-prompt "$NUM_SAMPLES_PER_PROMPT"
  --answer-constraint-mode "$ANSWER_CONSTRAINT_MODE"
  --exploration-response-style "$EXPLORATION_RESPONSE_STYLE"
  --eval-response-style "$EVAL_RESPONSE_STYLE"
  --exploration-answer-constraint-mode "$EXPLORATION_ANSWER_CONSTRAINT_MODE"
  --eval-answer-constraint-mode "$EVAL_ANSWER_CONSTRAINT_MODE"
  --exploration-thinking-mode "$EXPLORATION_THINKING_MODE"
  --eval-thinking-mode "$EVAL_THINKING_MODE"
  --actual-thinking-mode "$ACTUAL_THINKING_MODE"
  --distill-epochs "$DISTILL_EPOCHS"
  --distill-batch-size "$DISTILL_BATCH_SIZE"
  --max-distill-examples "$MAX_DISTILL_EXAMPLES"
  --actual-grpo-epochs "$ACTUAL_GRPO_EPOCHS"
  --actual-grpo-batch-size "$ACTUAL_GRPO_BATCH_SIZE"
  --actual-grpo-grad-accum "$ACTUAL_GRPO_GRAD_ACCUM"
  --actual-min-accepted-trajectories "$ACTUAL_MIN_ACCEPTED_TRAJECTORIES"
  --actual-dataset-policy "$ACTUAL_DATASET_POLICY"
  --actual-max-incorrect-ratio "$ACTUAL_MAX_INCORRECT_RATIO"
  --max-actual-examples "$MAX_ACTUAL_EXAMPLES"
  --novelty-feature-source "$NOVELTY_FEATURE_SOURCE"
  --novelty-metric "$NOVELTY_METRIC"
  --novelty-text-contract full_raw
  --rnd-hidden "$RND_HIDDEN"
  --reward-correct "$REWARD_CORRECT"
  --reward-incorrect "$REWARD_INCORRECT"
  --reward-shape "$REWARD_SHAPE"
  --incorrect-novelty-scale "$INCORRECT_NOVELTY_SCALE"
  --trajectory-batch-size "$TRAJECTORY_BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --round-budget-mode "$ROUND_BUDGET_MODE"
  --max-prompt-len "$MAX_PROMPT_LEN"
  --max-completion-len "$MAX_COMPLETION_LEN"
  --exploration-max-completion-len "$EXPLORATION_MAX_COMPLETION_LEN"
  --eval-max-completion-len "$EVAL_MAX_COMPLETION_LEN"
  --max-total-len "$MAX_TOTAL_LEN"
  --format-penalty-weight "$FORMAT_PENALTY_WEIGHT"
)

if [ "$SAVE_BASE_MODEL" = "1" ]; then
  COMMON_ARGS+=(--save-base-model)
fi

if [ -n "$SAVED_MODEL_ROOT" ]; then
  COMMON_ARGS+=(--saved-model-root "$SAVED_MODEL_ROOT")
fi

if [ -n "$BENCHMARK_DATASET_NAME" ]; then
  COMMON_ARGS+=(--benchmark-dataset-name "$BENCHMARK_DATASET_NAME")
  COMMON_ARGS+=(--benchmark-dataset-split-policy "$BENCHMARK_DATASET_SPLIT_POLICY")
  COMMON_ARGS+=(--benchmark-eval-mode "$BENCHMARK_EVAL_MODE")
  COMMON_ARGS+=(--benchmark-num-rollouts "$BENCHMARK_NUM_ROLLOUTS")
  COMMON_ARGS+=(--benchmark-rollout-temperature "$BENCHMARK_ROLLOUT_TEMPERATURE")
  COMMON_ARGS+=(--benchmark-rollout-top-p "$BENCHMARK_ROLLOUT_TOP_P")
  COMMON_ARGS+=(--benchmark-rollout-pass-k "$BENCHMARK_ROLLOUT_PASS_K")
  COMMON_ARGS+=(--benchmark-rollout-max-completion-len "$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN")
  COMMON_ARGS+=(--benchmark-eval-batch-size "$BENCHMARK_EVAL_BATCH_SIZE")
  if [ -n "$BENCHMARK_DATASET_CONFIG_NAME" ]; then
    COMMON_ARGS+=(--benchmark-dataset-config-name "$BENCHMARK_DATASET_CONFIG_NAME")
  fi
  if [ -n "$MAX_BENCHMARK" ]; then
    COMMON_ARGS+=(--max-benchmark-examples "$MAX_BENCHMARK")
  fi
fi

if [ -n "$NOVELTY_LAYERS" ]; then
  COMMON_ARGS+=(--novelty-layers "$NOVELTY_LAYERS")
  COMMON_ARGS+=(--novelty-layer-pool "$NOVELTY_LAYER_POOL")
  COMMON_ARGS+=(--novelty-layer-agg "$NOVELTY_LAYER_AGG")
fi

if [ -n "$NOVELTY_FEATURE_MAX_LENGTH" ]; then
  COMMON_ARGS+=(--novelty-feature-max-length "$NOVELTY_FEATURE_MAX_LENGTH")
fi

if [ -n "$DATASET_CONFIG_NAME" ]; then
  COMMON_ARGS+=(--dataset-config-name "$DATASET_CONFIG_NAME")
fi

if [ "$TRAIN_SELECTION_POLICY" = "base_solvable" ]; then
  if [ -n "$TRAIN_SELECTION_MANIFEST_PATH" ]; then
    COMMON_ARGS+=(--train-selection-manifest-path "$TRAIN_SELECTION_MANIFEST_PATH")
  else
    COMMON_ARGS+=(--train-selection-manifest-path "${RUN_ROOT}/train_selection_seed${SEED}.json")
  fi
fi

COMMON_ARGS_NO_WANDB=()
for arg in "${COMMON_ARGS[@]}"; do
  if [ "$arg" = "--use-wandb" ]; then
    continue
  fi
  COMMON_ARGS_NO_WANDB+=("$arg")
done

PREP_ARGS=("${COMMON_ARGS_NO_WANDB[@]}")
PREP_ARGS+=(
  --no-wandb
  --train-probe-pool-size 64
  --train-probe-samples-per-prompt 2
  --train-probe-batch-size 16
  --num-samples-per-prompt 1
  --max-eval-examples 8
  --answer-constraint-mode integer
  --exploration-response-style answer_only
  --eval-response-style answer_only
  --exploration-answer-constraint-mode auto
  --eval-answer-constraint-mode auto
  --exploration-max-completion-len 24
  --eval-max-completion-len 24
)

if [ -n "$TRAIN_SELECTION_MANIFEST_PATH" ]; then
  SELECTION_MANIFEST_RUN_PATH="$TRAIN_SELECTION_MANIFEST_PATH"
  SELECTION_MANIFEST_FILE="$(basename "$TRAIN_SELECTION_MANIFEST_PATH")"
else
  SELECTION_MANIFEST_FILE="train_selection_seed${SEED}.json"
  SELECTION_MANIFEST_RUN_PATH="${RUN_ROOT}/${SELECTION_MANIFEST_FILE}"
fi
SELECTION_MANIFEST_REMOTE_ABS="${REMOTE_ROOT}/${SELECTION_MANIFEST_RUN_PATH}"
SELECTION_MANIFEST_LOCAL="/tmp/${QUEUE_TAG}_${SELECTION_MANIFEST_FILE}"

format_cli_args() {
  local formatted=""
  local arg
  for arg in "$@"; do
    formatted+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$formatted"
}

prepare_train_selection_manifest() {
  local prep_worker="${WORKERS[0]}"
  local prep_session="${SESSION_PREFIX}_prepare_seed${SEED}"
  local prep_run_dir="${RUN_ROOT}/_selection_seed${SEED}"
  local local_script="/tmp/${prep_session}.sh"
  local remote_script="/tmp/${prep_session}.sh"
  local prep_common=""
  local -a prep_workers=()
  local tmx_grpo_grad_ckpt="${TMX_GRPO_GRAD_CKPT-}"
  local tmx_use_xla_world_size="${TMX_USE_XLA_WORLD_SIZE-}"
  local tmx_xla_safe_generate="${TMX_XLA_SAFE_GENERATE-}"
  local tmx_disable_bf16="${TMX_DISABLE_BF16-}"
  local tmx_xla_skip_loader_mark_step="${TMX_XLA_SKIP_LOADER_MARK_STEP-}"
  local tmx_xla_mark_step_every="${TMX_XLA_MARK_STEP_EVERY-}"
  local tmx_xla_incremental_generate="${TMX_XLA_INCREMENTAL_GENERATE-}"
  local tmx_debug_generate="${TMX_DEBUG_GENERATE-}"
  local xla_disable_functionalization="${XLA_DISABLE_FUNCTIONALIZATION-}"
  prep_common="$(format_cli_args "${PREP_ARGS[@]}")"

  if tmx_gcloud alpha compute tpus tpu-vm ssh \
    "$TPU_NAME" \
    --worker="$prep_worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="test -f '$SELECTION_MANIFEST_REMOTE_ABS'"
  then
    existing_count="$(
      tmx_gcloud alpha compute tpus tpu-vm ssh \
        "$TPU_NAME" \
        --worker="$prep_worker" \
        --zone="$ZONE" \
        "$TPU_TRANSPORT_FLAG" \
        --command="python3 - <<'PY' '$SELECTION_MANIFEST_REMOTE_ABS'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    payload = json.load(f)
selected = payload.get('selected_row_ids') or []
if selected:
    print(len(selected))
else:
    print(int(payload.get('selected_count', 0) or 0))
PY" 2>/dev/null || true
)"
    existing_count="$(printf '%s' "$existing_count" | tail -n 1 | tr -d '\r')"
    if [ -n "${existing_count:-}" ] && [ "${existing_count}" = "${MAX_TRAIN_EXAMPLES}" ]; then
      echo "Using existing train-selection manifest for seed $SEED on worker $prep_worker: selected=${existing_count}, path=${SELECTION_MANIFEST_RUN_PATH}"
      return
    fi
    echo "Discarding stale train-selection manifest for seed $SEED on worker $prep_worker: selected=${existing_count:-unknown}, expected=${MAX_TRAIN_EXAMPLES}, path=${SELECTION_MANIFEST_RUN_PATH}"
    tmx_gcloud alpha compute tpus tpu-vm ssh \
      "$TPU_NAME" \
      --worker="$prep_worker" \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG" \
      --command="rm -f '$SELECTION_MANIFEST_REMOTE_ABS'"
  fi

  if [ "${PREP_TARGET_WORKERS+set}" = "set" ] && [ "${#PREP_TARGET_WORKERS[@]}" -gt 0 ]; then
    prep_workers=("${PREP_TARGET_WORKERS[@]}")
  else
    prep_workers=("${WORKERS[@]}")
  fi

  cat >"$local_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi
. "$VENV_DIR/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HF_TOKEN"
fi
if [ -n "${HF_ACCESS_TOKEN:-}" ]; then
  export HF_ACCESS_TOKEN="$HF_ACCESS_TOKEN"
fi
if [ -n "${HF_HOME:-}" ]; then
  export HF_HOME="$HF_HOME"
fi
if [ -n "${HF_HUB_CACHE:-}" ]; then
  export HF_HUB_CACHE="$HF_HUB_CACHE"
fi
if [ -n "${HF_DATASETS_CACHE:-}" ]; then
  export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
fi
if [ -n "${TRANSFORMERS_CACHE:-}" ]; then
  export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
fi
if [ -n "${HF_HUB_OFFLINE:-}" ]; then
  export HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
fi
if [ -n "${HF_DATASETS_OFFLINE:-}" ]; then
  export HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE"
fi
if [ -n "${TRANSFORMERS_OFFLINE:-}" ]; then
  export TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
fi
if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
  default_libtpu_path="$REMOTE_ROOT/$VENV_DIR/lib/python3.10/site-packages/libtpu/libtpu.so"
  if [ -f "\$default_libtpu_path" ]; then
    export TPU_LIBRARY_PATH="\$default_libtpu_path"
  fi
fi
export PJRT_DEVICE=TPU
export TMX_ALLOW_SINGLE_WORKER_TPU="${TMX_ALLOW_SINGLE_WORKER_TPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$TPU_HOST_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$TPU_HOST_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$TPU_HOST_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$TPU_HOST_THREADS}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$TPU_HOST_THREADS}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-$TPU_HOST_INTEROP_THREADS}"
export TMX_TPU_HOST_TORCH_THREADS="${TMX_TPU_HOST_TORCH_THREADS:-$TPU_HOST_THREADS}"
export TMX_TPU_HOST_TORCH_INTEROP_THREADS="${TMX_TPU_HOST_TORCH_INTEROP_THREADS:-$TPU_HOST_INTEROP_THREADS}"
if [ "$tmx_disable_bf16" = "1" ]; then
  unset XLA_USE_BF16
else
  export XLA_USE_BF16=1
fi
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_REQUIRE_TPU=1
export TPU_SKIP_MDS_QUERY=1
if [ -n "$TPU_ACCELERATOR_TYPE" ]; then
  export TPU_ACCELERATOR_TYPE="$TPU_ACCELERATOR_TYPE"
fi
export TMX_WANDB_MASTER_ONLY=0
if [ -n "$tmx_grpo_grad_ckpt" ]; then
  export TMX_GRPO_GRAD_CKPT="$tmx_grpo_grad_ckpt"
fi
if [ -n "$tmx_use_xla_world_size" ]; then
  export TMX_USE_XLA_WORLD_SIZE="$tmx_use_xla_world_size"
fi
if [ -n "$tmx_xla_safe_generate" ]; then
  export TMX_XLA_SAFE_GENERATE="$tmx_xla_safe_generate"
fi
if [ -n "$tmx_disable_bf16" ]; then
  export TMX_DISABLE_BF16="$tmx_disable_bf16"
fi
if [ -n "$tmx_xla_skip_loader_mark_step" ]; then
  export TMX_XLA_SKIP_LOADER_MARK_STEP="$tmx_xla_skip_loader_mark_step"
fi
if [ -n "$tmx_xla_mark_step_every" ]; then
  export TMX_XLA_MARK_STEP_EVERY="$tmx_xla_mark_step_every"
fi
if [ -n "$tmx_xla_incremental_generate" ]; then
  export TMX_XLA_INCREMENTAL_GENERATE="$tmx_xla_incremental_generate"
fi
if [ -n "$tmx_debug_generate" ]; then
  export TMX_DEBUG_GENERATE="$tmx_debug_generate"
fi
if [ -n "$xla_disable_functionalization" ]; then
  export XLA_DISABLE_FUNCTIONALIZATION="$xla_disable_functionalization"
fi
export TPU_VISIBLE_CHIPS=0
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_ID="$prep_worker"
export TPU_WORKER_HOSTNAMES=localhost
mkdir -p "$RUN_ROOT" "$prep_run_dir"
mkdir -p "$(dirname "$SELECTION_MANIFEST_RUN_PATH")"
python -u main.py$prep_common \\
  --prepare-train-selection-only \\
  --plots-dir "$prep_run_dir/plots" \\
  --explorer-traj-path "$prep_run_dir/explorer_trajectories.jsonl"
EOF
  chmod +x "$local_script"

  tmx_gcloud alpha compute tpus tpu-vm scp \
    "$local_script" \
    "${TPU_NAME}:${remote_script}" \
    --worker="$prep_worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG"

  tmx_gcloud alpha compute tpus tpu-vm ssh \
    "$TPU_NAME" \
    --worker="$prep_worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="chmod +x '$remote_script' && bash '$remote_script'"

  rm -f "$SELECTION_MANIFEST_LOCAL"
  tmx_gcloud alpha compute tpus tpu-vm scp \
    "${TPU_NAME}:${SELECTION_MANIFEST_REMOTE_ABS}" \
    "$SELECTION_MANIFEST_LOCAL" \
    --worker="$prep_worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG"

  local worker
  for worker in "${prep_workers[@]}"; do
    tmx_gcloud alpha compute tpus tpu-vm ssh \
      "$TPU_NAME" \
      --worker="$worker" \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG" \
      --command="mkdir -p '$REMOTE_ROOT/$RUN_ROOT' '$REMOTE_ROOT/$(dirname "$SELECTION_MANIFEST_RUN_PATH")'"

    tmx_gcloud alpha compute tpus tpu-vm scp \
      "$SELECTION_MANIFEST_LOCAL" \
      "${TPU_NAME}:${SELECTION_MANIFEST_REMOTE_ABS}" \
      --worker="$worker" \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG"
  done

  local selected_count
  selected_count="$(python3 - <<'PY' "$SELECTION_MANIFEST_LOCAL"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)
selected_row_ids = payload.get("selected_row_ids") or []
if selected_row_ids:
    print(len(selected_row_ids))
else:
    print(int(payload.get("selected_count", 0) or 0))
PY
)"
  echo "Prepared train-selection manifest for seed $SEED on worker $prep_worker: selected=${selected_count}, path=${SELECTION_MANIFEST_RUN_PATH}"
}

if [ "$RECIPE_MODE" = "baseline_audit" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_base_raw"
    "${QUEUE_TAG}_base_grpo"
  )

  RUN_ARGS=(
    "--pipeline-mode base_only --base-eval-mode raw --lambda-novelty 0.0 --anchor-metric none"
    "--pipeline-mode base_only --base-eval-mode grpo --lambda-novelty 0.0 --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "stable_answer_warmstart" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_answer_ws_distill_dl5e6"
    "${QUEUE_TAG}_two_answer_ws_sftgrpo_kl010_lr1e6_dl5e6"
    "${QUEUE_TAG}_two_answer_ws_sftgrpo_kl010_lr1e6_dl1e5"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode distill --distill-lr 5e-6 --lambda-novelty 0.03 --anchor-metric none"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 1e-5 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "stable_answer_baseinit" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_answer_base_sftgrpo_kl010_lr1e6_dl5e6"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --base-eval-mode raw --actual-init-source base --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "stable_answer_baseinit_piecewise" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_answer_base_sftgrpo_pw_kl010_lr1e6_dl5e6"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --base-eval-mode raw --actual-init-source base --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape piecewise --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "stable_trace_warmstart" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_trace_ws_distill_dl5e6"
    "${QUEUE_TAG}_two_trace_ws_distill_dl1e5"
    "${QUEUE_TAG}_two_trace_ws_sftgrpo_kl010_lr1e6_dl5e6"
    "${QUEUE_TAG}_two_trace_ws_sftgrpo_kl010_lr1e6_dl1e5"
    "${QUEUE_TAG}_two_trace_ws_sftgrpo_kl005_lr1e6_dl5e6"
    "${QUEUE_TAG}_two_trace_ws_sftgrpo_kl010_lr5e7_dl5e6"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode distill --distill-lr 5e-6 --lambda-novelty 0.03 --anchor-metric none"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode distill --distill-lr 1e-5 --lambda-novelty 0.03 --anchor-metric none"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 1e-5 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 5e-7 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "stable_trace_baseinit" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_trace_base_sftgrpo_kl010_lr1e6_dl5e6"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --base-eval-mode raw --actual-init-source base --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "stable_trace_baseinit_piecewise" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_trace_base_sftgrpo_pw_kl010_lr1e6_dl5e6"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --base-eval-mode raw --actual-init-source base --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape piecewise --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "stable_trace_multi_round" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_mr_trace_ws_distill_dl5e6_r2_topic"
    "${QUEUE_TAG}_mr_trace_ws_distill_dl5e6_r4_topic"
    "${QUEUE_TAG}_mr_trace_ws_distill_dl5e6_r4_seq"
    "${QUEUE_TAG}_mr_trace_ws_sftgrpo_kl010_lr1e6_dl5e6_r4_topic"
    "${QUEUE_TAG}_mr_trace_ws_sftgrpo_kl010_lr1e6_dl5e6_r4_seq"
  )

  RUN_ARGS=(
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode distill --distill-lr 5e-6 --lambda-novelty 0.03 --anchor-metric none --num-rounds 2 --round-shard-policy topic_balanced --round-budget-mode constant_total"
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode distill --distill-lr 5e-6 --lambda-novelty 0.03 --anchor-metric none --num-rounds 4 --round-shard-policy topic_balanced --round-budget-mode constant_total"
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode distill --distill-lr 5e-6 --lambda-novelty 0.03 --anchor-metric none --num-rounds 4 --round-shard-policy sequential --round-budget-mode constant_total"
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --num-rounds 4 --round-shard-policy topic_balanced --round-budget-mode constant_total"
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --num-rounds 4 --round-shard-policy sequential --round-budget-mode constant_total"
  )
elif [ "$RECIPE_MODE" = "actual_fix" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_acconly_cf100_kl005_lr2e6"
    "${QUEUE_TAG}_two_acconly_cf100_kl010_lr2e6"
    "${QUEUE_TAG}_two_corronly_cf100_kl005_lr2e6"
    "${QUEUE_TAG}_two_acconly_cf075_kl005_lr2e6"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 2e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 2e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy correct_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 2e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 0.75 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 2e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "actual_fix_extra" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_acconly_cf100_kl005_lr1e6"
    "${QUEUE_TAG}_two_acconly_cf100_kl015_lr1e6"
    "${QUEUE_TAG}_two_corronly_cf100_kl010_lr1e6"
    "${QUEUE_TAG}_two_acconly_cf100_kl005_lr1e6_rclip1"
    "${QUEUE_TAG}_two_acconly_cf100_kl005_lr1e6_nonorm"
    "${QUEUE_TAG}_two_acconly_cf100_kl010_lr1e6_nonorm_rclip1"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 1e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.15 --actual-grpo-lr 1e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy correct_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 1e-6 --actual-grpo-reward-clip 1.0 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 1e-6 --no-actual-reward-norm --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --actual-grpo-reward-clip 1.0 --no-actual-reward-norm --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
  )
elif [ "$RECIPE_MODE" = "actual_fix_warmstart" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_two_ws_acconly_cf100_kl010_lr1e6_dl1e5"
    "${QUEUE_TAG}_two_ws_acconly_cf100_kl015_lr5e7_dl1e5"
    "${QUEUE_TAG}_two_ws_corronly_cf100_kl010_lr1e6_dl1e5"
    "${QUEUE_TAG}_two_ws_acconly_cf100_kl010_lr1e6_nonorm_rclip1_dl1e5"
    "${QUEUE_TAG}_two_ws_distill_dl1e5"
    "${QUEUE_TAG}_two_ws_distill_dl5e6"
    "${QUEUE_TAG}_two_ws_acconly_cf100_kl005_lr5e7_adv1_dl5e6"
    "${QUEUE_TAG}_two_ws_acconly_cf100_kl020_lr5e7_rclip05_dl5e6"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 1e-5 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.15 --actual-grpo-lr 5e-7 --distill-lr 1e-5 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy correct_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 1e-5 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --actual-grpo-reward-clip 1.0 --no-actual-reward-norm --distill-lr 1e-5 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode distill --distill-lr 1e-5 --lambda-novelty 0.03 --anchor-metric none --max-train-examples 64 --max-distill-examples 256"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode distill --distill-lr 5e-6 --lambda-novelty 0.03 --anchor-metric none --max-train-examples 64 --max-distill-examples 256"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 5e-7 --actual-grpo-adv-clip 1.0 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
    "--pipeline-mode two_model --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.20 --actual-grpo-lr 5e-7 --actual-grpo-reward-clip 0.5 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
  )
elif [ "$RECIPE_MODE" = "actual_fix_warmstart_multi_round" ]; then
  RUN_NAMES=(
    "${QUEUE_TAG}_mr_ws_acconly_cf100_kl010_lr1e6_dl1e5_r4_topic"
    "${QUEUE_TAG}_mr_ws_acconly_cf100_kl010_lr1e6_dl1e5_r4_seq"
    "${QUEUE_TAG}_mr_ws_acconly_cf100_kl010_lr1e6_dl5e6_r4_topic"
    "${QUEUE_TAG}_mr_ws_acconly_cf100_kl010_lr1e6_dl5e6_r4_seq"
  )

  RUN_ARGS=(
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 1e-5 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --num-rounds 4 --round-shard-policy topic_balanced --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 1e-5 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --num-rounds 4 --round-shard-policy sequential --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --num-rounds 4 --round-shard-policy topic_balanced --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
    "--pipeline-mode multi_round --actual-init-source baseline_grpo --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.0 --actual-target-correct-fraction 1.00 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none --num-rounds 4 --round-shard-policy sequential --max-train-examples 64 --max-distill-examples 256 --max-actual-examples 256"
  )
elif [ "$RECIPE_MODE" = "dapo_standard_baseinit" ]; then
  # Standard DAPO methodology: long completions, \boxed{} format, base init, no solvable filtering.
  RUN_NAMES=(
    "${QUEUE_TAG}_two_dapo_base_sftgrpo_kl010_lr1e6_dl5e6"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --base-eval-mode raw --actual-init-source base --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.25 --actual-target-correct-fraction 0.75 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape piecewise --anchor-metric none --reward-incorrect -1.0"
  )
elif [ "$RECIPE_MODE" = "dapo_standard_sweep" ]; then
  # Sweep over key hyperparameters with standard DAPO methodology.
  RUN_NAMES=(
    "${QUEUE_TAG}_two_dapo_base_sftgrpo_kl005_lr5e7"
    "${QUEUE_TAG}_two_dapo_base_sftgrpo_kl010_lr1e6"
    "${QUEUE_TAG}_two_dapo_base_sftgrpo_kl015_lr5e7"
    "${QUEUE_TAG}_two_dapo_base_distill_dl1e5"
  )

  RUN_ARGS=(
    "--pipeline-mode two_model --base-eval-mode grpo --actual-init-source base --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.25 --actual-target-correct-fraction 0.75 --actual-grpo-kl-coef 0.05 --actual-grpo-lr 5e-7 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --base-eval-mode grpo --actual-init-source base --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.25 --actual-target-correct-fraction 0.75 --actual-grpo-kl-coef 0.10 --actual-grpo-lr 1e-6 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --base-eval-mode grpo --actual-init-source base --actual-train-mode sft_grpo --actual-reward-source correctness --actual-dataset-policy accepted_only --actual-max-incorrect-ratio 0.25 --actual-target-correct-fraction 0.75 --actual-grpo-kl-coef 0.15 --actual-grpo-lr 5e-7 --distill-lr 5e-6 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --base-eval-mode grpo --actual-init-source base --actual-train-mode distill --distill-lr 1e-5 --lambda-novelty 0.03 --anchor-metric none"
  )
else
  RUN_NAMES=(
    "${QUEUE_TAG}_base_grpo"
    "${QUEUE_TAG}_two_distill_l003"
    "${QUEUE_TAG}_two_corr_cf050_l003"
    "${QUEUE_TAG}_two_corr_cf075_l003"
    "${QUEUE_TAG}_two_corr_cf050_piecewise_l003"
    "${QUEUE_TAG}_two_corr_cf050_piecewise_l005"
    "${QUEUE_TAG}_two_trace_piecewise_l003"
    "${QUEUE_TAG}_mr_corr_cf050_piecewise_l003"
  )

  RUN_ARGS=(
    "--pipeline-mode base_only --lambda-novelty 0.03 --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode distill --lambda-novelty 0.03 --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-grpo-kl-coef 0.0 --actual-target-correct-fraction 0.50 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-grpo-kl-coef 0.0 --actual-target-correct-fraction 0.75 --lambda-novelty 0.03 --reward-shape linear --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-grpo-kl-coef 0.0 --actual-target-correct-fraction 0.50 --lambda-novelty 0.03 --reward-shape piecewise --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-reward-source correctness --actual-grpo-kl-coef 0.0 --actual-target-correct-fraction 0.50 --lambda-novelty 0.05 --reward-shape piecewise --anchor-metric none"
    "--pipeline-mode two_model --actual-train-mode sft_grpo --actual-target-contract full_trace --actual-reward-source correctness --actual-grpo-kl-coef 0.0 --actual-target-correct-fraction 0.50 --lambda-novelty 0.03 --reward-shape piecewise --anchor-metric none"
    "--pipeline-mode multi_round --actual-train-mode sft_grpo --actual-reward-source correctness --actual-grpo-kl-coef 0.0 --actual-target-correct-fraction 0.50 --num-rounds 4 --lambda-novelty 0.03 --reward-shape piecewise --anchor-metric none"
  )
fi

if [ "$RUN_OFFSET" -gt 0 ] || [ "$RUN_LIMIT" -gt 0 ]; then
  start_idx=$(( RUN_OFFSET < 0 ? 0 : RUN_OFFSET ))
  total_runs="${#RUN_NAMES[@]}"
  if [ "$start_idx" -gt "$total_runs" ]; then
    start_idx="$total_runs"
  fi
  end_idx="$total_runs"
  if [ "$RUN_LIMIT" -gt 0 ]; then
    candidate_end=$(( start_idx + RUN_LIMIT ))
    if [ "$candidate_end" -lt "$end_idx" ]; then
      end_idx="$candidate_end"
    fi
  fi

  sliced_names=()
  sliced_args=()
  idx="$start_idx"
  while [ "$idx" -lt "$end_idx" ]; do
    sliced_names+=("${RUN_NAMES[$idx]}")
    sliced_args+=("${RUN_ARGS[$idx]}")
    idx=$((idx + 1))
  done
  RUN_NAMES=("${sliced_names[@]}")
  RUN_ARGS=("${sliced_args[@]}")
fi

declare -a CHIP_IDS=()
if [ -n "$CHIPS" ]; then
  for chip_id in ${CHIPS//,/ }; do
    [ -n "$chip_id" ] || continue
    CHIP_IDS+=("$chip_id")
  done
else
  chip_id=0
  while [ "$chip_id" -lt "$LOCAL_CHIPS" ]; do
    CHIP_IDS+=("$chip_id")
    chip_id=$((chip_id + 1))
  done
fi

if [ "${#CHIP_IDS[@]}" -eq 0 ]; then
  echo "No chip ids resolved. Set CHIPS or LOCAL_CHIPS."
  exit 1
fi

slot_count=$(( ${#WORKERS[@]} * ${#CHIP_IDS[@]} ))

declare -a SLOT_LOCAL_SCRIPTS=()
declare -a SLOT_REMOTE_SCRIPTS=()
declare -a SLOT_SESSIONS=()
declare -a SLOT_WORKERS=()
declare -a SLOT_CREATED=()
declare -a PREP_TARGET_WORKERS=()

build_slot_script() {
  local worker="$1"
  local chip="$2"
  local session="$3"
  local local_script="$4"
  local tmx_grpo_grad_ckpt="${TMX_GRPO_GRAD_CKPT-}"
  local tmx_use_xla_world_size="${TMX_USE_XLA_WORLD_SIZE-}"
  local tmx_xla_safe_generate="${TMX_XLA_SAFE_GENERATE-}"
  local tmx_disable_bf16="${TMX_DISABLE_BF16-}"
  local tmx_xla_skip_loader_mark_step="${TMX_XLA_SKIP_LOADER_MARK_STEP-}"
  local tmx_xla_mark_step_every="${TMX_XLA_MARK_STEP_EVERY-}"
  local tmx_xla_incremental_generate="${TMX_XLA_INCREMENTAL_GENERATE-}"
  local tmx_debug_generate="${TMX_DEBUG_GENERATE-}"
  local xla_disable_functionalization="${XLA_DISABLE_FUNCTIONALIZATION-}"
  local tmx_resume_training="${TMX_RESUME_TRAINING-}"
  local tmx_skip_intermediate_evals="${TMX_SKIP_INTERMEDIATE_EVALS-}"
  local tmx_grpo_save_strategy="${TMX_GRPO_SAVE_STRATEGY-}"
  local tmx_grpo_save_steps="${TMX_GRPO_SAVE_STEPS-}"
  local tmx_grpo_save_total_limit="${TMX_GRPO_SAVE_TOTAL_LIMIT-}"
  local tmx_allow_xla_grpo_intermediate_save="${TMX_ALLOW_XLA_GRPO_INTERMEDIATE_SAVE-}"
  local tmx_grpo_logging_strategy="${TMX_GRPO_LOGGING_STRATEGY-}"
  local tmx_grpo_logging_steps="${TMX_GRPO_LOGGING_STEPS-}"
  local tmx_xla_persistent_cache_dir="${TMX_XLA_PERSISTENT_CACHE_DIR-}"
  local tmx_xla_persistent_cache_readonly="${TMX_XLA_PERSISTENT_CACHE_READONLY-}"
  local tmx_abort_on_degenerate_explorer="${TMX_ABORT_ON_DEGENERATE_EXPLORER-}"
  local tmx_abort_on_bad_traj_window="${TMX_ABORT_ON_BAD_TRAJ_WINDOW-}"
  local tmx_traj_health_window_prompts="${TMX_TRAJ_HEALTH_WINDOW_PROMPTS-}"
  local tmx_traj_health_min_correct="${TMX_TRAJ_HEALTH_MIN_CORRECT-}"
  local run_artifact_gcs_prefix="${RUN_ARTIFACT_GCS_PREFIX-}"
  local run_sync_interval_seconds="${RUN_SYNC_INTERVAL_SECONDS:-90}"

  rm -f "$local_script"
  cat >"$local_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi
. "$VENV_DIR/bin/activate"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_ENTITY="$WANDB_ENTITY"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY="$WANDB_API_KEY"
fi
if [ -n "${WANDB_MODE:-}" ]; then
  export WANDB_MODE="$WANDB_MODE"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HF_TOKEN"
fi
if [ -n "${HF_ACCESS_TOKEN:-}" ]; then
  export HF_ACCESS_TOKEN="$HF_ACCESS_TOKEN"
fi
if [ -n "${HF_HOME:-}" ]; then
  export HF_HOME="$HF_HOME"
fi
if [ -n "${HF_HUB_CACHE:-}" ]; then
  export HF_HUB_CACHE="$HF_HUB_CACHE"
fi
if [ -n "${HF_DATASETS_CACHE:-}" ]; then
  export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
fi
if [ -n "${TRANSFORMERS_CACHE:-}" ]; then
  export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
fi
if [ -n "${HF_HUB_OFFLINE:-}" ]; then
  export HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
fi
if [ -n "${HF_DATASETS_OFFLINE:-}" ]; then
  export HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE"
fi
if [ -n "${TRANSFORMERS_OFFLINE:-}" ]; then
  export TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
fi
if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
  default_libtpu_path="$REMOTE_ROOT/$VENV_DIR/lib/python3.10/site-packages/libtpu/libtpu.so"
  if [ -f "\$default_libtpu_path" ]; then
    export TPU_LIBRARY_PATH="\$default_libtpu_path"
  fi
fi
export PJRT_DEVICE=TPU
export TMX_ALLOW_SINGLE_WORKER_TPU="${TMX_ALLOW_SINGLE_WORKER_TPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$TPU_HOST_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$TPU_HOST_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$TPU_HOST_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$TPU_HOST_THREADS}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$TPU_HOST_THREADS}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-$TPU_HOST_INTEROP_THREADS}"
export TMX_TPU_HOST_TORCH_THREADS="${TMX_TPU_HOST_TORCH_THREADS:-$TPU_HOST_THREADS}"
export TMX_TPU_HOST_TORCH_INTEROP_THREADS="${TMX_TPU_HOST_TORCH_INTEROP_THREADS:-$TPU_HOST_INTEROP_THREADS}"
if [ "$tmx_disable_bf16" = "1" ]; then
  unset XLA_USE_BF16
else
  export XLA_USE_BF16=1
fi
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_REQUIRE_TPU=1
export TPU_SKIP_MDS_QUERY=1
if [ -n "$TPU_ACCELERATOR_TYPE" ]; then
  export TPU_ACCELERATOR_TYPE="$TPU_ACCELERATOR_TYPE"
fi
export TMX_WANDB_MASTER_ONLY=0
if [ -n "$tmx_grpo_grad_ckpt" ]; then
  export TMX_GRPO_GRAD_CKPT="$tmx_grpo_grad_ckpt"
fi
if [ -n "$tmx_use_xla_world_size" ]; then
  export TMX_USE_XLA_WORLD_SIZE="$tmx_use_xla_world_size"
fi
if [ -n "$tmx_xla_safe_generate" ]; then
  export TMX_XLA_SAFE_GENERATE="$tmx_xla_safe_generate"
fi
if [ -n "$tmx_disable_bf16" ]; then
  export TMX_DISABLE_BF16="$tmx_disable_bf16"
fi
if [ -n "$tmx_xla_skip_loader_mark_step" ]; then
  export TMX_XLA_SKIP_LOADER_MARK_STEP="$tmx_xla_skip_loader_mark_step"
fi
if [ -n "$tmx_xla_mark_step_every" ]; then
  export TMX_XLA_MARK_STEP_EVERY="$tmx_xla_mark_step_every"
fi
if [ -n "$tmx_xla_incremental_generate" ]; then
  export TMX_XLA_INCREMENTAL_GENERATE="$tmx_xla_incremental_generate"
fi
if [ -n "$tmx_debug_generate" ]; then
  export TMX_DEBUG_GENERATE="$tmx_debug_generate"
fi
if [ -n "$xla_disable_functionalization" ]; then
  export XLA_DISABLE_FUNCTIONALIZATION="$xla_disable_functionalization"
fi
if [ -n "$tmx_resume_training" ]; then
  export TMX_RESUME_TRAINING="$tmx_resume_training"
fi
if [ -n "$tmx_skip_intermediate_evals" ]; then
  export TMX_SKIP_INTERMEDIATE_EVALS="$tmx_skip_intermediate_evals"
fi
if [ -n "$tmx_grpo_save_strategy" ]; then
  export TMX_GRPO_SAVE_STRATEGY="$tmx_grpo_save_strategy"
fi
if [ -n "$tmx_grpo_save_steps" ]; then
  export TMX_GRPO_SAVE_STEPS="$tmx_grpo_save_steps"
fi
if [ -n "$tmx_grpo_save_total_limit" ]; then
  export TMX_GRPO_SAVE_TOTAL_LIMIT="$tmx_grpo_save_total_limit"
fi
if [ -n "$tmx_allow_xla_grpo_intermediate_save" ]; then
  export TMX_ALLOW_XLA_GRPO_INTERMEDIATE_SAVE="$tmx_allow_xla_grpo_intermediate_save"
fi
if [ -n "$tmx_grpo_logging_strategy" ]; then
  export TMX_GRPO_LOGGING_STRATEGY="$tmx_grpo_logging_strategy"
fi
if [ -n "$tmx_grpo_logging_steps" ]; then
  export TMX_GRPO_LOGGING_STEPS="$tmx_grpo_logging_steps"
fi
if [ -n "$tmx_abort_on_degenerate_explorer" ]; then
  export TMX_ABORT_ON_DEGENERATE_EXPLORER="$tmx_abort_on_degenerate_explorer"
fi
if [ -n "$tmx_abort_on_bad_traj_window" ]; then
  export TMX_ABORT_ON_BAD_TRAJ_WINDOW="$tmx_abort_on_bad_traj_window"
fi
if [ -n "$tmx_traj_health_window_prompts" ]; then
  export TMX_TRAJ_HEALTH_WINDOW_PROMPTS="$tmx_traj_health_window_prompts"
fi
if [ -n "$tmx_traj_health_min_correct" ]; then
  export TMX_TRAJ_HEALTH_MIN_CORRECT="$tmx_traj_health_min_correct"
fi
if [ "$tmx_xla_persistent_cache_dir" = "disable" ] || [ "$tmx_xla_persistent_cache_dir" = "__disable__" ] || [ "$tmx_xla_persistent_cache_dir" = "off" ] || [ "$tmx_xla_persistent_cache_dir" = "0" ] || [ "$tmx_xla_persistent_cache_dir" = "false" ]; then
  unset TMX_XLA_PERSISTENT_CACHE_DIR
elif [ -n "$tmx_xla_persistent_cache_dir" ]; then
  export TMX_XLA_PERSISTENT_CACHE_DIR="$tmx_xla_persistent_cache_dir"
else
  export TMX_XLA_PERSISTENT_CACHE_DIR="/tmp/tmx_xla_cache_${QUEUE_TAG}_w${worker}_c${chip}"
fi
if [ -n "$tmx_xla_persistent_cache_readonly" ]; then
  export TMX_XLA_PERSISTENT_CACHE_READONLY="$tmx_xla_persistent_cache_readonly"
fi
export TPU_VISIBLE_CHIPS="$chip"
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_ID="$worker"
export TPU_WORKER_HOSTNAMES=localhost
mkdir -p "$RUN_ROOT" "$RUN_ROOT/_slot_logs"
slot_log="$RUN_ROOT/_slot_logs/${session}.log"
gcs_run_root=""
sync_loop_pid=""
if [ -n "$run_artifact_gcs_prefix" ]; then
  gcs_run_root="${run_artifact_gcs_prefix%/}/${QUEUE_TAG}"
fi
gcs_restore_run_root() {
  if [ -z "\$gcs_run_root" ]; then
    return
  fi
  if gcloud storage ls "\$gcs_run_root" >/dev/null 2>&1; then
    mkdir -p "$RUN_ROOT"
    gcloud storage rsync -r "\$gcs_run_root" "$RUN_ROOT" >/dev/null 2>&1 || true
  fi
}
gcs_sync_run_root() {
  if [ -z "\$gcs_run_root" ]; then
    return
  fi
  gcloud storage rsync -r "$RUN_ROOT" "\$gcs_run_root" >/dev/null 2>&1 || true
}
start_gcs_sync_loop() {
  if [ -z "\$gcs_run_root" ]; then
    return
  fi
  (
    while true; do
      gcs_sync_run_root
      sleep "$run_sync_interval_seconds"
    done
  ) &
  sync_loop_pid=\$!
}
stop_gcs_sync_loop() {
  if [ -n "\${sync_loop_pid:-}" ]; then
    kill "\$sync_loop_pid" >/dev/null 2>&1 || true
    wait "\$sync_loop_pid" 2>/dev/null || true
    sync_loop_pid=""
  fi
}
gcs_restore_run_root
start_gcs_sync_loop
trap 'stop_gcs_sync_loop; gcs_sync_run_root' EXIT
echo "[slot] worker=$worker chip=$chip session=$session start=\$(date -Iseconds)" | tee -a "\$slot_log"
EOF
}

append_run() {
  local local_script="$1"
  local run_name="$2"
  local run_args="$3"
  local run_dir="$RUN_ROOT/$run_name"
  local common=""
  common="$(format_cli_args "${COMMON_ARGS[@]}")"
  cat >>"$local_script" <<EOF
run_dir="$run_dir"
run_log="\$run_dir/console.log"
mkdir -p "\$run_dir"
echo "[run] start name=$run_name time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
if python -u main.py$common $run_args \\
  --wandb-run-name "$run_name" \\
  --plots-dir "\$run_dir/plots" \\
  --explorer-traj-path "\$run_dir/explorer_trajectories.jsonl" \\
  --explorer-output-dir "\$run_dir/explorer_ckpts" \\
  2>&1 | tee -a "\$run_log"
then
  echo "[run] success name=$run_name time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
else
  status=\${PIPESTATUS[0]}
  echo "[run] failure name=$run_name code=\$status time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
  if [ "$RETRY_ON_FAILURE" = "1" ]; then
    echo "[run] retry name=$run_name after=${RETRY_SLEEP_SECONDS}s" | tee -a "\$slot_log" "\$run_log"
    sleep "$RETRY_SLEEP_SECONDS"
    if python -u main.py$common $run_args \\
      --wandb-run-name "${run_name}_retry1" \\
      --plots-dir "\$run_dir/plots_retry1" \\
      --explorer-traj-path "\$run_dir/explorer_trajectories_retry1.jsonl" \\
      --explorer-output-dir "\$run_dir/explorer_ckpts" \\
      2>&1 | tee -a "\$run_log"
    then
      echo "[run] retry-success name=$run_name time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
    else
      retry_status=\${PIPESTATUS[0]}
      echo "[run] retry-failure name=$run_name code=\$retry_status time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
    fi
  fi
fi
EOF
}

for idx in "${!RUN_NAMES[@]}"; do
  slot=$(( idx % slot_count ))
  chips_per_worker="${#CHIP_IDS[@]}"
  worker="${WORKERS[$((slot / chips_per_worker))]}"
  chip="${CHIP_IDS[$((slot % chips_per_worker))]}"
  session="${SESSION_PREFIX}_w${worker}_c${chip}"
  local_script="/tmp/${session}.sh"
  remote_script="/tmp/${session}.sh"

  if [ -z "${SLOT_CREATED[$slot]:-}" ]; then
    build_slot_script "$worker" "$chip" "$session" "$local_script"
    SLOT_LOCAL_SCRIPTS+=("$local_script")
    SLOT_REMOTE_SCRIPTS+=("$remote_script")
    SLOT_SESSIONS+=("$session")
    SLOT_WORKERS+=("$worker")
    SLOT_CREATED[$slot]=1
  fi

  append_run "$local_script" "${RUN_NAMES[$idx]}" "${RUN_ARGS[$idx]}"
done

if [ "${SLOT_WORKERS+set}" = "set" ]; then
  for worker in "${SLOT_WORKERS[@]}"; do
    already_present=0
    if [ "${PREP_TARGET_WORKERS+set}" = "set" ]; then
      for prep_worker in "${PREP_TARGET_WORKERS[@]}"; do
        if [ "$prep_worker" = "$worker" ]; then
          already_present=1
          break
        fi
      done
    fi
    if [ "$already_present" -eq 0 ]; then
      PREP_TARGET_WORKERS+=("$worker")
    fi
  done
fi

if [ "$TRAIN_SELECTION_POLICY" = "base_solvable" ]; then
  prepare_train_selection_manifest
fi

for local_script in "${SLOT_LOCAL_SCRIPTS[@]}"; do
  echo 'echo "[slot] finished at $(date -Iseconds)" | tee -a "$slot_log"' >>"$local_script"
  chmod +x "$local_script"
done

for idx in "${!SLOT_LOCAL_SCRIPTS[@]}"; do
  local_script="${SLOT_LOCAL_SCRIPTS[$idx]}"
  remote_script="${SLOT_REMOTE_SCRIPTS[$idx]}"
  session="${SLOT_SESSIONS[$idx]}"
  worker="${SLOT_WORKERS[$idx]}"

  tmx_gcloud alpha compute tpus tpu-vm scp \
    "$local_script" \
    "${TPU_NAME}:${remote_script}" \
    --worker="$worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG"

  tmx_gcloud alpha compute tpus tpu-vm ssh \
    "$TPU_NAME" \
    --worker="$worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="chmod +x '$remote_script' && (tmux has-session -t '$session' 2>/dev/null && tmux kill-session -t '$session' || true) && tmux new-session -d -s '$session' 'bash $remote_script'"
done

echo "Queued ${#RUN_NAMES[@]} GSM8K recipe runs across ${#SLOT_SESSIONS[@]} tmux sessions."
echo "Queue tag: $QUEUE_TAG"
