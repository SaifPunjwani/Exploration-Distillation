#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SSH_HOST="${SSH_HOST:?set SSH_HOST to the ssh host alias of the TPU VM}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
SESSION_NAME="${SESSION_NAME:-ew4a_q3dapo_fsdp}"
QUEUE_TAG="${QUEUE_TAG:-ew4a_qwen3_1p7b_dapo_fsdp}"
RUN_ARTIFACT_GCS_PREFIX="${RUN_ARTIFACT_GCS_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/${QUEUE_TAG}}"
FSDP_MODE="${FSDP_MODE:-train}"
FSDP_VISIBLE_CHIPS="${FSDP_VISIBLE_CHIPS:-0,1,2,3}"
FSDP_NUM_PROCESSES="${FSDP_NUM_PROCESSES:-}"

count_visible_chips() {
  python3 - "$1" <<'PY'
import sys

items = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
print(len(items) if items else 0)
PY
}

if [ -z "$FSDP_NUM_PROCESSES" ]; then
  FSDP_NUM_PROCESSES="$(count_visible_chips "$FSDP_VISIBLE_CHIPS")"
fi

ALLOW_DIRTY=1 DIRECT_SSH_HOST="$SSH_HOST" SYNC_WORKERS=0 REMOTE_ROOT="$REMOTE_ROOT" \
  bash scripts/sync_tpu_slice_repo.sh

TRAIN_ARGS=(
  main.py
  --device tpu
  --use-wandb
  --seed "${SEED:-0}"
  --model-name "${MODEL_NAME:-Qwen/Qwen3-1.7B}"
  --dataset-name "${DATASET_NAME:-dapo_math_17k}"
  --dataset-split-policy "${DATASET_SPLIT_POLICY:-auto}"
  --topic-split-mode "${TOPIC_SPLIT_MODE:-heuristic_math_topics}"
  --topic-partition-policy "${TOPIC_PARTITION_POLICY:-explicit_or_heuristic}"
  --decoupling-mode "${DECOUPLING_MODE:-decoupled}"
  --pipeline-mode "${PIPELINE_MODE:-two_model}"
  --grpo-loss-type "${GRPO_LOSS_TYPE:-dr_grpo}"
  --train-selection-policy "${TRAIN_SELECTION_POLICY:-random}"
  --train-probe-pool-size "${TRAIN_PROBE_POOL_SIZE:-256}"
  --train-probe-samples-per-prompt "${TRAIN_PROBE_SAMPLES_PER_PROMPT:-4}"
  --train-probe-temperature "${TRAIN_PROBE_TEMPERATURE:-0.8}"
  --train-probe-batch-size "${TRAIN_PROBE_BATCH_SIZE:-4}"
  --max-train-examples "${MAX_TRAIN_EXAMPLES:-128}"
  --max-eval-examples "${MAX_EVAL_EXAMPLES:-32}"
  --grpo-max-steps "${GRPO_MAX_STEPS:-8}"
  --grpo-batch-size "${GRPO_BATCH_SIZE:-1}"
  --grpo-num-generations "${GRPO_NUM_GENERATIONS:-4}"
  --grpo-generation-batch-size "${GRPO_GENERATION_BATCH_SIZE:-1}"
  --num-samples-per-prompt "${NUM_SAMPLES_PER_PROMPT:-4}"
  --answer-constraint-mode "${ANSWER_CONSTRAINT_MODE:-none}"
  --max-prompt-len "${MAX_PROMPT_LEN:-512}"
  --max-completion-len "${MAX_COMPLETION_LEN:-1024}"
  --exploration-max-completion-len "${EXPLORATION_MAX_COMPLETION_LEN:-1024}"
  --eval-max-completion-len "${EVAL_MAX_COMPLETION_LEN:-1024}"
  --max-total-len "${MAX_TOTAL_LEN:-1536}"
  --distill-epochs "${DISTILL_EPOCHS:-2}"
  --distill-batch-size "${DISTILL_BATCH_SIZE:-1}"
  --max-distill-examples "${MAX_DISTILL_EXAMPLES:-128}"
  --actual-grpo-epochs "${ACTUAL_GRPO_EPOCHS:-3}"
  --actual-grpo-batch-size "${ACTUAL_GRPO_BATCH_SIZE:-1}"
  --actual-grpo-grad-accum "${ACTUAL_GRPO_GRAD_ACCUM:-1}"
  --actual-min-accepted-trajectories "${ACTUAL_MIN_ACCEPTED_TRAJECTORIES:-4}"
  --max-actual-examples "${MAX_ACTUAL_EXAMPLES:-128}"
  --novelty-feature-source "${NOVELTY_FEATURE_SOURCE:-last}"
  --novelty-feature-max-length "${NOVELTY_FEATURE_MAX_LENGTH:-128}"
  --novelty-metric "${NOVELTY_METRIC:-sqrt_mse}"
  --novelty-text-contract "${NOVELTY_TEXT_CONTRACT:-full_raw}"
  --rnd-hidden "${RND_HIDDEN:-512}"
  --lambda-novelty "${LAMBDA_NOVELTY:-0.03}"
  --exploration-response-style "${EXPLORATION_RESPONSE_STYLE:-reasoning_final}"
  --eval-response-style "${EVAL_RESPONSE_STYLE:-reasoning_final}"
  --exploration-answer-constraint-mode "${EXPLORATION_ANSWER_CONSTRAINT_MODE:-none}"
  --eval-answer-constraint-mode "${EVAL_ANSWER_CONSTRAINT_MODE:-none}"
  --exploration-thinking-mode "${EXPLORATION_THINKING_MODE:-no_think}"
  --eval-thinking-mode "${EVAL_THINKING_MODE:-no_think}"
  --actual-target-contract "${ACTUAL_TARGET_CONTRACT:-full_trace}"
  --actual-init-source "${ACTUAL_INIT_SOURCE:-base}"
  --actual-train-mode "${ACTUAL_TRAIN_MODE:-sft_grpo}"
  --actual-reward-source "${ACTUAL_REWARD_SOURCE:-correctness}"
  --actual-dataset-policy "${ACTUAL_DATASET_POLICY:-accepted_only}"
  --actual-max-incorrect-ratio "${ACTUAL_MAX_INCORRECT_RATIO:-0.25}"
  --actual-target-correct-fraction "${ACTUAL_TARGET_CORRECT_FRACTION:-0.75}"
  --actual-grpo-kl-coef "${ACTUAL_GRPO_KL_COEF:-0.10}"
  --actual-grpo-lr "${ACTUAL_GRPO_LR:-1e-6}"
  --distill-lr "${DISTILL_LR:-5e-6}"
  --base-eval-mode "${BASE_EVAL_MODE:-grpo}"
  --reward-correct "${REWARD_CORRECT:-1.0}"
  --reward-incorrect "${REWARD_INCORRECT:-0.0}"
  --reward-shape "${REWARD_SHAPE:-linear}"
  --anchor-metric "${ANCHOR_METRIC:-none}"
  --trajectory-batch-size "${TRAJECTORY_BATCH_SIZE:-1}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-1}"
  --round-budget-mode "${ROUND_BUDGET_MODE:-constant_total}"
  --format-penalty-weight "${FORMAT_PENALTY_WEIGHT:-0.1}"
  --plots-dir "runs/${QUEUE_TAG}/plots"
  --explorer-traj-path "runs/${QUEUE_TAG}/explorer_trajectories.jsonl"
  --explorer-output-dir "runs/${QUEUE_TAG}/explorer_ckpts"
  --saved-model-root "${SAVED_MODEL_ROOT:?set SAVED_MODEL_ROOT to the saved-model directory on the TPU VM}"
)

if [ -n "${INCORRECT_NOVELTY_SCALE:-}" ]; then
  TRAIN_ARGS+=(--incorrect-novelty-scale "$INCORRECT_NOVELTY_SCALE")
fi

if [ -n "${WANDB_RUN_NAME:-}" ]; then
  TRAIN_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
fi

printf -v quoted_train_args '%q ' "${TRAIN_ARGS[@]}"

remote_cmd=$(
  cat <<EOF
cd '$REMOTE_ROOT'
tmux kill-session -t '$SESSION_NAME' 2>/dev/null || true
pkill -9 -f "accelerate launch --config_file /tmp/tmx_accelerate_fsdp" 2>/dev/null || true
pkill -9 -f "from multiprocessing.spawn import spawn_main" 2>/dev/null || true
pkill -9 -f "from multiprocessing.resource_tracker import main" 2>/dev/null || true
fuser -k /dev/vfio/0 2>/dev/null || true
sleep 2
mkdir -p '$REMOTE_ROOT/runs/$QUEUE_TAG'
tmux new-session -d -s '$SESSION_NAME' \
  "cd '$REMOTE_ROOT' && \
   mkdir -p 'runs/${QUEUE_TAG}' && \
   export WANDB_PROJECT='${WANDB_PROJECT:-two-model-explore}' && \
   export WANDB_ENTITY='${WANDB_ENTITY:-}' && \
   export RUN_ARTIFACT_GCS_PREFIX='${RUN_ARTIFACT_GCS_PREFIX}' && \
   export FSDP_MODE='${FSDP_MODE}' && \
   export FSDP_VISIBLE_CHIPS='${FSDP_VISIBLE_CHIPS}' && \
   export FSDP_NUM_PROCESSES='${FSDP_NUM_PROCESSES}' && \
   export TMX_GRPO_GRAD_CKPT='${TMX_GRPO_GRAD_CKPT:-1}' && \
   export TMX_XLA_SAFE_GENERATE='${TMX_XLA_SAFE_GENERATE:-1}' && \
   export TMX_XLA_INCREMENTAL_GENERATE='${TMX_XLA_INCREMENTAL_GENERATE:-1}' && \
   export TMX_XLA_SKIP_LOADER_MARK_STEP='${TMX_XLA_SKIP_LOADER_MARK_STEP:-1}' && \
   export TMX_GRPO_LOGGING_STRATEGY='${TMX_GRPO_LOGGING_STRATEGY:-steps}' && \
   export TMX_GRPO_LOGGING_STEPS='${TMX_GRPO_LOGGING_STEPS:-1}' && \
   export TMX_GRPO_SAVE_STRATEGY='${TMX_GRPO_SAVE_STRATEGY:-steps}' && \
   export TMX_GRPO_SAVE_STEPS='${TMX_GRPO_SAVE_STEPS:-4}' && \
   export TMX_GRPO_SAVE_TOTAL_LIMIT='${TMX_GRPO_SAVE_TOTAL_LIMIT:-3}' && \
   export TMX_SKIP_INTERMEDIATE_EVALS='${TMX_SKIP_INTERMEDIATE_EVALS:-0}' && \
   export TMX_DISABLE_BF16='${TMX_DISABLE_BF16:-1}' && \
   export TMX_USE_XLA_WORLD_SIZE='${TMX_USE_XLA_WORLD_SIZE:-1}' && \
   bash scripts/run_tpu_fsdp_local.sh ${quoted_train_args} > 'runs/${QUEUE_TAG}/console.log' 2>&1"
tmux ls | grep '$SESSION_NAME'
EOF
)

ssh -o StrictHostKeyChecking=no "$SSH_HOST" "$remote_cmd"
printf '[ew4a-q3dapo-fsdp] launched session=%s chips=%s processes=%s mode=%s\n' \
  "$SESSION_NAME" "$FSDP_VISIBLE_CHIPS" "$FSDP_NUM_PROCESSES" "$FSDP_MODE"
