#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-a}"
TPU_TRANSPORT_FLAG="$(tmx_tpu_transport_flag)"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
RUN_PREFIX="${RUN_PREFIX:-ew4a_qwen3_1p7b_avg32_$(date +%Y%m%d_%H%M%S)}"

SYNC_REPO="${SYNC_REPO:-1}"
BOOTSTRAP_VLLM="${BOOTSTRAP_VLLM:-1}"
ALLOW_DIRTY="${ALLOW_DIRTY:-1}"

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"

BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_TOP_K="${BENCHMARK_ROLLOUT_TOP_K:-0}"
BENCHMARK_ROLLOUT_PASS_K="${BENCHMARK_ROLLOUT_PASS_K:-1,4,8,16,32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-16384}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-32}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
USE_WANDB="${USE_WANDB:-1}"

THINK_WORKERS=(${THINK_WORKERS:-0 1 2 3 4 5 6 7})
NOTHINK_WORKERS=(${NOTHINK_WORKERS:-8 9 10 11 12 13 14 15})

log() {
  printf '[ew4a-qwen3] %s\n' "$*"
}

remote_ssh() {
  local worker="$1"
  local command="$2"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker="$worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="$command"
}

if [ "$SYNC_REPO" = "1" ]; then
  log "syncing repo to all workers"
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  SYNC_WORKERS=all \
  ALLOW_DIRTY="$ALLOW_DIRTY" \
  bash scripts/sync_tpu_slice_repo.sh
fi

if [ "$BOOTSTRAP_VLLM" = "1" ]; then
  log "bootstrapping vllm env on all workers"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker=all \
    --batch-size=all \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' bash scripts/bootstrap_vllm_tpu_env.sh"
fi

launch_group() {
  local group_name="$1"
  local thinking_mode="$2"
  shift 2
  local workers=("$@")
  local shard_count="${#workers[@]}"
  local idx worker run_name session_name remote_cmd

  log "launching group=$group_name thinking_mode=$thinking_mode workers=${workers[*]}"
  for idx in "${!workers[@]}"; do
    worker="${workers[$idx]}"
    run_name="${RUN_PREFIX}_${group_name}_s${idx}_w${worker}"
    session_name="orch_${run_name}"
    remote_cmd="cd '$REMOTE_ROOT' && mkdir -p 'runs/$run_name' && tmux kill-session -t '$session_name' >/dev/null 2>&1 || true && tmux new-session -d -s '$session_name' \"cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' MODEL_NAME='$MODEL_NAME' SERVED_MODEL_NAME='$SERVED_MODEL_NAME' TOKENIZER_NAME='$TOKENIZER_NAME' RUN_NAME='$run_name' PORT='$PORT' TP_SIZE='$TP_SIZE' MAX_MODEL_LEN='$MAX_MODEL_LEN' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' MAX_NUM_SEQS='$MAX_NUM_SEQS' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' BENCHMARK_DATASET_NAME='$BENCHMARK_DATASET_NAME' BENCHMARK_DATASET_CONFIG_NAME='$BENCHMARK_DATASET_CONFIG_NAME' MAX_BENCHMARK='$MAX_BENCHMARK' BENCHMARK_NUM_ROLLOUTS='$BENCHMARK_NUM_ROLLOUTS' BENCHMARK_ROLLOUT_TEMPERATURE='$BENCHMARK_ROLLOUT_TEMPERATURE' BENCHMARK_ROLLOUT_TOP_P='$BENCHMARK_ROLLOUT_TOP_P' BENCHMARK_ROLLOUT_TOP_K='$BENCHMARK_ROLLOUT_TOP_K' BENCHMARK_ROLLOUT_PASS_K='$BENCHMARK_ROLLOUT_PASS_K' BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN='$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN' REQUEST_CONCURRENCY='$REQUEST_CONCURRENCY' REQUEST_TIMEOUT_SECONDS='$REQUEST_TIMEOUT_SECONDS' MAX_PROMPT_LEN='$MAX_PROMPT_LEN' THINKING_MODE='$thinking_mode' PROBLEM_SHARD_INDEX='$idx' PROBLEM_SHARD_COUNT='$shard_count' USE_WANDB='$USE_WANDB' WANDB_GROUP='${RUN_PREFIX}_${group_name}' bash scripts/run_use1d_vllm_remote_eval.sh > 'runs/$run_name/orchestrator.log' 2>&1\""
    remote_ssh "$worker" "$remote_cmd"
    log "launched worker=$worker run_name=$run_name session=$session_name"
  done
}

launch_group think think "${THINK_WORKERS[@]}"
launch_group nothink no_think "${NOTHINK_WORKERS[@]}"

log "remote orchestration launched"
log "rerun with the same RUN_PREFIX to resume from per-problem checkpoints"
