#!/usr/bin/env bash
# Orchestrate 2 sequential full-pipeline runs on a single v6e-64 TPU slice.
# Run 1: lambda=0 (correctness only)
# Run 2: lambda=0.5 (with novelty)
# vLLM stays on workers 1-15, training on worker 0.
# After Run 1 finishes, Run 2 reuses the same vLLM servers (SKIP bootstrapping).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

TARGET_PRESET="${TARGET_PRESET:-use1d}"
case "$TARGET_PRESET" in
  use1d)
    TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
    TRAIN_ZONE="${TRAIN_ZONE:-us-east1-d}"
    TRAIN_SSH_HOST="${TRAIN_SSH_HOST:?set TRAIN_SSH_HOST to the ssh host alias of the training TPU VM}"
    ;;
  ew4a)
    TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
    TRAIN_ZONE="${TRAIN_ZONE:-europe-west4-a}"
    TRAIN_SSH_HOST="${TRAIN_SSH_HOST:?set TRAIN_SSH_HOST to the ssh host alias of the training TPU VM}"
    ;;
  v4)
    TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
    TRAIN_ZONE="${TRAIN_ZONE:-us-central2-b}"
    TRAIN_SSH_HOST="${TRAIN_SSH_HOST:?set TRAIN_SSH_HOST to the ssh host alias of the training TPU VM}"
    ;;
  *)
    echo "Unsupported TARGET_PRESET=$TARGET_PRESET" >&2
    exit 1
    ;;
esac

TRAIN_REMOTE_ROOT="${TRAIN_REMOTE_ROOT:?set TRAIN_REMOTE_ROOT to the absolute repo checkout path on the training TPU VM}"

BASE_ENV=(
  "PATH=$PATH"
  "HOME=$HOME"
  "USER=${USER:-saifpunjwani}"
  "LOGNAME=${LOGNAME:-${USER:-saifpunjwani}}"
  "SHELL=${SHELL:-/bin/bash}"
  "TERM=${TERM:-xterm-256color}"
  "LANG=${LANG:-C.UTF-8}"
  "LC_ALL=${LC_ALL:-C.UTF-8}"
  "TMPDIR=${TMPDIR:-/tmp}"
  "CLOUDSDK_CONFIG=${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}"
  "XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-$HOME/.config}"
  "SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-}"
  "GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-}"
  "WANDB_API_KEY=${WANDB_API_KEY:-}"
  "HF_ACCESS_TOKEN=${HF_ACCESS_TOKEN:-}"
  "HF_TOKEN=${HF_TOKEN:-}"
  "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}"
)

train_worker_ssh() {
  local cmd="$1"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$TRAIN_SSH_HOST" "$cmd"
}

wait_for_run_exit() {
  local run_name="$1"
  local status
  while true; do
    status="$(train_worker_ssh "if tmux has-session -t 'training_${run_name}' 2>/dev/null; then echo RUNNING; elif [ -f '$TRAIN_REMOTE_ROOT/runs/${run_name}/exit_status.txt' ]; then printf 'EXIT='; cat '$TRAIN_REMOTE_ROOT/runs/${run_name}/exit_status.txt'; else echo MISSING; fi" 2>/dev/null || true)"
    case "$status" in
      RUNNING)
        sleep 60
        ;;
      EXIT=0)
        echo "Run $run_name finished successfully at $(date)"
        return 0
        ;;
      EXIT=*)
        echo "Run $run_name failed with $status" >&2
        return 1
        ;;
      *)
        sleep 10
        ;;
    esac
  done
}

launch_clean() {
  env -i "${BASE_ENV[@]}" "${COMMON_VARS[@]}" "$@" bash scripts/launch_dual_slice_training.sh
}

launch_logged() {
  local log_file="$1"
  shift
  mkdir -p runs_logs
  launch_clean "$@" 2>&1 | tee "$log_file"
}

# Common settings for both runs.
COMMON_VARS=(
  TRAIN_TPU_NAME="$TRAIN_TPU_NAME"
  TRAIN_ZONE="$TRAIN_ZONE"
  TRAIN_SSH_HOST="$TRAIN_SSH_HOST"
  TRAIN_REMOTE_ROOT="$TRAIN_REMOTE_ROOT"
  TRAIN_WORKER=0
  TRAIN_LAUNCH_MODE=fsdp_local
  SINGLE_SLICE_MODE=1
  STRICT_LAUNCH_CONTRACT=1
  EXPECTED_TRAIN_TPU_NAME="$TRAIN_TPU_NAME"
  EXPECTED_TRAIN_ZONE="$TRAIN_ZONE"
  EXPECTED_TRAIN_SSH_HOST="$TRAIN_SSH_HOST"
  EXPECTED_TRAIN_WORKER=0
  EXPECTED_TRAIN_LAUNCH_MODE=fsdp_local
  EXPECTED_SINGLE_SLICE_MODE=1
  FSDP_VISIBLE_CHIPS=0,1,2,3
  FSDP_NUM_PROCESSES=4
  DDP_VISIBLE_CHIPS=0
  EXPECTED_DDP_VISIBLE_CHIPS=0
  DDP_NUM_PROCESSES=1
  EXPECTED_DDP_NUM_PROCESSES=1
  MAX_PROMPT_LEN=2048
  EXPECTED_MAX_PROMPT_LEN=2048
  MAX_COMPLETION_LEN=8192
  EXPECTED_MAX_COMPLETION_LEN=8192
  EXPLORATION_MAX_COMPLETION_LEN=8192
  EXPECTED_EXPLORATION_MAX_COMPLETION_LEN=8192
  EVAL_MAX_COMPLETION_LEN=8192
  EXPECTED_EVAL_MAX_COMPLETION_LEN=8192
  TRAJ_MAX_COMPLETION_LEN=8192
  MAX_TOTAL_LEN=10240
  EXPECTED_MAX_TOTAL_LEN=10240
  MAX_MODEL_LEN=10240
  GRPO_MAX_STEPS=32
  EXPECTED_GRPO_MAX_STEPS=32
  MAX_TRAIN_EXAMPLES=20000
  EXPECTED_MAX_TRAIN_EXAMPLES=20000
  MAX_EVAL_EXAMPLES=32
  EXPECTED_MAX_EVAL_EXAMPLES=32
  GRPO_BATCH_SIZE=1
  EXPECTED_GRPO_BATCH_SIZE=1
  GRPO_GRAD_ACCUM=2
  EXPECTED_GRPO_GRAD_ACCUM=2
  GRPO_NUM_GENERATIONS=8
  EXPECTED_GRPO_NUM_GENERATIONS=8
  GRPO_GENERATION_BATCH_SIZE=8
  EXPECTED_GRPO_GENERATION_BATCH_SIZE=8
  NUM_SAMPLES_PER_PROMPT=8
  EXPECTED_NUM_SAMPLES_PER_PROMPT=8
  TRAJECTORY_BATCH_SIZE=32
  EXPECTED_TRAJECTORY_BATCH_SIZE=32
  EVAL_BATCH_SIZE=1
  EXPECTED_EVAL_BATCH_SIZE=1
  DISTILL_BATCH_SIZE=1
  ACTUAL_ONLINE_GRPO_MAX_STEPS=32
  DATASET_NAME=dapo_math_17k
  ACTUAL_TARGET_CONTRACT=full_trace
  ACTUAL_DATASET_POLICY=accepted_only
  ACTUAL_REWARD_SOURCE=correctness
  NOVELTY_FEATURE_SOURCE=multilayer
  NOVELTY_FEATURE_MAX_LENGTH=10240
  EXPDIS_FSDP=1
  EXPECTED_EXPDIS_FSDP=1
  EXPDIS_USE_XLA_WORLD_SIZE=1
  EXPECTED_EXPDIS_USE_XLA_WORLD_SIZE=1
  EXPDIS_EXPLORER_DECOUPLED_TRAIN_FSDP=1
  EXPDIS_ALLOW_SINGLE_WORKER_TPU=1
  EXPDIS_WANDB_MINIMAL=0
  EXPDIS_OPTIM=adamw_torch
  EXPDIS_DECOUPLED_TRAIN_TOKEN_CHUNK=0
  EXPDIS_DECOUPLED_GRAD_CKPT=1
  EXPDIS_SKIP_VLLM_LOGPROBS=0
  EXPECTED_EXPDIS_SKIP_VLLM_LOGPROBS=0
  # Reload temporarily disabled — restart path still leaves workers dead after
  # round 1. Cut Explorer to 128 steps so we stay near the pre-collapse zone
  # from 5:14 baseline and complete the E2E pipeline tonight without reload.
  EXPDIS_EXPLORER_VLLM_RELOAD_EVERY_N_ROUNDS=0
  EXPECTED_EXPDIS_EXPLORER_VLLM_RELOAD_EVERY_N_ROUNDS=0
  EXPDIS_EXPLORER_VLLM_RELOAD_WORKERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
  EXPDIS_TRAJ_USE_VLLM=1
  EXPECTED_EXPDIS_TRAJ_USE_VLLM=1
  EXPDIS_XLA_FLASH_ATTENTION=1
  EXPECTED_EXPDIS_XLA_FLASH_ATTENTION=1
  EXPDIS_XLA_FORCE_EAGER_ATTN=0
  EXPECTED_EXPDIS_XLA_FORCE_EAGER_ATTN=0
  EXPDIS_ENABLE_THINKING=auto
  EXPECTED_EXPDIS_ENABLE_THINKING=auto
  EXPDIS_STRICT_GRPO_GEOMETRY=1
  EXPECTED_EXPDIS_STRICT_GRPO_GEOMETRY=1
  EXPLORATION_THINKING_MODE=auto
  EVAL_THINKING_MODE=auto
  ACTUAL_THINKING_MODE=auto
  BENCHMARK_DATASET_NAME=aime_2024
  BENCHMARK_NUM_ROLLOUTS=32
  MAX_BENCHMARK_EXAMPLES=30
  VLLM_ENV_CACHE_GCS_URI="gs://llmvae-checkpoints/two-model-exploration/vllm_env_cache_v6e"
  BOOTSTRAP_PARALLELISM=8
  VLLM_LAUNCH_PARALLELISM=8
  # Robustness: push pipeline through soft-gate failures so we always get an
  # Actual checkpoint at the end, even if intermediate data looks bad.
  EXPDIS_ABORT_ON_DEGENERATE_EXPLORER=0
  EXPDIS_ABORT_ON_BAD_TRAJ_WINDOW=0
  # Distill eager attention: avoids the torch_xla rank-mismatch crash we see
  # with flash attention on the SFT path after Explorer completes.
  EXPDIS_DISTILL_EAGER_ATTN=1
  # Keep reloaded vLLM under the same served model name as bootstrap so the
  # training client (which cached Qwen/Qwen3-1.7B at startup) doesn't start
  # sending 404-triggering requests to the new server.
  EXPDIS_EXPLORER_VLLM_MODEL_NAME="Qwen/Qwen3-1.7B"
)

RUN1_NAME="pipeline_lambda0_${TIMESTAMP}"
RUN2_NAME="pipeline_lambda05_${TIMESTAMP}"

mkdir -p runs_logs
cat > runs_logs/active_runs.env <<EOF
RUN1_NAME=$RUN1_NAME
RUN2_NAME=$RUN2_NAME
TRAIN_TPU=$TRAIN_TPU_NAME
TRAIN_ZONE=$TRAIN_ZONE
TRAIN_SSH_HOST=$TRAIN_SSH_HOST
TARGET_PRESET=$TARGET_PRESET
EOF

echo "============================================"
echo " RUN 1: lambda=0 (correctness only)"
echo " Started: $(date)"
echo "============================================"

# Pre-flight: disk cleanup on TPU worker 0 to prevent safetensors-save crashes mid-training
echo "[pre-flight] Cleaning disk on TPU worker 0..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 "$TRAIN_SSH_HOST" "
  df -h / | head -2
  # Delete old run directories (keep only last 2)
  cd '$TRAIN_REMOTE_ROOT/runs' 2>/dev/null && \
    ls -1t | tail -n +3 | xargs -I {} rm -rf {} 2>/dev/null || true
  # Nuke wandb runs older than last one (wandb fills disk with log data over time)
  cd '$TRAIN_REMOTE_ROOT/wandb' 2>/dev/null && \
    ls -1dt run-* 2>/dev/null | tail -n +2 | xargs -I {} rm -rf {} 2>/dev/null || true
  # Delete XLA cache if disk >70%
  used=\$(df / | awk 'NR==2 {gsub(/%/,\"\",\$5); print \$5}')
  if [ \"\${used:-0}\" -gt 70 ]; then
    echo '[pre-flight] disk >70%, pruning XLA cache'
    rm -rf /tmp/expdis_xla_cache_shared/* 2>/dev/null || true
    find /dev/shm/xla_cache_overflow -type f -delete 2>/dev/null || true
  fi
  # Clean old expdis_model_save dirs
  rm -rf /tmp/expdis_model_save_* 2>/dev/null || true
  rm -rf /tmp/vllm_explorer_model_* /tmp/vllm_actual_model_* 2>/dev/null || true
  df -h / | head -2
" 2>&1 | tail -10 || echo "[pre-flight] cleanup SSH failed (TPU may be starting up)"

RUN1_DEPLOY_LOG="runs_logs/deploy_${RUN1_NAME}.log"
launch_logged "$RUN1_DEPLOY_LOG" \
  RUN_NAME="$RUN1_NAME" \
  LAMBDA_NOVELTY=0 \
  EXPDIS_NOVELTY_ON_TPU=0

echo ""
echo "============================================"
echo " RUN 1 LAUNCHED. Waiting for completion..."
echo " Training is on the TPU in tmux."
echo "============================================"

wait_for_run_exit "$RUN1_NAME"

echo ""
echo "============================================"
echo " RUN 2: lambda=0.5 (with novelty)"
echo " Started: $(date)"
echo "============================================"

RUN2_DEPLOY_LOG="runs_logs/deploy_${RUN2_NAME}.log"

# Run 2: reuse existing vLLM servers (skip bootstrap/launch).
# Discover the vLLM URLs from the running servers.
VLLM_URLS="$(train_worker_ssh '
  for i in $(seq 1 15); do
    ip=$(getent hosts "$(hostname | sed "s/-0$/-$i/")" 2>/dev/null | awk "{print \$1}" || true)
    if [ -n "$ip" ]; then
      if curl -s --connect-timeout 2 "http://${ip}:8000/health" >/dev/null 2>&1; then
        printf "http://%s:8000/v1," "$ip"
      fi
    fi
  done
' 2>/dev/null | sed 's/,$//')"

if [ -z "$VLLM_URLS" ]; then
  echo "WARNING: Could not discover vLLM URLs from Run 1. Re-bootstrapping."
  launch_logged "$RUN2_DEPLOY_LOG" \
    RUN_NAME="$RUN2_NAME" \
    LAMBDA_NOVELTY=0.5 \
    EXPDIS_NOVELTY_ON_TPU=1
else
  echo "Reusing ${#VLLM_URLS} vLLM servers from Run 1"
  launch_logged "$RUN2_DEPLOY_LOG" \
    RUN_NAME="$RUN2_NAME" \
    LAMBDA_NOVELTY=0.5 \
    EXPDIS_NOVELTY_ON_TPU=1 \
    VLLM_SERVER_URLS="$VLLM_URLS" \
    SKIP_INFERENCE_SLICE=0 \
    BOOTSTRAP_VLLM=0 \
    LAUNCH_VLLM=0 \
    VLLM_PREFETCH_MODEL=0 \
    SYNC_INFER_REPO=0
fi

echo ""
echo "============================================"
echo " RUN 2 LAUNCHED at $(date)"
echo " Both runs submitted."
echo "============================================"
