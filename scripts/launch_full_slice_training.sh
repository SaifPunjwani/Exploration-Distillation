#!/usr/bin/env bash
# Full-slice training orchestrator: vLLM on all 16 hosts + training on worker 0
#
# Architecture:
#   Workers 1-15: vLLM inference servers (TP=4 each)
#   Worker 0: GRPO training pipeline
#
# The training pipeline on worker 0 sends generation requests to the inference
# workers, then does GRPO/SFT updates locally.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh
resolve_tpu_target_defaults
resolve_tpu_worker_env

TPU_NAME="${TPU_NAME}"
ZONE="${ZONE}"
TPU_TRANSPORT_FLAG="$(expdis_tpu_transport_flag)"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}"

# Model
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"

# vLLM server config
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"

# Training config
DATASET_NAME="${DATASET_NAME:-gsm8k}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-1024}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-16}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-128}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-32}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-16}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-4}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}"

# Actual-stage config
ACTUAL_TARGET_CONTRACT="${ACTUAL_TARGET_CONTRACT:-canonical}"
ACTUAL_DATASET_POLICY="${ACTUAL_DATASET_POLICY:-accepted_only}"
ACTUAL_REWARD_SOURCE="${ACTUAL_REWARD_SOURCE:-correctness}"

# Orchestration
ALLOW_DIRTY="${ALLOW_DIRTY:-1}"
SYNC_REPO="${SYNC_REPO:-1}"
BOOTSTRAP_VLLM="${BOOTSTRAP_VLLM:-1}"
BOOTSTRAP_PARALLELISM="${BOOTSTRAP_PARALLELISM:-4}"
VLLM_ENV_CACHE_GCS_URI="${VLLM_ENV_CACHE_GCS_URI:-}"
RUN_NAME="${RUN_NAME:-fullslice_$(date +%Y%m%d_%H%M%S)}"
TRAINER_WORKER="${TRAINER_WORKER:-0}"
INFERENCE_START_WORKER="${INFERENCE_START_WORKER:-1}"
ALLOW_PARTIAL_VLLM_SLICE="${ALLOW_PARTIAL_VLLM_SLICE:-0}"
MIN_READY_SERVERS="${MIN_READY_SERVERS:-}"
# Training device: "cpu" (default, legacy) or "tpu" (4-chip FSDP on trainer worker)
TRAINING_DEVICE="${TRAINING_DEVICE:-cpu}"

if [ -z "$MAX_MODEL_LEN" ]; then
  MAX_MODEL_LEN="$((MAX_PROMPT_LEN + MAX_COMPLETION_LEN + 256))"
fi

log() { printf '[full-slice] %s\n' "$*"; }

# ── Get worker list ─────────────────────────────────────────────────
worker_count="$(python3 -c "
import json, subprocess
desc = json.loads(subprocess.check_output([
    'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'describe',
    '$TPU_NAME', '--zone=$ZONE', '--format=json'
]))
eps = desc.get('networkEndpoints', [])
print(len(eps))
")"

worker_ips="$(python3 -c "
import json, subprocess
desc = json.loads(subprocess.check_output([
    'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'describe',
    '$TPU_NAME', '--zone=$ZONE', '--format=json'
]))
eps = desc.get('networkEndpoints', [])
print(','.join(ep['ipAddress'] for ep in eps))
")"

log "TPU=$TPU_NAME workers=$worker_count IPs=$worker_ips"
if [ "$TRAINER_WORKER" -lt 0 ] || [ "$TRAINER_WORKER" -ge "$worker_count" ]; then
  log "ERROR: TRAINER_WORKER=$TRAINER_WORKER is outside worker range 0..$((worker_count - 1))"
  exit 1
fi
if [ "$INFERENCE_START_WORKER" -le "$TRAINER_WORKER" ]; then
  INFERENCE_START_WORKER=$((TRAINER_WORKER + 1))
fi
if [ "$INFERENCE_START_WORKER" -ge "$worker_count" ]; then
  log "ERROR: No workers left for inference after reserving trainer worker $TRAINER_WORKER"
  exit 1
fi
inference_worker_count=$((worker_count - INFERENCE_START_WORKER))
inference_worker_ips="$(python3 -c "
import json, subprocess
desc = json.loads(subprocess.check_output([
    'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'describe',
    '$TPU_NAME', '--zone=$ZONE', '--format=json'
]))
eps = desc.get('networkEndpoints', [])
start = int('$INFERENCE_START_WORKER')
print(','.join(ep['ipAddress'] for ep in eps[start:]))
")"
if [ -z "$MIN_READY_SERVERS" ]; then
  MIN_READY_SERVERS="$inference_worker_count"
fi
log "trainer_worker=$TRAINER_WORKER inference_workers=$INFERENCE_START_WORKER-$((worker_count - 1)) min_ready_servers=$MIN_READY_SERVERS"

# ── Step 1: Sync repo to all workers ───────────────────────────────
if [ "$SYNC_REPO" = "1" ]; then
  log "Syncing repo to all $worker_count workers..."
  TPU_NAME="$TPU_NAME" ZONE="$ZONE" SYNC_WORKERS="all" ALLOW_DIRTY="$ALLOW_DIRTY" \
    bash scripts/sync_tpu_slice_repo.sh
fi

# ── Step 2: Bootstrap vLLM on all workers ──────────────────────────
if [ "$BOOTSTRAP_VLLM" = "1" ]; then
  log "Bootstrapping vLLM on all workers (parallelism=$BOOTSTRAP_PARALLELISM)..."

  # Primary worker first (builds env, caches to GCS)
  log "  worker 0: bootstrapping primary vLLM env"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker=0 --zone="$ZONE" "$TPU_TRANSPORT_FLAG" \
    --command="cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' bash scripts/bootstrap_vllm_tpu_env.sh"

  # Remaining workers in parallel (restore from GCS cache)
  active=0
  for w in $(seq 1 $((worker_count - 1))); do
    log "  worker $w: bootstrapping vLLM from cache"
    (
      gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
        --worker="$w" --zone="$ZONE" "$TPU_TRANSPORT_FLAG" \
        --command="cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' bash scripts/bootstrap_vllm_tpu_env.sh"
    ) &
    active=$((active + 1))
    if [ "$active" -ge "$BOOTSTRAP_PARALLELISM" ]; then
      wait
      active=0
    fi
  done
  wait
  log "vLLM bootstrapped on all workers"
fi

# ── Step 2b: Bootstrap training venv on worker 0 ──────────────────
BOOTSTRAP_TRAINING="${BOOTSTRAP_TRAINING:-1}"
if [ "$BOOTSTRAP_TRAINING" = "1" ]; then
  log "Bootstrapping training venv on worker 0..."
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker=0 --zone="$ZONE" "$TPU_TRANSPORT_FLAG" \
    --command="cd '$REMOTE_ROOT' && VENV_DIR=.venv_tpu_slice USE_TPU=1 bash scripts/bootstrap_tpu_env.sh"
  log "Training venv ready on worker 0"
fi

# ── Step 3: Launch vLLM servers on all workers ─────────────────────
log "Launching vLLM servers on $inference_worker_count inference workers..."
for w in $(seq "$INFERENCE_START_WORKER" $((worker_count - 1))); do
  session="vllm_server_${RUN_NAME}_w${w}"
  log "  worker $w: starting vLLM server (session=$session)"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker="$w" --zone="$ZONE" "$TPU_TRANSPORT_FLAG" \
    --command="cd '$REMOTE_ROOT' && tmux kill-session -t '$session' 2>/dev/null || true && tmux new-session -d -s '$session' \"VLLM_VENV_DIR='$VLLM_VENV_DIR' MODEL_NAME='$MODEL_NAME' SERVED_MODEL_NAME='$SERVED_MODEL_NAME' PORT='$PORT' TP_SIZE='$TP_SIZE' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_SEQS='$MAX_NUM_SEQS' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' bash scripts/run_vllm_server.sh > /tmp/vllm_server_${RUN_NAME}.log 2>&1\""
done

# ── Step 4: Wait for vLLM servers to be healthy ────────────────────
# Health checks run on worker 0 (which can reach internal IPs)
log "Waiting for vLLM servers to become healthy (via worker $TRAINER_WORKER)..."
VLLM_SERVER_URLS=""

health_check_script="
IPS='$inference_worker_ips'
PORT='$PORT'
TOTAL='$inference_worker_count'
MIN_READY='$MIN_READY_SERVERS'
ALLOW_PARTIAL='$ALLOW_PARTIAL_VLLM_SLICE'
for attempt in \$(seq 1 60); do
  ready=0
  urls=''
  IFS=',' read -r -a ip_array <<< \"\$IPS\"
  for ip in \"\${ip_array[@]}\"; do
    if curl -s --connect-timeout 3 \"http://\${ip}:\${PORT}/health\" >/dev/null 2>&1; then
      ready=\$((ready + 1))
      urls=\"\${urls:+\$urls,}http://\${ip}:\${PORT}/v1\"
    fi
  done
  echo \"[health] attempt \$attempt: \$ready/\$TOTAL servers healthy\"
  if [ \"\$ready\" -ge \"\$TOTAL\" ]; then
    echo \"URLS=\$urls\"
    exit 0
  fi
  if [ \"\$ALLOW_PARTIAL\" = \"1\" ] && [ \"\$ready\" -ge \"\$MIN_READY\" ] && [ \"\$attempt\" -ge 30 ]; then
    echo \"URLS=\$urls\"
    echo \"[health] proceeding with \$ready/\$TOTAL after \$attempt attempts\"
    exit 0
  fi
  sleep 10
done
echo '[health] ERROR: No servers became healthy'
exit 1
"

health_output="$(gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --worker="$TRAINER_WORKER" --zone="$ZONE" "$TPU_TRANSPORT_FLAG" \
  --command="$health_check_script")"

echo "$health_output" | grep -v '^URLS=' | while read -r line; do log "$line"; done
VLLM_SERVER_URLS="$(echo "$health_output" | grep '^URLS=' | head -1 | sed 's/^URLS=//')"

if [ -z "$VLLM_SERVER_URLS" ]; then
  log "ERROR: No vLLM servers became healthy"
  exit 1
fi

log "vLLM servers ready: $(echo "$VLLM_SERVER_URLS" | tr ',' '\n' | wc -l) servers"
healthy_server_count="$(echo "$VLLM_SERVER_URLS" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' ')"
EXPDIS_VLLM_CONCURRENCY_VALUE="${EXPDIS_VLLM_CONCURRENCY:-}"
if [ -z "$EXPDIS_VLLM_CONCURRENCY_VALUE" ]; then
  target_concurrency=$((GRPO_BATCH_SIZE * 2))
  concurrency_cap=$((healthy_server_count * 4))
  if [ "$target_concurrency" -le "$concurrency_cap" ]; then
    EXPDIS_VLLM_CONCURRENCY_VALUE="$target_concurrency"
  else
    EXPDIS_VLLM_CONCURRENCY_VALUE="$concurrency_cap"
  fi
  if [ "$EXPDIS_VLLM_CONCURRENCY_VALUE" -lt 4 ]; then
    EXPDIS_VLLM_CONCURRENCY_VALUE=4
  fi
fi
log "Derived EXPDIS_VLLM_CONCURRENCY=$EXPDIS_VLLM_CONCURRENCY_VALUE from healthy_server_count=$healthy_server_count"

# ── Step 5: Launch training on worker 0 ────────────────────────────
RUN_DIR="runs/$RUN_NAME"
log "Launching training pipeline on worker $TRAINER_WORKER (run=$RUN_NAME)..."

# Common environment exports for training
COMMON_EXPORTS="export WANDB_PROJECT=expdis && \\
export WANDB_ENTITY='${WANDB_ENTITY:-}' && \\
export TOKENIZERS_PARALLELISM=false && \\
export PYTHONUNBUFFERED=1 && \\
export EXPDIS_GRPO_GRAD_CKPT='${EXPDIS_GRPO_GRAD_CKPT:-0}' && \\
export EXPDIS_GRPO_LOGGING_STRATEGY=steps && \\
export EXPDIS_GRPO_LOGGING_STEPS=1 && \\
export EXPDIS_GRPO_SAVE_STRATEGY='${EXPDIS_GRPO_SAVE_STRATEGY:-no}' && \\
export EXPDIS_GRPO_SAVE_STEPS='${EXPDIS_GRPO_SAVE_STEPS:-16}' && \\
export EXPDIS_SKIP_INTERMEDIATE_EVALS=0 && \\
export EXPDIS_DISABLE_BF16='${EXPDIS_DISABLE_BF16:-0}' && \\
export EXPDIS_ENABLE_THINKING='${EXPDIS_ENABLE_THINKING:-false}' && \\
export EXPDIS_OPTIM='${EXPDIS_OPTIM:-adamw_torch}' && \\
export EXPDIS_DEBUG_GENERATE='${EXPDIS_DEBUG_GENERATE:-0}' && \\
export EXPDIS_ABORT_ON_DEGENERATE_EXPLORER='${EXPDIS_ABORT_ON_DEGENERATE_EXPLORER:-1}' && \\
export EXPDIS_ABORT_ON_BAD_TRAJ_WINDOW='${EXPDIS_ABORT_ON_BAD_TRAJ_WINDOW:-1}' && \\
export EXPDIS_FEW_SHOT_EXAMPLES='${EXPDIS_FEW_SHOT_EXAMPLES:-0}' && \\
export EXPDIS_VLLM_SERVER_URLS='$VLLM_SERVER_URLS' && \\
export EXPDIS_VLLM_MODEL_NAME='$SERVED_MODEL_NAME' && \\
export EXPDIS_VLLM_CONCURRENCY='$EXPDIS_VLLM_CONCURRENCY_VALUE' && \\
export EXPDIS_GCS_CHECKPOINT_BASE='${EXPDIS_GCS_CHECKPOINT_BASE:-gs://llmvae-checkpoints/two-model-exploration/checkpoints}'"

# run_expdis_torch.py arguments (shared between CPU and TPU modes)
MAIN_ARGS="--use-wandb \\
  --seed 0 \\
  --model-name '$MODEL_NAME' \\
  --dataset-name '$DATASET_NAME' \\
  --train-selection-policy random \\
  --train-probe-pool-size 256 \\
  --train-probe-samples-per-prompt 4 \\
  --train-probe-temperature 0.8 \\
  --train-probe-batch-size 4 \\
  --dataset-split-policy auto \\
  --topic-split-mode heuristic_math_topics \\
  --topic-partition-policy explicit_or_heuristic \\
  --decoupling-mode decoupled \\
  --base-eval-mode raw \\
  --actual-target-contract '$ACTUAL_TARGET_CONTRACT' \\
  --max-train-examples $MAX_TRAIN_EXAMPLES \\
  --max-eval-examples $MAX_EVAL_EXAMPLES \\
  --grpo-max-steps $GRPO_MAX_STEPS \\
  --grpo-batch-size $GRPO_BATCH_SIZE \\
  --grpo-loss-type dr_grpo \\
  --grpo-num-generations $GRPO_NUM_GENERATIONS \\
  --grpo-generation-batch-size $GRPO_GENERATION_BATCH_SIZE \\
  --num-samples-per-prompt $NUM_SAMPLES_PER_PROMPT \\
  --answer-constraint-mode none \\
  --exploration-response-style reasoning_final \\
  --eval-response-style reasoning_final \\
  --exploration-answer-constraint-mode none \\
  --eval-answer-constraint-mode none \\
  --distill-epochs 2 \\
  --distill-batch-size 1 \\
  --max-distill-examples $MAX_TRAIN_EXAMPLES \\
  --actual-grpo-epochs 3 \\
  --actual-grpo-batch-size 1 \\
  --actual-grpo-grad-accum 1 \\
  --actual-min-accepted-trajectories 4 \\
  --actual-dataset-policy '$ACTUAL_DATASET_POLICY' \\
  --actual-max-incorrect-ratio 0.25 \\
  --actual-target-correct-fraction 0.75 \\
  --actual-grpo-kl-coef 0.10 \\
  --actual-grpo-lr 1e-6 \\
  --distill-lr 5e-6 \\
  --max-actual-examples $MAX_TRAIN_EXAMPLES \\
  --novelty-feature-source last \\
  --novelty-feature-max-length 128 \\
  --novelty-metric sqrt_mse \\
  --novelty-text-contract full_raw \\
  --rnd-hidden 512 \\
  --reward-correct 1.0 \\
  --reward-incorrect 0.0 \\
  --lambda-novelty 0.03 \\
  --reward-shape linear \\
  --anchor-metric none \\
  --trajectory-batch-size 1 \\
  --eval-batch-size 1 \\
  --round-budget-mode constant_total \\
  --max-prompt-len $MAX_PROMPT_LEN \\
  --max-completion-len $MAX_COMPLETION_LEN \\
  --exploration-max-completion-len $MAX_COMPLETION_LEN \\
  --eval-max-completion-len $MAX_COMPLETION_LEN \\
  --max-total-len $((MAX_PROMPT_LEN + MAX_COMPLETION_LEN)) \\
  --format-penalty-weight 0.1 \\
  --pipeline-mode two_model \\
  --actual-init-source base \\
  --actual-train-mode sft_grpo \\
  --actual-reward-source '$ACTUAL_REWARD_SOURCE' \\
  --wandb-run-name '$RUN_NAME' \\
  --plots-dir '$RUN_DIR/plots' \\
  --explorer-traj-path '$RUN_DIR/explorer_trajectories.jsonl' \\
  --explorer-output-dir '$RUN_DIR/explorer_ckpts' \\
  --saved-model-root '$RUN_DIR/saved_models'"

if [ "$TRAINING_DEVICE" = "tpu" ]; then
  log "Training with TPU FSDP v2/SPMD (4-chip on worker $TRAINER_WORKER)"
  # FSDP v2 uses SPMD: single process sees all 4 chips via xr.use_spmd()
  # Isolation env vars prevent libtpu from trying to coordinate with other workers
  TRAIN_CMD="cd '$REMOTE_ROOT' && source .venv_tpu_slice/bin/activate && \\
$COMMON_EXPORTS && \\
export PJRT_DEVICE=TPU && \\
export EXPDIS_FSDP=1 && \\
export EXPDIS_USE_XLA_WORLD_SIZE=1 && \\
export EXPDIS_FSDP_WRAP_CLS=Qwen3DecoderLayer && \\
export EXPDIS_TPU_SLICE_LAUNCHED=1 && \\
export EXPDIS_REQUIRE_TPU=1 && \\
export EXPDIS_ALLOW_SINGLE_WORKER_TPU=1 && \\
export TPU_SKIP_MDS_QUERY=1 && \\
export TPU_WORKER_HOSTNAMES=localhost,localhost,localhost,localhost && \\
export TPU_WORKER_ID=0 && \\
export TPU_HOST_BOUNDS=2,2,1 && \\
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 && \\
export TPU_PROCESS_BOUNDS=1,1,1 && \\
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 && \\
export TPU_ACCELERATOR_TYPE='${TPU_ACCELERATOR_TYPE:-v5litepod-64}' && \\
export EXPDIS_XLA_PERSISTENT_CACHE_DIR=/tmp/expdis_xla_cache_fsdp_spmd && \\
mkdir -p '$RUN_DIR' && \\
python -u run_expdis_torch.py \\
  --device tpu \\
  $MAIN_ARGS \\
  2>&1 | tee -a '$RUN_DIR/console.log'"
else
  log "Training with CPU on worker $TRAINER_WORKER"
  TRAIN_CMD="cd '$REMOTE_ROOT' && source .venv_tpu_slice/bin/activate && \\
$COMMON_EXPORTS && \\
export PJRT_DEVICE=CPU && \\
mkdir -p '$RUN_DIR' && \\
python -u run_expdis_torch.py \\
  --device cpu \\
  $MAIN_ARGS \\
  2>&1 | tee -a '$RUN_DIR/console.log'"
fi

session="training_${RUN_NAME}"
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --worker="$TRAINER_WORKER" --zone="$ZONE" "$TPU_TRANSPORT_FLAG" \
  --command="tmux kill-session -t '$session' 2>/dev/null || true; tmux new-session -d -s '$session' \"$TRAIN_CMD\""

log "Training launched in tmux session '$session' on worker $TRAINER_WORKER"
log "Monitor: gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=$TRAINER_WORKER --zone=$ZONE --command='tmux capture-pane -t $session -p -S -50'"
