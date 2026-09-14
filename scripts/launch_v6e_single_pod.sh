#!/usr/bin/env bash
# Single-pod full pipeline: vLLM on workers 1-15, TPU training on worker 0.
# Target: v6e-64 (64 chips, 16 hosts, 4 chips/host, 32GB HBM/chip)
#
# Architecture:
#   - Workers 1-15: vLLM inference (TP=4 each, 60 chips)
#   - Worker 0: TPU training with flash attention (1 chip, 32GB HBM)
#   - No gradient checkpointing needed — 32GB fits 8k easily (~30% faster than v5lite)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh

# === Pod target (single pod for both inference and training) ===
POD_TPU_NAME="${POD_TPU_NAME:?set POD_TPU_NAME to the TPU pod name}"
POD_ZONE="${POD_ZONE:-us-east1-d}"
POD_SSH_HOST="${POD_SSH_HOST:?set POD_SSH_HOST to the ssh host alias of the TPU pod}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}"

TRANSPORT_FLAG="$(tmx_tpu_transport_flag)"
ALLOW_DIRTY="${ALLOW_DIRTY:-1}"
RUN_NAME="${RUN_NAME:-v6e_single_pod_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"
RUN_ARTIFACT_GCS_PREFIX="${RUN_ARTIFACT_GCS_PREFIX:-gs://two-model-exploration-checkpoints/v6e_single_pod}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
DATASET_NAME="${DATASET_NAME:-dapo_math_17k}"

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
RELOAD_WORKER="${RELOAD_WORKER:-1}"

SYNC_REPO="${SYNC_REPO:-1}"
BOOTSTRAP_VLLM="${BOOTSTRAP_VLLM:-1}"
BOOTSTRAP_TRAINING="${BOOTSTRAP_TRAINING:-1}"
BOOTSTRAP_PARALLELISM="${BOOTSTRAP_PARALLELISM:-4}"
VLLM_LAUNCH_PARALLELISM="${VLLM_LAUNCH_PARALLELISM:-4}"
VLLM_ENV_CACHE_GCS_URI="${VLLM_ENV_CACHE_GCS_URI:-}"
VLLM_PREFETCH_MODEL="${VLLM_PREFETCH_MODEL:-1}"
VLLM_PREFETCH_RETRIES="${VLLM_PREFETCH_RETRIES:-3}"
VLLM_PREFETCH_PARALLELISM="${VLLM_PREFETCH_PARALLELISM:-$BOOTSTRAP_PARALLELISM}"
VLLM_HF_HOME="${VLLM_HF_HOME:-$REMOTE_ROOT/.hf}"
VLLM_HF_HUB_CACHE="${VLLM_HF_HUB_CACHE:-$VLLM_HF_HOME/hub}"
VLLM_HF_DATASETS_CACHE="${VLLM_HF_DATASETS_CACHE:-$VLLM_HF_HOME/datasets}"
VLLM_TRANSFORMERS_CACHE="${VLLM_TRANSFORMERS_CACHE:-$VLLM_HF_HOME/transformers}"
VLLM_LOCAL_MODEL_ROOT="${VLLM_LOCAL_MODEL_ROOT:-$REMOTE_ROOT/.models}"
VLLM_OFFLINE_AFTER_PREFETCH="${VLLM_OFFLINE_AFTER_PREFETCH:-1}"
MIN_READY_SERVERS="${MIN_READY_SERVERS:-}"
ALLOW_PARTIAL_VLLM_SLICE="${ALLOW_PARTIAL_VLLM_SLICE:-0}"
LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
PARTIAL_READY_DELAY_ATTEMPTS="${PARTIAL_READY_DELAY_ATTEMPTS:-30}"

# === Serious run training args (matching saturation script) ===
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-8192}"
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-$MAX_COMPLETION_LEN}"
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-8192}"
TRAJ_MAX_COMPLETION_LEN="${TRAJ_MAX_COMPLETION_LEN:-$MAX_COMPLETION_LEN}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-10240}"
TEMPERATURE_SCHEDULE="${TEMPERATURE_SCHEDULE:-1.2,1.0,0.8}"
EXPLORATION_THINKING_MODE="${EXPLORATION_THINKING_MODE:-think}"
EVAL_THINKING_MODE="${EVAL_THINKING_MODE:-think}"
ACTUAL_THINKING_MODE="${ACTUAL_THINKING_MODE:-think}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-1000}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-32}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-2}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-8}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-8}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-8}"
TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

ACTUAL_TARGET_CONTRACT="${ACTUAL_TARGET_CONTRACT:-full_trace}"
ACTUAL_DATASET_POLICY="${ACTUAL_DATASET_POLICY:-accepted_only}"
ACTUAL_REWARD_SOURCE="${ACTUAL_REWARD_SOURCE:-correctness}"
ACTUAL_ONLINE_GRPO_MAX_STEPS="${ACTUAL_ONLINE_GRPO_MAX_STEPS:-}"

# Derive MAX_MODEL_LEN for vLLM server
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
if [ -z "$MAX_MODEL_LEN" ]; then
  server_completion_cap="$MAX_COMPLETION_LEN"
  for candidate in "$EXPLORATION_MAX_COMPLETION_LEN" "$EVAL_MAX_COMPLETION_LEN" "${TRAJ_MAX_COMPLETION_LEN:-0}"; do
    if [ "${candidate:-0}" -gt "$server_completion_cap" ]; then
      server_completion_cap="$candidate"
    fi
  done
  derived_model_len="$((MAX_PROMPT_LEN + server_completion_cap + 256))"
  if [ "$MAX_TOTAL_LEN" -gt "$derived_model_len" ]; then
    MAX_MODEL_LEN="$MAX_TOTAL_LEN"
  else
    MAX_MODEL_LEN="$derived_model_len"
  fi
fi

MODEL_TAG="$(printf '%s' "$MODEL_NAME" | sed 's#[^A-Za-z0-9._-]#_#g')"
VLLM_LOCAL_MODEL_DIR="${VLLM_LOCAL_MODEL_DIR:-$VLLM_LOCAL_MODEL_ROOT/$MODEL_TAG}"
if [ -d "$MODEL_NAME" ]; then
  VLLM_MODEL_FOR_SERVE="$MODEL_NAME"
else
  VLLM_MODEL_FOR_SERVE="$VLLM_LOCAL_MODEL_DIR"
fi

log() {
  printf '[single-pod] %s\n' "$*"
}

describe_worker_count() {
  local tpu_name="$1"
  local zone="$2"
  local desc
  desc="$(tmx_gcloud alpha compute tpus tpu-vm describe "$tpu_name" --zone "$zone" --format=json)"
  python3 - "$desc" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
eps = payload.get("networkEndpoints") or payload.get("workerNetworkEndpoints") or []
print(len(eps))
PY
}

describe_worker_ips() {
  local tpu_name="$1"
  local zone="$2"
  local desc
  desc="$(tmx_gcloud alpha compute tpus tpu-vm describe "$tpu_name" --zone "$zone" --format=json)"
  python3 - "$desc" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
eps = payload.get("networkEndpoints") or payload.get("workerNetworkEndpoints") or []
print(",".join(ep.get("ipAddress", "") for ep in eps if ep.get("ipAddress")))
PY
}

# === Discover pod workers ===
WORKER_COUNT="$(describe_worker_count "$POD_TPU_NAME" "$POD_ZONE")"
WORKER_IPS="$(describe_worker_ips "$POD_TPU_NAME" "$POD_ZONE")"

# vLLM runs on workers 1..N-1 (skip worker 0 — reserved for training)
VLLM_WORKER_COUNT=$((WORKER_COUNT - 1))

if [ -z "$MIN_READY_SERVERS" ]; then
  MIN_READY_SERVERS="$VLLM_WORKER_COUNT"
fi

# Extract worker 0 IP and vLLM worker IPs separately
IFS=',' read -r -a ALL_IPS <<< "$WORKER_IPS"
WORKER0_IP="${ALL_IPS[0]}"
VLLM_WORKER_IPS=""
for i in $(seq 1 $((${#ALL_IPS[@]} - 1))); do
  VLLM_WORKER_IPS="${VLLM_WORKER_IPS:+$VLLM_WORKER_IPS,}${ALL_IPS[$i]}"
done

log "pod=$POD_TPU_NAME ($POD_ZONE) via $POD_SSH_HOST"
log "total_workers=$WORKER_COUNT worker0=$WORKER0_IP (TPU training)"
log "vllm_workers=$VLLM_WORKER_COUNT ips=$VLLM_WORKER_IPS"
log "run=$RUN_NAME model=$MODEL_NAME dataset=$DATASET_NAME"

# === Step 1: Sync repo to all workers ===
if [ "$SYNC_REPO" = "1" ]; then
  log "syncing repo to all pod workers"
  TPU_NAME="$POD_TPU_NAME" ZONE="$POD_ZONE" SYNC_WORKERS="all" ALLOW_DIRTY="$ALLOW_DIRTY" \
    DIRECT_SSH_HOST="$POD_SSH_HOST" REMOTE_ROOT="$REMOTE_ROOT" bash scripts/sync_tpu_slice_repo.sh
fi

# === Step 2: Bootstrap vLLM env on workers 1-15 (not worker 0) ===
if [ "$BOOTSTRAP_VLLM" = "1" ]; then
  log "bootstrapping vLLM env on workers 1-$((WORKER_COUNT - 1))"
  active=0
  for w in $(seq 1 $((WORKER_COUNT - 1))); do
    (
      tmx_gcloud alpha compute tpus tpu-vm ssh "$POD_TPU_NAME" \
        --worker="$w" --zone="$POD_ZONE" "$TRANSPORT_FLAG" \
        --command="cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' bash scripts/bootstrap_vllm_tpu_env.sh"
    ) &
    active=$((active + 1))
    if [ "$active" -ge "$BOOTSTRAP_PARALLELISM" ]; then
      wait
      active=0
    fi
  done
  wait
fi

# === Step 3: Bootstrap training env on worker 0 ===
if [ "$BOOTSTRAP_TRAINING" = "1" ]; then
  log "bootstrapping training env on worker 0 (with libtpu-nightly for v6e)"
  ssh -o StrictHostKeyChecking=no "$POD_SSH_HOST" \
    "cd '$REMOTE_ROOT' && VENV_DIR=.venv_tpu_slice USE_TPU=1 LIBTPU_NIGHTLY_VERSION=0.1.dev20241115+nightly bash scripts/bootstrap_tpu_env.sh"
fi

# === Step 4: Prefetch model on all workers (training + vLLM all need it) ===
if [ "$VLLM_PREFETCH_MODEL" = "1" ] && [ ! -d "$MODEL_NAME" ]; then
  log "prefetching model on all workers"
  prefetch_active=0
  for w in $(seq 0 $((WORKER_COUNT - 1))); do
    (
      tmx_gcloud alpha compute tpus tpu-vm ssh "$POD_TPU_NAME" \
        --worker="$w" --zone="$POD_ZONE" "$TRANSPORT_FLAG" \
        --command="cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' MODEL_NAME='$MODEL_NAME' LOCAL_MODEL_ROOT='$VLLM_LOCAL_MODEL_ROOT' VLLM_LOCAL_MODEL_DIR='$VLLM_LOCAL_MODEL_DIR' VLLM_PREFETCH_RETRIES='$VLLM_PREFETCH_RETRIES' HF_HOME='$VLLM_HF_HOME' HF_HUB_CACHE='$VLLM_HF_HUB_CACHE' HF_DATASETS_CACHE='$VLLM_HF_DATASETS_CACHE' TRANSFORMERS_CACHE='$VLLM_TRANSFORMERS_CACHE' bash scripts/prepare_hf_model_snapshot.sh"
    ) &
    prefetch_active=$((prefetch_active + 1))
    if [ "$prefetch_active" -ge "$VLLM_PREFETCH_PARALLELISM" ]; then
      wait
      prefetch_active=0
    fi
  done
  wait
fi

# === Step 5: Launch vLLM on workers 1-15 only (worker 0 reserved for training) ===
if [ "$LAUNCH_VLLM" = "1" ]; then
  log "launching vLLM on workers 1-$((WORKER_COUNT - 1))"
  launch_active=0
  for w in $(seq 1 $((WORKER_COUNT - 1))); do
    session="vllm_server_${RUN_NAME}_w${w}"
    remote_vllm_cmd=$(
      cat <<EOF
set -euo pipefail
cd '$REMOTE_ROOT'
tmux kill-server 2>/dev/null || true
pkill -9 -f '[v]llm serve' 2>/dev/null || true
fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
sleep 2
cat > /tmp/${session}.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
export VLLM_VENV_DIR='$VLLM_VENV_DIR'
export MODEL_NAME='$VLLM_MODEL_FOR_SERVE'
export SOURCE_MODEL_NAME='$MODEL_NAME'
export SERVED_MODEL_NAME='$SERVED_MODEL_NAME'
export PORT='$PORT'
export TP_SIZE='$TP_SIZE'
export MAX_MODEL_LEN='$MAX_MODEL_LEN'
export MAX_NUM_SEQS='$MAX_NUM_SEQS'
export MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS'
export HF_HOME='$VLLM_HF_HOME'
export HF_HUB_CACHE='$VLLM_HF_HUB_CACHE'
export HF_DATASETS_CACHE='$VLLM_HF_DATASETS_CACHE'
export TRANSFORMERS_CACHE='$VLLM_TRANSFORMERS_CACHE'
export LOCAL_MODEL_ROOT='$VLLM_LOCAL_MODEL_ROOT'
export VLLM_LOCAL_MODEL_DIR='$VLLM_LOCAL_MODEL_DIR'
export VLLM_PREFETCH_MODEL=0
export HF_HUB_OFFLINE_AFTER_PREFETCH='$VLLM_OFFLINE_AFTER_PREFETCH'
export DOWNLOAD_DIR='$VLLM_HF_HUB_CACHE'
bash scripts/run_vllm_server.sh > /tmp/vllm_server_${RUN_NAME}.log 2>&1
INNER
chmod +x /tmp/${session}.sh
tmux new-session -d -s '$session' 'bash /tmp/${session}.sh'
EOF
    )
    (
      tmx_gcloud alpha compute tpus tpu-vm ssh "$POD_TPU_NAME" \
        --worker="$w" --zone="$POD_ZONE" "$TRANSPORT_FLAG" \
        --command="$remote_vllm_cmd"
    ) &
    launch_active=$((launch_active + 1))
    if [ "$launch_active" -ge "$VLLM_LAUNCH_PARALLELISM" ]; then
      wait
      launch_active=0
    fi
  done
  wait
fi

# === Step 6: Health check from worker 0 (only checking workers 1-15) ===
log "waiting for vLLM health on $VLLM_WORKER_COUNT servers"
health_check_script="
IPS='$VLLM_WORKER_IPS'
PORT='$PORT'
TOTAL='$VLLM_WORKER_COUNT'
MIN_READY='$MIN_READY_SERVERS'
ALLOW_PARTIAL='$ALLOW_PARTIAL_VLLM_SLICE'
PARTIAL_READY_DELAY='$PARTIAL_READY_DELAY_ATTEMPTS'
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
  if [ \"\$ALLOW_PARTIAL\" = \"1\" ] && [ \"\$ready\" -ge \"\$MIN_READY\" ] && [ \"\$attempt\" -ge \"\$PARTIAL_READY_DELAY\" ]; then
    echo \"URLS=\$urls\"
    echo \"[health] proceeding with \$ready/\$TOTAL after \$attempt attempts\"
    exit 0
  fi
  sleep 10
done
echo '[health] ERROR: vLLM servers not healthy'
exit 1
"

health_output="$(tmx_gcloud alpha compute tpus tpu-vm ssh "$POD_TPU_NAME" \
  --worker=0 --zone="$POD_ZONE" "$TRANSPORT_FLAG" \
  --command="$health_check_script")"

echo "$health_output" | grep -v '^URLS=' | while read -r line; do
  log "$line"
done
VLLM_SERVER_URLS="$(echo "$health_output" | grep '^URLS=' | head -1 | sed 's/^URLS=//')"
if [ -z "$VLLM_SERVER_URLS" ]; then
  log "ERROR: failed to discover healthy vLLM servers"
  exit 1
fi

healthy_server_count="$(echo "$VLLM_SERVER_URLS" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' ')"
TMX_VLLM_CONCURRENCY_VALUE="${TMX_VLLM_CONCURRENCY:-}"
if [ -z "$TMX_VLLM_CONCURRENCY_VALUE" ]; then
  target_concurrency=$((GRPO_BATCH_SIZE * GRPO_NUM_GENERATIONS))
  if [ "$GRPO_GENERATION_BATCH_SIZE" -gt "$target_concurrency" ]; then
    target_concurrency="$GRPO_GENERATION_BATCH_SIZE"
  fi
  if [ "$TRAJECTORY_BATCH_SIZE" -gt "$target_concurrency" ]; then
    target_concurrency="$TRAJECTORY_BATCH_SIZE"
  fi
  if [ "$EVAL_BATCH_SIZE" -gt "$target_concurrency" ]; then
    target_concurrency="$EVAL_BATCH_SIZE"
  fi
  concurrency_cap=$((healthy_server_count * 4))
  if [ "$target_concurrency" -le "$concurrency_cap" ]; then
    TMX_VLLM_CONCURRENCY_VALUE="$target_concurrency"
  else
    TMX_VLLM_CONCURRENCY_VALUE="$concurrency_cap"
  fi
  if [ "$TMX_VLLM_CONCURRENCY_VALUE" -lt 64 ]; then
    TMX_VLLM_CONCURRENCY_VALUE=64
  fi
fi
# Reload worker is worker 1 (first vLLM server), not worker 0 (training)
reload_worker_url="$(echo "$VLLM_WORKER_IPS" | tr ',' '\n' | head -1 | sed "s#^#http://#; s#\$#:${PORT}/v1#")"

log "healthy=$healthy_server_count/$VLLM_WORKER_COUNT concurrency=$TMX_VLLM_CONCURRENCY_VALUE"
log "vLLM URLs: $VLLM_SERVER_URLS"

# === Step 7: Build training args ===
RUN_DIR="runs/$RUN_NAME"

TRAIN_ARGS=(
  main.py
  --device tpu
  --use-wandb
  --seed 0
  --model-name "$MODEL_NAME"
  --dataset-name "$DATASET_NAME"
  --train-selection-policy random
  --train-probe-pool-size 256
  --train-probe-samples-per-prompt 4
  --train-probe-temperature 0.8
  --train-probe-batch-size 4
  --dataset-split-policy auto
  --topic-split-mode heuristic_math_topics
  --topic-partition-policy explicit_or_heuristic
  --decoupling-mode decoupled
  --base-eval-mode raw
  --actual-target-contract "$ACTUAL_TARGET_CONTRACT"
  --max-train-examples "$MAX_TRAIN_EXAMPLES"
  --max-eval-examples "$MAX_EVAL_EXAMPLES"
  --grpo-max-steps "$GRPO_MAX_STEPS"
  --grpo-batch-size "$GRPO_BATCH_SIZE"
  --grpo-grad-accum "$GRPO_GRAD_ACCUM"
  --grpo-loss-type dr_grpo
  --grpo-num-generations "$GRPO_NUM_GENERATIONS"
  --grpo-generation-batch-size "$GRPO_GENERATION_BATCH_SIZE"
  --num-samples-per-prompt "$NUM_SAMPLES_PER_PROMPT"
  --temperature-schedule "$TEMPERATURE_SCHEDULE"
  --answer-constraint-mode none
  --exploration-response-style reasoning_final
  --eval-response-style reasoning_final
  --exploration-answer-constraint-mode none
  --eval-answer-constraint-mode none
  --exploration-thinking-mode "$EXPLORATION_THINKING_MODE"
  --eval-thinking-mode "$EVAL_THINKING_MODE"
  --actual-thinking-mode "$ACTUAL_THINKING_MODE"
  --distill-epochs 2
  --distill-batch-size 1
  --max-distill-examples 4096
  --actual-grpo-epochs 1
  --actual-grpo-batch-size 1
  --actual-grpo-grad-accum 2
  --actual-min-accepted-trajectories 4
  --actual-dataset-policy "$ACTUAL_DATASET_POLICY"
  --actual-max-incorrect-ratio 0.25
  --actual-target-correct-fraction 0.75
  --actual-grpo-kl-coef 0.10
  --actual-grpo-lr 1e-6
  --distill-lr 5e-6
  --max-actual-examples 4096
  --novelty-feature-source multilayer
  --novelty-feature-max-length 512
  --novelty-metric sqrt_mse
  --novelty-text-contract full_raw
  --rnd-hidden 512
  --reward-correct 1.0
  --reward-incorrect 0.0
  --lambda-novelty 0.10
  --reward-shape linear
  --anchor-metric none
  --trajectory-batch-size "$TRAJECTORY_BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --round-budget-mode constant_total
  --max-prompt-len "$MAX_PROMPT_LEN"
  --max-completion-len "$MAX_COMPLETION_LEN"
  --exploration-max-completion-len "$EXPLORATION_MAX_COMPLETION_LEN"
  --eval-max-completion-len "$EVAL_MAX_COMPLETION_LEN"
  --max-total-len "$MAX_TOTAL_LEN"
  --format-penalty-weight 0.1
  --pipeline-mode two_model
  --actual-init-source base
  --actual-train-mode sft_online_grpo
  --actual-reward-source "$ACTUAL_REWARD_SOURCE"
  --wandb-run-name "$WANDB_RUN_NAME"
  --plots-dir "$RUN_DIR/plots"
  --explorer-traj-path "$RUN_DIR/explorer_trajectories.jsonl"
  --explorer-output-dir "$RUN_DIR/explorer_ckpts"
  --saved-model-root "$REMOTE_ROOT/$RUN_DIR/saved_models"
  --incorrect-novelty-scale 0.0
)

if [ -n "$ACTUAL_ONLINE_GRPO_MAX_STEPS" ]; then
  TRAIN_ARGS+=(--actual-online-grpo-max-steps "$ACTUAL_ONLINE_GRPO_MAX_STEPS")
fi

if [ -n "${BENCHMARK_DATASET_NAME:-}" ]; then
  TRAIN_ARGS+=(--benchmark-dataset-name "$BENCHMARK_DATASET_NAME")
  TRAIN_ARGS+=(--benchmark-eval-mode "${BENCHMARK_EVAL_MODE:-sampled}")
  TRAIN_ARGS+=(--benchmark-num-rollouts "${BENCHMARK_NUM_ROLLOUTS:-32}")
  TRAIN_ARGS+=(--max-benchmark-examples "${MAX_BENCHMARK_EXAMPLES:-30}")
fi

printf -v quoted_train_args '%q ' "${TRAIN_ARGS[@]}"

# === Step 8: Launch TPU training on worker 0 ===
remote_cmd=$(
  cat <<EOF
set -euo pipefail
cd '$REMOTE_ROOT'
tmux kill-session -t 'training_$RUN_NAME' 2>/dev/null || true
sleep 1
mkdir -p '$REMOTE_ROOT/$RUN_DIR'
cat > /tmp/training_${RUN_NAME}.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
source .venv_tpu_slice/bin/activate
export WANDB_PROJECT='$WANDB_PROJECT'
export WANDB_ENTITY='$WANDB_ENTITY'
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# === TPU single-chip training (worker 0 has 4 chips, use chip 0 only) ===
export TPU_VISIBLE_CHIPS=0
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export PJRT_DEVICE=TPU

# === Flash attention for 8k training on 32GB chip ===
export TMX_XLA_FLASH_ATTENTION=1
export TMX_XLA_FORCE_EAGER_ATTN=0
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_OPTIM=adamw_torch

# === Decoupled training controls ===
# No gradient checkpointing — 32GB HBM fits 8k easily without it (~30% faster)
export TMX_DECOUPLED_TRAIN_TOKEN_CHUNK=0
export TMX_DECOUPLED_GRAD_CKPT=0
export TMX_DECOUPLED_TRAIN_DEBUG_PROGRESS=1
export TMX_GRPO_GRAD_CKPT=0
export TMX_EXPLORER_DECOUPLED_GRPO=1
export TMX_GRPO_LOGPROB_CHUNK_SIZE=2

# === vLLM for generation ===
export TMX_GRPO_USE_VLLM=1
export TMX_TRAJ_USE_VLLM=1
export TMX_VLLM_SERVER_URLS='$VLLM_SERVER_URLS'
export TMX_VLLM_MODEL_NAME='$SERVED_MODEL_NAME'
export TMX_VLLM_CONCURRENCY='$TMX_VLLM_CONCURRENCY_VALUE'
export TMX_VLLM_MAX_RETRIES=5
export TMX_TRAJ_MAX_COMPLETION_LEN='$TRAJ_MAX_COMPLETION_LEN'

# === Novelty on TPU (fast) ===
export TMX_NOVELTY_ON_TPU=1
export TMX_NOVELTY_BATCH_SIZE=16

# === Explorer decoupled tuning ===
export TMX_EXPLORER_TRAIN_TOP_P=0.95
export TMX_EXPLORER_TRAIN_TOP_K=20
export TMX_EXPLORER_DECOUPLED_BUFFER_GROUPS=32
export TMX_EXPLORER_DECOUPLED_BATCH_GROUPS=1
export TMX_EXPLORER_DECOUPLED_GRAD_ACCUM=1
export TMX_EXPLORER_DECOUPLED_ROW_MICROBATCH=1
export TMX_EXPLORER_DECOUPLED_TRAIN_FSDP=0
export TMX_EXPLORER_DECOUPLED_TRAIN_FSDP_GRAD_CKPT=0

# === XLA graph caching for fixed shapes ===
export TMX_DECOUPLED_ROW_MARK_STEP=1
export TMX_DECOUPLED_ACCUM_MARK_STEP=1
export TMX_DECOUPLED_ACCUM_MARK_RESET_SCOPE=0
export TMX_DECOUPLED_PRE_OPTIMIZER_MARK_STEP=1
export TMX_DECOUPLED_TRAIN_FIXED_SHAPES=1
export TMX_DECOUPLED_TRAIN_FIXED_PROMPT_WIDTH=0
export TMX_DECOUPLED_TRAIN_FIXED_COMPLETION_WIDTH=0

# === Misc ===
export TMX_DEBUG_GENERATE=0
export TMX_WANDB_MINIMAL=1
export TMX_DISABLE_BF16=0
export TMX_STRICT_GRPO_GEOMETRY=0
export TMX_RESUME_STEP='${TMX_RESUME_STEP:-0}'
export TMX_RESUME_CHECKPOINT='${TMX_RESUME_CHECKPOINT:-}'
export TMX_RESUME_RUN_NAME='${TMX_RESUME_RUN_NAME:-}'
export TMX_RESUME_WANDB_ID='${TMX_RESUME_WANDB_ID:-}'
export TMX_RESUME_TRAINING=0
export TMX_TRAJ_TARGET_ACCEPTED=2048
export TMX_GRPO_LOGGING_STRATEGY=steps
export TMX_GRPO_LOGGING_STEPS=1
export TMX_GRPO_SAVE_STRATEGY=steps
export TMX_GRPO_SAVE_STEPS=50
export TMX_GRPO_SAVE_TOTAL_LIMIT=3
export TMX_ACTUAL_GRPO_SAVE_STEPS=25
export TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT=4
export TMX_SKIP_INTERMEDIATE_EVALS=1
export TMX_SKIP_FINAL_BENCHMARK_EVALS=1
export TMX_STOP_AFTER_EXPLORER=0
export TMX_STOP_AFTER_TRAJECTORIES=0
export TMX_ABORT_ON_DEGENERATE_EXPLORER=0
export TMX_ABORT_ON_BAD_TRAJ_WINDOW=0

# === vLLM reload (point at inference workers, not worker 0) ===
export TMX_GCS_CHECKPOINT_BASE='$RUN_ARTIFACT_GCS_PREFIX'
export TMX_GCS_CHECKPOINT_ROOT='$RUN_ARTIFACT_GCS_PREFIX'
export TMX_EXPLORER_VLLM_RELOAD_TPU_NAME='$POD_TPU_NAME'
export TMX_EXPLORER_VLLM_RELOAD_ZONE='$POD_ZONE'
export TMX_EXPLORER_VLLM_RELOAD_WORKERS='$(seq -s" " 1 $((WORKER_COUNT - 1)))'
export TMX_EXPLORER_VLLM_MODEL_NAME=explorer_model
export TMX_VLLM_RELOAD_TPU_NAME='$POD_TPU_NAME'
export TMX_VLLM_RELOAD_ZONE='$POD_ZONE'
export TMX_VLLM_RELOAD_WORKER='$RELOAD_WORKER'
export TMX_VLLM_RELOAD_WORKER_URL='$reload_worker_url'
export TMX_ACTUAL_VLLM_MODEL_NAME=actual_model
export PORT='$PORT'
export TP_SIZE='$TP_SIZE'
export MAX_MODEL_LEN='$MAX_MODEL_LEN'
export MAX_NUM_SEQS='$MAX_NUM_SEQS'
export MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS'
export VLLM_VENV_DIR='$VLLM_VENV_DIR'

# === Threading (balance with XLA background threads) ===
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONPATH='$REMOTE_ROOT'

mkdir -p '$REMOTE_ROOT/$RUN_DIR'
python -u ${quoted_train_args} 2>&1 | tee -a '$REMOTE_ROOT/$RUN_DIR/console.log'
INNER
chmod +x /tmp/training_${RUN_NAME}.sh
tmux new-session -d -s 'training_$RUN_NAME' 'bash /tmp/training_${RUN_NAME}.sh'
tmux ls | grep 'training_$RUN_NAME'
EOF
)
log "launching TPU training on worker 0 (flash attention, single chip)"
ssh -o StrictHostKeyChecking=no "$POD_SSH_HOST" "$remote_cmd"

log "launched: run=$RUN_NAME"
log "monitor: ssh $POD_SSH_HOST 'tmux capture-pane -t training_$RUN_NAME -p -S -80'"
log "vllm logs: gcloud alpha compute tpus tpu-vm ssh $POD_TPU_NAME --worker=1 --zone=$POD_ZONE $TRANSPORT_FLAG --command='tail -50 /tmp/vllm_server_${RUN_NAME}.log'"
