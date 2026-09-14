#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh

TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
TRAIN_ZONE="${TRAIN_ZONE:-us-east1-d}"
TRAIN_SSH_HOST="${TRAIN_SSH_HOST:?set TRAIN_SSH_HOST to the ssh host alias of the training TPU VM}"
TRAIN_REMOTE_ROOT="${TRAIN_REMOTE_ROOT:?set TRAIN_REMOTE_ROOT to the absolute repo checkout path on the training TPU VM}"
TRAIN_WORKER="${TRAIN_WORKER:-0}"

SINGLE_SLICE_MODE="${SINGLE_SLICE_MODE:-0}"
if [ "$SINGLE_SLICE_MODE" = "1" ]; then
  INFER_TPU_NAME="${INFER_TPU_NAME:-$TRAIN_TPU_NAME}"
  INFER_ZONE="${INFER_ZONE:-$TRAIN_ZONE}"
  INFER_SSH_HOST="${INFER_SSH_HOST:-$TRAIN_SSH_HOST}"
  INFER_REMOTE_ROOT="${INFER_REMOTE_ROOT:-$TRAIN_REMOTE_ROOT}"
  VLLM_FIRST_WORKER="${VLLM_FIRST_WORKER:-1}"
else
  INFER_TPU_NAME="${INFER_TPU_NAME:?set INFER_TPU_NAME to the inference (vLLM) TPU VM (slice) name}"
  INFER_ZONE="${INFER_ZONE:-europe-west4-b}"
  INFER_SSH_HOST="${INFER_SSH_HOST:?set INFER_SSH_HOST to the ssh host alias of the inference (vLLM) TPU VM}"
  INFER_REMOTE_ROOT="${INFER_REMOTE_ROOT:?set INFER_REMOTE_ROOT to the absolute repo checkout path on the inference TPU VM}"
  VLLM_FIRST_WORKER="${VLLM_FIRST_WORKER:-0}"
fi
VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}"

INFER_TRANSPORT_FLAG="$(expdis_tpu_transport_flag)"
ALLOW_DIRTY="${ALLOW_DIRTY:-1}"
RUN_NAME="${RUN_NAME:-dualslice_qwen3_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_NAME}"
RUN_ARTIFACT_GCS_PREFIX="${RUN_ARTIFACT_GCS_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/checkpoints}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
DATASET_NAME="${DATASET_NAME:-dapo_math_17k}"

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
RELOAD_WORKER="${RELOAD_WORKER:-0}"

SYNC_INFER_REPO="${SYNC_INFER_REPO:-1}"
SYNC_TRAIN_REPO="${SYNC_TRAIN_REPO:-1}"
BOOTSTRAP_VLLM="${BOOTSTRAP_VLLM:-1}"
BOOTSTRAP_TRAINING="${BOOTSTRAP_TRAINING:-1}"
BOOTSTRAP_PARALLELISM="${BOOTSTRAP_PARALLELISM:-4}"
VLLM_LAUNCH_PARALLELISM="${VLLM_LAUNCH_PARALLELISM:-4}"
VLLM_ENV_CACHE_GCS_URI="${VLLM_ENV_CACHE_GCS_URI:-}"
VLLM_PREFETCH_MODEL="${VLLM_PREFETCH_MODEL:-1}"
VLLM_PREFETCH_RETRIES="${VLLM_PREFETCH_RETRIES:-3}"
VLLM_PREFETCH_PARALLELISM="${VLLM_PREFETCH_PARALLELISM:-$BOOTSTRAP_PARALLELISM}"
VLLM_HF_HOME="${VLLM_HF_HOME:-$INFER_REMOTE_ROOT/.hf}"
VLLM_HF_HUB_CACHE="${VLLM_HF_HUB_CACHE:-$VLLM_HF_HOME/hub}"
VLLM_HF_DATASETS_CACHE="${VLLM_HF_DATASETS_CACHE:-$VLLM_HF_HOME/datasets}"
VLLM_TRANSFORMERS_CACHE="${VLLM_TRANSFORMERS_CACHE:-$VLLM_HF_HOME/transformers}"
VLLM_LOCAL_MODEL_ROOT="${VLLM_LOCAL_MODEL_ROOT:-$INFER_REMOTE_ROOT/.models}"
VLLM_OFFLINE_AFTER_PREFETCH="${VLLM_OFFLINE_AFTER_PREFETCH:-1}"
MIN_READY_SERVERS="${MIN_READY_SERVERS:-}"
ALLOW_PARTIAL_VLLM_SLICE="${ALLOW_PARTIAL_VLLM_SLICE:-0}"
LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
PARTIAL_READY_DELAY_ATTEMPTS="${PARTIAL_READY_DELAY_ATTEMPTS:-30}"
SKIP_INFERENCE_SLICE="${SKIP_INFERENCE_SLICE:-0}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-1024}"
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-$MAX_COMPLETION_LEN}"
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-$MAX_COMPLETION_LEN}"
TRAJ_MAX_COMPLETION_LEN="${TRAJ_MAX_COMPLETION_LEN:-}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-$((MAX_PROMPT_LEN + MAX_COMPLETION_LEN))}"
TEMPERATURE_SCHEDULE="${TEMPERATURE_SCHEDULE:-1.0}"
EXPLORATION_THINKING_MODE="${EXPLORATION_THINKING_MODE:-auto}"
EVAL_THINKING_MODE="${EVAL_THINKING_MODE:-auto}"
ACTUAL_THINKING_MODE="${ACTUAL_THINKING_MODE:-auto}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-32}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-128}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-32}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-16}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-4}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}"

if [ -z "${TRAJECTORY_BATCH_SIZE:-}" ]; then
  if [ "${EXPDIS_TRAJ_USE_VLLM:-0}" = "1" ] || [ -n "${EXPDIS_EXPLORER_PRETRAINED_PATH:-}" ]; then
    TRAJECTORY_BATCH_SIZE=16
  else
    TRAJECTORY_BATCH_SIZE=1
  fi
fi

if [ -z "${EVAL_BATCH_SIZE:-}" ]; then
  if [ "${EXPDIS_TRAJ_USE_VLLM:-0}" = "1" ] || [ -n "${EXPDIS_EXPLORER_PRETRAINED_PATH:-}" ]; then
    EVAL_BATCH_SIZE=16
  else
    EVAL_BATCH_SIZE=1
  fi
fi

ACTUAL_TARGET_CONTRACT="${ACTUAL_TARGET_CONTRACT:-full_trace}"
ACTUAL_DATASET_POLICY="${ACTUAL_DATASET_POLICY:-accepted_only}"
ACTUAL_REWARD_SOURCE="${ACTUAL_REWARD_SOURCE:-correctness}"
ACTUAL_ONLINE_GRPO_MAX_STEPS="${ACTUAL_ONLINE_GRPO_MAX_STEPS:-}"

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

expect_var() {
  local key="$1"
  local expected_key="EXPECTED_${key}"
  local expected_value="${!expected_key:-}"
  if [ -z "$expected_value" ]; then
    return 0
  fi
  local actual_value="${!key:-}"
  if [ "$actual_value" != "$expected_value" ]; then
    printf '[dual-slice] ERROR: %s=%s but expected %s\n' "$key" "$actual_value" "$expected_value" >&2
    exit 1
  fi
}

if [ "${STRICT_LAUNCH_CONTRACT:-0}" = "1" ]; then
  expected_keys=(
    TRAIN_TPU_NAME
    TRAIN_ZONE
    TRAIN_SSH_HOST
    TRAIN_WORKER
    TRAIN_LAUNCH_MODE
    DDP_VISIBLE_CHIPS
    DDP_NUM_PROCESSES
    SINGLE_SLICE_MODE
    MAX_PROMPT_LEN
    MAX_COMPLETION_LEN
    EXPLORATION_MAX_COMPLETION_LEN
    EVAL_MAX_COMPLETION_LEN
    MAX_TOTAL_LEN
    MAX_TRAIN_EXAMPLES
    MAX_EVAL_EXAMPLES
    GRPO_MAX_STEPS
    GRPO_BATCH_SIZE
    GRPO_GRAD_ACCUM
    GRPO_NUM_GENERATIONS
    GRPO_GENERATION_BATCH_SIZE
    NUM_SAMPLES_PER_PROMPT
    TRAJECTORY_BATCH_SIZE
    EVAL_BATCH_SIZE
    EXPDIS_FSDP
    EXPDIS_USE_XLA_WORLD_SIZE
    EXPDIS_TRAJ_USE_VLLM
    EXPDIS_XLA_FLASH_ATTENTION
    EXPDIS_XLA_FORCE_EAGER_ATTN
    EXPDIS_ENABLE_THINKING
    EXPDIS_STRICT_GRPO_GEOMETRY
  )
  for key in "${expected_keys[@]}"; do
    expect_var "$key"
  done
fi

MODEL_TAG="$(printf '%s' "$MODEL_NAME" | sed 's#[^A-Za-z0-9._-]#_#g')"
VLLM_LOCAL_MODEL_DIR="${VLLM_LOCAL_MODEL_DIR:-$VLLM_LOCAL_MODEL_ROOT/$MODEL_TAG}"
if [ -d "$MODEL_NAME" ]; then
  VLLM_MODEL_FOR_SERVE="$MODEL_NAME"
else
  VLLM_MODEL_FOR_SERVE="$VLLM_LOCAL_MODEL_DIR"
fi

log() {
  printf '[dual-slice] %s\n' "$*"
}

# Route an SSH command to the training worker. For worker 0 use direct ssh
# (via the $TRAIN_SSH_HOST alias). For higher-indexed workers, fall back to
# gcloud SSH with --worker=N since direct aliases only point at worker 0.
train_ssh() {
  local cmd="$1"
  if [ "$TRAIN_WORKER" = "0" ]; then
    ssh -o StrictHostKeyChecking=no "$TRAIN_SSH_HOST" "$cmd"
  else
    local train_transport
    train_transport="$(expdis_tpu_transport_flag)"
    expdis_gcloud alpha compute tpus tpu-vm ssh "$TRAIN_TPU_NAME" \
      --worker="$TRAIN_WORKER" --zone="$TRAIN_ZONE" "$train_transport" \
      --command="$cmd"
  fi
}

describe_worker_count() {
  local tpu_name="$1"
  local zone="$2"
  local desc
  desc="$(expdis_gcloud alpha compute tpus tpu-vm describe "$tpu_name" --zone "$zone" --format=json)"
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
  desc="$(expdis_gcloud alpha compute tpus tpu-vm describe "$tpu_name" --zone "$zone" --format=json)"
  python3 - "$desc" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
eps = payload.get("networkEndpoints") or payload.get("workerNetworkEndpoints") or []
print(",".join(ep.get("ipAddress", "") for ep in eps if ep.get("ipAddress")))
PY
}

if [ "$SKIP_INFERENCE_SLICE" = "1" ]; then
  INFER_WORKER_COUNT=0
  INFER_WORKER_IPS=""
else
  INFER_WORKER_COUNT="$(describe_worker_count "$INFER_TPU_NAME" "$INFER_ZONE")"
  INFER_WORKER_IPS="$(describe_worker_ips "$INFER_TPU_NAME" "$INFER_ZONE")"
fi
TRAIN_ACCELERATOR_TYPE="$(expdis_gcloud alpha compute tpus tpu-vm describe "$TRAIN_TPU_NAME" --zone "$TRAIN_ZONE" --format='value(acceleratorType)' 2>/dev/null || true)"
TRAIN_ACCELERATOR_TYPE="${TRAIN_ACCELERATOR_TYPE:-v6e-64}"

# In single-slice mode, skip worker 0 for vLLM (reserved for training).
# VLLM_SERVER_COUNT = number of workers actually running vLLM.
VLLM_SERVER_COUNT=$((INFER_WORKER_COUNT - VLLM_FIRST_WORKER))
if [ "$VLLM_SERVER_COUNT" -lt 0 ]; then VLLM_SERVER_COUNT=0; fi

# Filter IPs: only include workers >= VLLM_FIRST_WORKER for health checks.
if [ "$VLLM_FIRST_WORKER" -gt 0 ] && [ -n "$INFER_WORKER_IPS" ]; then
  INFER_VLLM_IPS="$(echo "$INFER_WORKER_IPS" | tr ',' '\n' | tail -n +"$((VLLM_FIRST_WORKER + 1))" | tr '\n' ',' | sed 's/,$//')"
else
  INFER_VLLM_IPS="$INFER_WORKER_IPS"
fi

if [ -z "$MIN_READY_SERVERS" ]; then
  MIN_READY_SERVERS="$VLLM_SERVER_COUNT"
fi
PRESET_VLLM_SERVER_URLS="${VLLM_SERVER_URLS:-}"

log "train_tpu=$TRAIN_TPU_NAME ($TRAIN_ZONE) via $TRAIN_SSH_HOST"
if [ "$SINGLE_SLICE_MODE" = "1" ]; then
  log "single-slice mode: vLLM on workers $VLLM_FIRST_WORKER-$((INFER_WORKER_COUNT-1)), training on worker 0"
fi
if [ "$SKIP_INFERENCE_SLICE" = "1" ]; then
  log "infer_tpu=disabled for this run"
else
  log "infer_tpu=$INFER_TPU_NAME ($INFER_ZONE) vllm_workers=$VLLM_SERVER_COUNT"
fi
if [ -n "$PRESET_VLLM_SERVER_URLS" ]; then
  log "reusing preset vLLM urls; skipping inference-slice sync/bootstrap/launch"
fi

if [ "$SKIP_INFERENCE_SLICE" != "1" ] && [ -z "$PRESET_VLLM_SERVER_URLS" ] && [ "$SYNC_INFER_REPO" = "1" ]; then
  if [ "$SINGLE_SLICE_MODE" = "1" ]; then
    log "syncing repo to all workers (single-slice mode)"
    TPU_NAME="$INFER_TPU_NAME" ZONE="$INFER_ZONE" SYNC_WORKERS="all" ALLOW_DIRTY="$ALLOW_DIRTY" \
      DIRECT_SSH_HOST="$INFER_SSH_HOST" REMOTE_ROOT="$INFER_REMOTE_ROOT" bash scripts/sync_tpu_slice_repo.sh
    SYNC_TRAIN_REPO=0
  else
    log "syncing repo to inference slice"
    TPU_NAME="$INFER_TPU_NAME" ZONE="$INFER_ZONE" SYNC_WORKERS="all" ALLOW_DIRTY="$ALLOW_DIRTY" \
      DIRECT_SSH_HOST="$INFER_SSH_HOST" REMOTE_ROOT="$INFER_REMOTE_ROOT" bash scripts/sync_tpu_slice_repo.sh
  fi
fi

if [ "$SYNC_TRAIN_REPO" = "1" ]; then
  log "syncing repo to training worker $TRAIN_WORKER"
  if [ "$TRAIN_WORKER" = "0" ]; then
    TPU_NAME="$TRAIN_TPU_NAME" ZONE="$TRAIN_ZONE" SYNC_WORKERS="0" ALLOW_DIRTY="$ALLOW_DIRTY" \
      DIRECT_SSH_HOST="$TRAIN_SSH_HOST" REMOTE_ROOT="$TRAIN_REMOTE_ROOT" bash scripts/sync_tpu_slice_repo.sh
  else
    TPU_NAME="$TRAIN_TPU_NAME" ZONE="$TRAIN_ZONE" SYNC_WORKERS="$TRAIN_WORKER" ALLOW_DIRTY="$ALLOW_DIRTY" \
      REMOTE_ROOT="$TRAIN_REMOTE_ROOT" bash scripts/sync_tpu_slice_repo.sh
  fi
fi

if [ "$SKIP_INFERENCE_SLICE" != "1" ] && [ -z "$PRESET_VLLM_SERVER_URLS" ] && [ "$BOOTSTRAP_VLLM" = "1" ]; then
  log "bootstrapping vLLM env on workers $VLLM_FIRST_WORKER-$((INFER_WORKER_COUNT-1))"
  # Bootstrap first vLLM worker synchronously, then the rest in parallel.
  expdis_gcloud alpha compute tpus tpu-vm ssh "$INFER_TPU_NAME" \
    --worker="$VLLM_FIRST_WORKER" --zone="$INFER_ZONE" "$INFER_TRANSPORT_FLAG" \
    --command="cd '$INFER_REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' bash scripts/bootstrap_vllm_tpu_env.sh"

  active=0
  for w in $(seq $((VLLM_FIRST_WORKER + 1)) $((INFER_WORKER_COUNT - 1))); do
    (
      expdis_gcloud alpha compute tpus tpu-vm ssh "$INFER_TPU_NAME" \
        --worker="$w" --zone="$INFER_ZONE" "$INFER_TRANSPORT_FLAG" \
        --command="cd '$INFER_REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' bash scripts/bootstrap_vllm_tpu_env.sh"
    ) &
    active=$((active + 1))
    if [ "$active" -ge "$BOOTSTRAP_PARALLELISM" ]; then
      wait
      active=0
    fi
  done
  wait
fi

if [ "$SKIP_INFERENCE_SLICE" != "1" ] && [ -z "$PRESET_VLLM_SERVER_URLS" ] && [ "$VLLM_PREFETCH_MODEL" = "1" ] && [ ! -d "$MODEL_NAME" ]; then
  log "prefetching model snapshot on vLLM workers"
  prefetch_active=0
  for w in $(seq "$VLLM_FIRST_WORKER" $((INFER_WORKER_COUNT - 1))); do
    (
      expdis_gcloud alpha compute tpus tpu-vm ssh "$INFER_TPU_NAME" \
        --worker="$w" --zone="$INFER_ZONE" "$INFER_TRANSPORT_FLAG" \
        --command="cd '$INFER_REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' MODEL_NAME='$MODEL_NAME' LOCAL_MODEL_ROOT='$VLLM_LOCAL_MODEL_ROOT' VLLM_LOCAL_MODEL_DIR='$VLLM_LOCAL_MODEL_DIR' VLLM_PREFETCH_RETRIES='$VLLM_PREFETCH_RETRIES' HF_HOME='$VLLM_HF_HOME' HF_HUB_CACHE='$VLLM_HF_HUB_CACHE' HF_DATASETS_CACHE='$VLLM_HF_DATASETS_CACHE' TRANSFORMERS_CACHE='$VLLM_TRANSFORMERS_CACHE' bash scripts/prepare_hf_model_snapshot.sh"
    ) &
    prefetch_active=$((prefetch_active + 1))
    if [ "$prefetch_active" -ge "$VLLM_PREFETCH_PARALLELISM" ]; then
      wait
      prefetch_active=0
    fi
  done
  wait
fi

if [ "$BOOTSTRAP_TRAINING" = "1" ]; then
  log "bootstrapping training env on training worker $TRAIN_WORKER"
  train_ssh "cd '$TRAIN_REMOTE_ROOT' && VENV_DIR=.venv_tpu_slice USE_TPU=1 bash scripts/bootstrap_tpu_env.sh"
fi

if [ "$SKIP_INFERENCE_SLICE" != "1" ] && [ -z "$PRESET_VLLM_SERVER_URLS" ] && [ "$LAUNCH_VLLM" = "1" ]; then
  log "launching vLLM on workers $VLLM_FIRST_WORKER-$((INFER_WORKER_COUNT-1))"
  launch_active=0
  for w in $(seq "$VLLM_FIRST_WORKER" $((INFER_WORKER_COUNT - 1))); do
    session="vllm_server_${RUN_NAME}_w${w}"
    remote_vllm_cmd=$(
      cat <<EOF
set -euo pipefail
cd '$INFER_REMOTE_ROOT'
tmux kill-server 2>/dev/null || true
pkill -9 -f '[v]llm serve' 2>/dev/null || true
fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
sleep 2
cat > /tmp/${session}.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd '$INFER_REMOTE_ROOT'
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
export EXPDIS_VLLM_HOT_RELOAD='${EXPDIS_VLLM_HOT_RELOAD:-0}'
bash scripts/run_vllm_server.sh > /tmp/vllm_server_${RUN_NAME}.log 2>&1
INNER
chmod +x /tmp/${session}.sh
tmux new-session -d -s '$session' 'bash /tmp/${session}.sh'
EOF
    )
    (
      expdis_gcloud alpha compute tpus tpu-vm ssh "$INFER_TPU_NAME" \
        --worker="$w" --zone="$INFER_ZONE" "$INFER_TRANSPORT_FLAG" \
        --command="$remote_vllm_cmd"
    ) &
    launch_active=$((launch_active + 1))
    if [ "$launch_active" -ge "$VLLM_LAUNCH_PARALLELISM" ]; then
      wait
      launch_active=0
    fi
  done
  wait
else
  if [ "$SKIP_INFERENCE_SLICE" = "1" ]; then
    log "skipping inference slice bootstrap/launch by request"
  else
    log "skipping vLLM launch; reusing existing inference slice"
  fi
fi

if [ "$SKIP_INFERENCE_SLICE" != "1" ] && [ -n "$PRESET_VLLM_SERVER_URLS" ]; then
  VLLM_SERVER_URLS="$PRESET_VLLM_SERVER_URLS"
  healthy_server_count="$(echo "$VLLM_SERVER_URLS" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' ')"
  EXPDIS_VLLM_CONCURRENCY_VALUE="${EXPDIS_VLLM_CONCURRENCY:-}"
  if [ -z "$EXPDIS_VLLM_CONCURRENCY_VALUE" ]; then
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
      EXPDIS_VLLM_CONCURRENCY_VALUE="$target_concurrency"
    else
      EXPDIS_VLLM_CONCURRENCY_VALUE="$concurrency_cap"
    fi
    if [ "$EXPDIS_VLLM_CONCURRENCY_VALUE" -lt 8 ]; then
      EXPDIS_VLLM_CONCURRENCY_VALUE=8
    fi
  fi
  reload_worker_url="$(echo "$VLLM_SERVER_URLS" | tr ',' '\n' | sed '/^$/d' | head -n 1)"
elif [ "$SKIP_INFERENCE_SLICE" != "1" ]; then
  log "waiting for vLLM health ($VLLM_SERVER_COUNT servers)"
  health_check_script="
IPS='$INFER_VLLM_IPS'
PORT='$PORT'
TOTAL='$VLLM_SERVER_COUNT'
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
echo '[health] ERROR: inference slice not healthy'
exit 1
"

  if [ "$SINGLE_SLICE_MODE" = "1" ]; then
    health_output="$(train_ssh "$health_check_script")"
  else
    health_output="$(expdis_gcloud alpha compute tpus tpu-vm ssh "$INFER_TPU_NAME" \
      --worker=0 --zone="$INFER_ZONE" "$INFER_TRANSPORT_FLAG" \
      --command="$health_check_script")"
  fi

  echo "$health_output" | grep -v '^URLS=' | while read -r line; do
    log "$line"
  done
  VLLM_SERVER_URLS="$(echo "$health_output" | grep '^URLS=' | head -1 | sed 's/^URLS=//')"
  if [ -z "$VLLM_SERVER_URLS" ]; then
    log "ERROR: failed to discover healthy vLLM servers"
    exit 1
  fi

  healthy_server_count="$(echo "$VLLM_SERVER_URLS" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' ')"
  EXPDIS_VLLM_CONCURRENCY_VALUE="${EXPDIS_VLLM_CONCURRENCY:-}"
  if [ -z "$EXPDIS_VLLM_CONCURRENCY_VALUE" ]; then
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
      EXPDIS_VLLM_CONCURRENCY_VALUE="$target_concurrency"
    else
      EXPDIS_VLLM_CONCURRENCY_VALUE="$concurrency_cap"
    fi
    if [ "$EXPDIS_VLLM_CONCURRENCY_VALUE" -lt 8 ]; then
      EXPDIS_VLLM_CONCURRENCY_VALUE=8
    fi
  fi
  reload_worker_url="$(echo "$INFER_VLLM_IPS" | tr ',' '\n' | sed -n "$((RELOAD_WORKER + 1))p" | sed "s#^#http://#; s#\$#:${PORT}/v1#")"
else
  VLLM_SERVER_URLS=""
  healthy_server_count=0
  EXPDIS_VLLM_CONCURRENCY_VALUE="${EXPDIS_VLLM_CONCURRENCY:-8}"
  reload_worker_url=""
fi
RUN_DIR="runs/$RUN_NAME"

TRAIN_ARGS=(
  run_expdis_torch.py
  --device "${TRAIN_DEVICE:-tpu}"
  --use-wandb
  --seed "${SEED:-0}"
  --model-name "$MODEL_NAME"
  --dataset-name "$DATASET_NAME"
  --train-selection-policy random
  --train-probe-pool-size "${TRAIN_PROBE_POOL_SIZE:-256}"
  --train-probe-samples-per-prompt "${TRAIN_PROBE_SAMPLES_PER_PROMPT:-4}"
  --train-probe-temperature "${TRAIN_PROBE_TEMPERATURE:-0.8}"
  --train-probe-batch-size "${TRAIN_PROBE_BATCH_SIZE:-4}"
  --dataset-split-policy "${DATASET_SPLIT_POLICY:-auto}"
  --topic-split-mode "${TOPIC_SPLIT_MODE:-heuristic_math_topics}"
  --topic-partition-policy "${TOPIC_PARTITION_POLICY:-explicit_or_heuristic}"
  --decoupling-mode "${DECOUPLING_MODE:-decoupled}"
  --base-eval-mode "${BASE_EVAL_MODE:-raw}"
  --actual-target-contract "$ACTUAL_TARGET_CONTRACT"
  --max-train-examples "$MAX_TRAIN_EXAMPLES"
  --max-eval-examples "$MAX_EVAL_EXAMPLES"
  --grpo-max-steps "$GRPO_MAX_STEPS"
  --grpo-batch-size "$GRPO_BATCH_SIZE"
  --grpo-grad-accum "$GRPO_GRAD_ACCUM"
  --grpo-loss-type "${GRPO_LOSS_TYPE:-dr_grpo}"
  --grpo-num-generations "$GRPO_NUM_GENERATIONS"
  --grpo-generation-batch-size "$GRPO_GENERATION_BATCH_SIZE"
  --num-samples-per-prompt "$NUM_SAMPLES_PER_PROMPT"
  --temperature-schedule "$TEMPERATURE_SCHEDULE"
  --answer-constraint-mode "${ANSWER_CONSTRAINT_MODE:-none}"
  --exploration-response-style "${EXPLORATION_RESPONSE_STYLE:-reasoning_final}"
  --eval-response-style "${EVAL_RESPONSE_STYLE:-reasoning_final}"
  --exploration-answer-constraint-mode "${EXPLORATION_ANSWER_CONSTRAINT_MODE:-none}"
  --eval-answer-constraint-mode "${EVAL_ANSWER_CONSTRAINT_MODE:-none}"
  --exploration-thinking-mode "$EXPLORATION_THINKING_MODE"
  --eval-thinking-mode "$EVAL_THINKING_MODE"
  --actual-thinking-mode "$ACTUAL_THINKING_MODE"
  --distill-epochs "${DISTILL_EPOCHS:-2}"
  --distill-batch-size "${DISTILL_BATCH_SIZE:-1}"
  --max-distill-examples "${MAX_DISTILL_EXAMPLES:-$MAX_TRAIN_EXAMPLES}"
  --actual-grpo-epochs "${ACTUAL_GRPO_EPOCHS:-3}"
  --actual-grpo-batch-size "${ACTUAL_GRPO_BATCH_SIZE:-1}"
  --actual-grpo-grad-accum "${ACTUAL_GRPO_GRAD_ACCUM:-1}"
  --actual-min-accepted-trajectories "${ACTUAL_MIN_ACCEPTED_TRAJECTORIES:-4}"
  --actual-dataset-policy "$ACTUAL_DATASET_POLICY"
  --actual-max-incorrect-ratio "${ACTUAL_MAX_INCORRECT_RATIO:-0.25}"
  --actual-target-correct-fraction "${ACTUAL_TARGET_CORRECT_FRACTION:-0.75}"
  --actual-grpo-kl-coef "${ACTUAL_GRPO_KL_COEF:-0.10}"
  --actual-grpo-lr "${ACTUAL_GRPO_LR:-1e-6}"
  --distill-lr "${DISTILL_LR:-5e-6}"
  --max-actual-examples "${MAX_ACTUAL_EXAMPLES:-$MAX_TRAIN_EXAMPLES}"
  --novelty-feature-source "${NOVELTY_FEATURE_SOURCE:-multilayer}"
  --novelty-feature-max-length "${NOVELTY_FEATURE_MAX_LENGTH:-$MAX_TOTAL_LEN}"
  --novelty-metric "${NOVELTY_METRIC:-sqrt_mse}"
  --novelty-text-contract "${NOVELTY_TEXT_CONTRACT:-full_raw}"
  --rnd-hidden "${RND_HIDDEN:-512}"
  --reward-correct "${REWARD_CORRECT:-1.0}"
  --reward-incorrect "${REWARD_INCORRECT:-0.0}"
  --lambda-novelty "${LAMBDA_NOVELTY:-0.03}"
  --reward-shape "${REWARD_SHAPE:-linear}"
  --anchor-metric "${ANCHOR_METRIC:-none}"
  --trajectory-batch-size "${TRAJECTORY_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --round-budget-mode "${ROUND_BUDGET_MODE:-constant_total}"
  --max-prompt-len "$MAX_PROMPT_LEN"
  --max-completion-len "$MAX_COMPLETION_LEN"
  --exploration-max-completion-len "$EXPLORATION_MAX_COMPLETION_LEN"
  --eval-max-completion-len "$EVAL_MAX_COMPLETION_LEN"
  --max-total-len "$MAX_TOTAL_LEN"
  --format-penalty-weight "${FORMAT_PENALTY_WEIGHT:-0.1}"
  --pipeline-mode "${PIPELINE_MODE:-two_model}"
  --actual-init-source "${ACTUAL_INIT_SOURCE:-base}"
  --actual-train-mode "${ACTUAL_TRAIN_MODE:-sft_grpo}"
  --actual-reward-source "$ACTUAL_REWARD_SOURCE"
  --wandb-run-name "$WANDB_RUN_NAME"
  --plots-dir "$RUN_DIR/plots"
  --explorer-traj-path "$RUN_DIR/explorer_trajectories.jsonl"
  --explorer-output-dir "$RUN_DIR/explorer_ckpts"
  --saved-model-root "${SAVED_MODEL_ROOT:-$TRAIN_REMOTE_ROOT/$RUN_DIR/saved_models}"
)

if [ -n "${INCORRECT_NOVELTY_SCALE:-}" ]; then
  TRAIN_ARGS+=(--incorrect-novelty-scale "$INCORRECT_NOVELTY_SCALE")
fi

if [ -n "$ACTUAL_ONLINE_GRPO_MAX_STEPS" ]; then
  TRAIN_ARGS+=(--actual-online-grpo-max-steps "$ACTUAL_ONLINE_GRPO_MAX_STEPS")
fi

if [ -n "${NOVELTY_ZSCORE:-}" ]; then
  TRAIN_ARGS+=(--novelty-zscore "$NOVELTY_ZSCORE")
fi

if [ -n "${BENCHMARK_DATASET_NAME:-}" ]; then
  TRAIN_ARGS+=(--benchmark-dataset-name "$BENCHMARK_DATASET_NAME")
  TRAIN_ARGS+=(--benchmark-eval-mode "${BENCHMARK_EVAL_MODE:-sampled}")
  TRAIN_ARGS+=(--benchmark-num-rollouts "${BENCHMARK_NUM_ROLLOUTS:-32}")
  TRAIN_ARGS+=(--max-benchmark-examples "${MAX_BENCHMARK_EXAMPLES:-30}")
fi

printf -v quoted_train_args '%q ' "${TRAIN_ARGS[@]}"

log "resolved run shape: target=${TRAIN_TPU_NAME}/worker${TRAIN_WORKER} launch_mode=${TRAIN_LAUNCH_MODE:-python} single_slice=${SINGLE_SLICE_MODE} prompt=${MAX_PROMPT_LEN} completion=${MAX_COMPLETION_LEN} total=${MAX_TOTAL_LEN} train_examples=${MAX_TRAIN_EXAMPLES} eval_examples=${MAX_EVAL_EXAMPLES} grpo_steps=${GRPO_MAX_STEPS} batch=${GRPO_BATCH_SIZE} grad_accum=${GRPO_GRAD_ACCUM} generations=${GRPO_NUM_GENERATIONS} samples_per_prompt=${NUM_SAMPLES_PER_PROMPT} traj_batch=${TRAJECTORY_BATCH_SIZE} eval_batch=${EVAL_BATCH_SIZE} flash=${EXPDIS_XLA_FLASH_ATTENTION:-1} force_eager=${EXPDIS_XLA_FORCE_EAGER_ATTN:-0}"

TRAIN_DEVICE="${TRAIN_DEVICE:-tpu}"
TRAIN_LAUNCH_MODE="${TRAIN_LAUNCH_MODE:-python}"
EXPDIS_GRPO_GRAD_CKPT_DEFAULT=1
if [ "$TRAIN_DEVICE" = "tpu" ] && [ "$TRAIN_LAUNCH_MODE" = "spmd_local" ]; then
  EXPDIS_GRPO_GRAD_CKPT_DEFAULT=0
fi

tpu_env_block=""
if [ "$TRAIN_DEVICE" = "tpu" ]; then
  if [ "$TRAIN_LAUNCH_MODE" = "ddp_local" ]; then
    tpu_env_block="export PJRT_DEVICE=TPU && \
   export EXPDIS_TPU_SLICE_LAUNCHED=1 && \
   export EXPDIS_REQUIRE_TPU=1 && \
   export EXPDIS_USE_XLA_WORLD_SIZE='${EXPDIS_USE_XLA_WORLD_SIZE:-1}' && \
   export TPU_SKIP_MDS_QUERY=1 && \
   export TPU_ACCELERATOR_TYPE='$TRAIN_ACCELERATOR_TYPE' && \
   export DDP_VISIBLE_CHIPS='${DDP_VISIBLE_CHIPS:-0,1,2,3}' && \
   export DDP_NUM_PROCESSES='${DDP_NUM_PROCESSES:-4}' && \
   export EXPDIS_XLA_PERSISTENT_CACHE_DIR='${EXPDIS_XLA_PERSISTENT_CACHE_DIR:-/tmp/expdis_xla_cache_shared}' && \
   export EXPDIS_XLA_SAFE_GENERATE='${EXPDIS_XLA_SAFE_GENERATE:-1}' && \
   export EXPDIS_XLA_INCREMENTAL_GENERATE='${EXPDIS_XLA_INCREMENTAL_GENERATE:-1}' && \
   export EXPDIS_XLA_SKIP_LOADER_MARK_STEP='${EXPDIS_XLA_SKIP_LOADER_MARK_STEP:-1}' && "
  elif [ "$TRAIN_LAUNCH_MODE" = "fsdp_local" ]; then
    tpu_env_block="export PJRT_DEVICE=TPU && \
   export EXPDIS_TPU_SLICE_LAUNCHED=1 && \
   export EXPDIS_REQUIRE_TPU=1 && \
   export EXPDIS_FSDP=1 && \
   export EXPDIS_FSDP_WRAP_CLS='${EXPDIS_FSDP_WRAP_CLS:-Qwen3DecoderLayer}' && \
   export EXPDIS_USE_XLA_WORLD_SIZE='${EXPDIS_USE_XLA_WORLD_SIZE:-1}' && \
   export TPU_SKIP_MDS_QUERY=1 && \
   export TPU_ACCELERATOR_TYPE='$TRAIN_ACCELERATOR_TYPE' && \
   export FSDP_VISIBLE_CHIPS='${FSDP_VISIBLE_CHIPS:-0,1,2,3}' && \
   export FSDP_NUM_PROCESSES='${FSDP_NUM_PROCESSES:-4}' && \
   export EXPDIS_XLA_PERSISTENT_CACHE_DIR='${EXPDIS_XLA_PERSISTENT_CACHE_DIR:-/tmp/expdis_xla_cache_shared}' && \
   export EXPDIS_XLA_SAFE_GENERATE='${EXPDIS_XLA_SAFE_GENERATE:-1}' && \
   export EXPDIS_XLA_INCREMENTAL_GENERATE='${EXPDIS_XLA_INCREMENTAL_GENERATE:-1}' && \
   export EXPDIS_XLA_SKIP_LOADER_MARK_STEP='${EXPDIS_XLA_SKIP_LOADER_MARK_STEP:-1}' && "
  elif [ "$TRAIN_LAUNCH_MODE" = "spmd_local" ]; then
    tpu_env_block="export PJRT_DEVICE=TPU && \
   export EXPDIS_FSDP=1 && \
   export EXPDIS_USE_XLA_WORLD_SIZE=1 && \
   export EXPDIS_FSDP_WRAP_CLS='${EXPDIS_FSDP_WRAP_CLS:-Qwen3DecoderLayer}' && \
   export EXPDIS_TPU_SLICE_LAUNCHED=1 && \
   export EXPDIS_REQUIRE_TPU=1 && \
   export EXPDIS_ALLOW_SINGLE_WORKER_TPU=1 && \
   export TPU_SKIP_MDS_QUERY=1 && \
   export TPU_WORKER_HOSTNAMES=localhost,localhost,localhost,localhost && \
   export TPU_WORKER_ID=0 && \
   export TPU_HOST_BOUNDS=2,2,1 && \
   export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 && \
   export TPU_PROCESS_BOUNDS=1,1,1 && \
   export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 && \
   export TPU_ACCELERATOR_TYPE='$TRAIN_ACCELERATOR_TYPE' && \
   export EXPDIS_XLA_PERSISTENT_CACHE_DIR='${EXPDIS_XLA_PERSISTENT_CACHE_DIR:-/tmp/expdis_xla_cache_shared}' && \
   export EXPDIS_XLA_SAFE_GENERATE='${EXPDIS_XLA_SAFE_GENERATE:-1}' && \
   export EXPDIS_XLA_INCREMENTAL_GENERATE='${EXPDIS_XLA_INCREMENTAL_GENERATE:-1}' && \
   export EXPDIS_XLA_SKIP_LOADER_MARK_STEP='${EXPDIS_XLA_SKIP_LOADER_MARK_STEP:-1}' && \
   export EXPDIS_XLA_PLACEHOLDER_METRICS='${EXPDIS_XLA_PLACEHOLDER_METRICS:-1}' && \
   export EXPDIS_XLA_SKIP_ACCELERATE_SYNC_MARK_STEP='${EXPDIS_XLA_SKIP_ACCELERATE_SYNC_MARK_STEP:-1}' && \
   export EXPDIS_XLA_PLACEHOLDER_STEP_LOSS='${EXPDIS_XLA_PLACEHOLDER_STEP_LOSS:-1}' && \
   export EXPDIS_XLA_CPU_INPUT_HANDOFF='${EXPDIS_XLA_CPU_INPUT_HANDOFF:-1}' && "
  else
    tpu_env_block="export PJRT_DEVICE=TPU && \
   export EXPDIS_FSDP=${EXPDIS_FSDP:-1} && \
   export EXPDIS_USE_XLA_WORLD_SIZE=${EXPDIS_USE_XLA_WORLD_SIZE:-1} && \
   export EXPDIS_FSDP_WRAP_CLS='Qwen3DecoderLayer' && \
   export EXPDIS_TPU_SLICE_LAUNCHED=1 && \
   export EXPDIS_REQUIRE_TPU=1 && \
   export EXPDIS_ALLOW_SINGLE_WORKER_TPU=1 && \
   export TPU_SKIP_MDS_QUERY=1 && \
   export TPU_WORKER_HOSTNAMES=localhost && \
   export TPU_WORKER_ID=0 && \
   export TPU_HOST_BOUNDS=1,1,1 && \
   export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 && \
   export TPU_PROCESS_BOUNDS=1,1,1 && \
   export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 && \
   export TPU_ACCELERATOR_TYPE='$TRAIN_ACCELERATOR_TYPE' && \
   export EXPDIS_XLA_PERSISTENT_CACHE_DIR='${EXPDIS_XLA_PERSISTENT_CACHE_DIR:-/tmp/expdis_xla_cache_shared}' && \
   export EXPDIS_XLA_SAFE_GENERATE='${EXPDIS_XLA_SAFE_GENERATE:-1}' && \
   export EXPDIS_XLA_INCREMENTAL_GENERATE='${EXPDIS_XLA_INCREMENTAL_GENERATE:-1}' && \
   export EXPDIS_XLA_SKIP_LOADER_MARK_STEP='${EXPDIS_XLA_SKIP_LOADER_MARK_STEP:-1}' && \
   export EXPDIS_XLA_PLACEHOLDER_METRICS='${EXPDIS_XLA_PLACEHOLDER_METRICS:-1}' && \
   export EXPDIS_XLA_SKIP_ACCELERATE_SYNC_MARK_STEP='${EXPDIS_XLA_SKIP_ACCELERATE_SYNC_MARK_STEP:-1}' && \
   export EXPDIS_XLA_PLACEHOLDER_STEP_LOSS='${EXPDIS_XLA_PLACEHOLDER_STEP_LOSS:-1}' && \
   export EXPDIS_XLA_CPU_INPUT_HANDOFF='${EXPDIS_XLA_CPU_INPUT_HANDOFF:-1}' && "
  fi
fi

train_entrypoint="python -u ${quoted_train_args}"
if [ "$TRAIN_DEVICE" = "tpu" ] && [ "$TRAIN_LAUNCH_MODE" = "ddp_local" ]; then
  train_entrypoint="bash scripts/run_tpu_ddp_local.sh ${quoted_train_args}"
elif [ "$TRAIN_DEVICE" = "tpu" ] && [ "$TRAIN_LAUNCH_MODE" = "fsdp_local" ]; then
  train_entrypoint="bash scripts/run_tpu_fsdp_local.sh ${quoted_train_args}"
elif [ "$TRAIN_DEVICE" = "tpu" ] && [ "$TRAIN_LAUNCH_MODE" = "spmd_local" ]; then
  train_entrypoint="python -u ${quoted_train_args}"
fi

remote_cmd=$(
  cat <<EOF
set -euo pipefail
cd '$TRAIN_REMOTE_ROOT'
tmux kill-session -t 'training_$RUN_NAME' 2>/dev/null || true
pkill -9 -f "from multiprocessing.spawn import spawn_mai[n]" 2>/dev/null || true
pkill -9 -f "from multiprocessing.resource_tracker import mai[n]" 2>/dev/null || true
# Kill any lingering vLLM on this worker so training can claim TPU chips.
pkill -9 -f '[v]llm serve' 2>/dev/null || true
fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
sleep 2
mkdir -p '$TRAIN_REMOTE_ROOT/$RUN_DIR'
cat > /tmp/training_${RUN_NAME}.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd '$TRAIN_REMOTE_ROOT'
source .venv_tpu_slice/bin/activate
export WANDB_PROJECT='$WANDB_PROJECT'
export WANDB_ENTITY='$WANDB_ENTITY'
export WANDB_API_KEY='${WANDB_API_KEY:-}'
export HF_ACCESS_TOKEN='${HF_ACCESS_TOKEN:-}'
export HF_TOKEN='${HF_TOKEN:-}'
export HUGGING_FACE_HUB_TOKEN='${HUGGING_FACE_HUB_TOKEN:-}'
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
${tpu_env_block}export EXPDIS_GRPO_GRAD_CKPT='${EXPDIS_GRPO_GRAD_CKPT:-$EXPDIS_GRPO_GRAD_CKPT_DEFAULT}'
export EXPDIS_DECOUPLED_GRAD_CKPT='${EXPDIS_DECOUPLED_GRAD_CKPT:-1}'
export EXPDIS_DECOUPLED_TRAIN_CPU='${EXPDIS_DECOUPLED_TRAIN_CPU:-0}'
export EXPDIS_EXPLORER_DECOUPLED_GRPO='${EXPDIS_EXPLORER_DECOUPLED_GRPO:-1}'
export EXPDIS_TPU_HOST_TORCH_THREADS='${EXPDIS_TPU_HOST_TORCH_THREADS:-4}'
export EXPDIS_TPU_HOST_TORCH_INTEROP_THREADS='${EXPDIS_TPU_HOST_TORCH_INTEROP_THREADS:-1}'
export EXPDIS_GRPO_LOGPROB_CHUNK_SIZE='${EXPDIS_GRPO_LOGPROB_CHUNK_SIZE:-2}'
export EXPDIS_GRPO_USE_VLLM='${EXPDIS_GRPO_USE_VLLM:-0}'
export EXPDIS_NOVELTY_BATCH_SIZE='${EXPDIS_NOVELTY_BATCH_SIZE:-16}'
export EXPDIS_NOVELTY_ON_TPU='${EXPDIS_NOVELTY_ON_TPU:-1}'
export EXPDIS_DECOUPLED_TRAIN_TOKEN_CHUNK='${EXPDIS_DECOUPLED_TRAIN_TOKEN_CHUNK:-2048}'
export EXPDIS_SKIP_VLLM_LOGPROBS='${EXPDIS_SKIP_VLLM_LOGPROBS:-0}'
export EXPDIS_EXPLORER_VLLM_RELOAD_EVERY_N_ROUNDS='${EXPDIS_EXPLORER_VLLM_RELOAD_EVERY_N_ROUNDS:-0}'
export EXPDIS_WANDB_RUN_NAME='${WANDB_RUN_NAME:-}'
export EXPDIS_EXPLORER_TRAIN_TOP_P='${EXPDIS_EXPLORER_TRAIN_TOP_P:-0.95}'
export EXPDIS_EXPLORER_TRAIN_TOP_K='${EXPDIS_EXPLORER_TRAIN_TOP_K:-20}'
export EXPDIS_EXPLORER_DECOUPLED_BUFFER_GROUPS='${EXPDIS_EXPLORER_DECOUPLED_BUFFER_GROUPS:-32}'
export EXPDIS_EXPLORER_DECOUPLED_BATCH_GROUPS='${EXPDIS_EXPLORER_DECOUPLED_BATCH_GROUPS:-1}'
export EXPDIS_EXPLORER_DECOUPLED_GRAD_ACCUM='${EXPDIS_EXPLORER_DECOUPLED_GRAD_ACCUM:-1}'
export EXPDIS_EXPLORER_DECOUPLED_ROW_MICROBATCH='${EXPDIS_EXPLORER_DECOUPLED_ROW_MICROBATCH:-1}'
export EXPDIS_EXPLORER_DECOUPLED_TRAIN_FSDP='${EXPDIS_EXPLORER_DECOUPLED_TRAIN_FSDP:-0}'
export EXPDIS_EXPLORER_DECOUPLED_TRAIN_FSDP_GRAD_CKPT='${EXPDIS_EXPLORER_DECOUPLED_TRAIN_FSDP_GRAD_CKPT:-1}'
export EXPDIS_DECOUPLED_TRAIN_DEBUG_PROGRESS='${EXPDIS_DECOUPLED_TRAIN_DEBUG_PROGRESS:-0}'
export EXPDIS_DECOUPLED_ROW_MARK_STEP='${EXPDIS_DECOUPLED_ROW_MARK_STEP:-1}'
export EXPDIS_DECOUPLED_ACCUM_MARK_STEP='${EXPDIS_DECOUPLED_ACCUM_MARK_STEP:-1}'
export EXPDIS_DECOUPLED_ACCUM_MARK_RESET_SCOPE='${EXPDIS_DECOUPLED_ACCUM_MARK_RESET_SCOPE:-0}'
export EXPDIS_DECOUPLED_PRE_OPTIMIZER_MARK_STEP='${EXPDIS_DECOUPLED_PRE_OPTIMIZER_MARK_STEP:-1}'
export EXPDIS_DECOUPLED_TRAIN_FIXED_SHAPES='${EXPDIS_DECOUPLED_TRAIN_FIXED_SHAPES:-1}'
export EXPDIS_DECOUPLED_TRAIN_FIXED_PROMPT_WIDTH='${EXPDIS_DECOUPLED_TRAIN_FIXED_PROMPT_WIDTH:-0}'
export EXPDIS_DECOUPLED_TRAIN_FIXED_COMPLETION_WIDTH='${EXPDIS_DECOUPLED_TRAIN_FIXED_COMPLETION_WIDTH:-0}'
export EXPDIS_DEBUG_GENERATE='${EXPDIS_DEBUG_GENERATE:-0}'
export EXPDIS_XLA_GENERATE_PROGRESS_EVERY='${EXPDIS_XLA_GENERATE_PROGRESS_EVERY:-0}'
export EXPDIS_WANDB_MINIMAL='${EXPDIS_WANDB_MINIMAL:-1}'
export EXPDIS_DISABLE_BF16='${EXPDIS_DISABLE_BF16:-0}'
export EXPDIS_STRICT_GRPO_GEOMETRY='${EXPDIS_STRICT_GRPO_GEOMETRY:-0}'
export EXPDIS_EXPLORER_PRETRAINED_PATH='${EXPDIS_EXPLORER_PRETRAINED_PATH:-}'
export EXPDIS_TRAJ_USE_VLLM='${EXPDIS_TRAJ_USE_VLLM:-0}'
export EXPDIS_XLA_FLASH_ATTENTION='${EXPDIS_XLA_FLASH_ATTENTION:-1}'
export EXPDIS_XLA_FORCE_EAGER_ATTN='${EXPDIS_XLA_FORCE_EAGER_ATTN:-0}'
export EXPDIS_RESUME_STEP='${EXPDIS_RESUME_STEP:-0}'
export EXPDIS_RESUME_CHECKPOINT='${EXPDIS_RESUME_CHECKPOINT:-}'
export EXPDIS_PRECOMPUTED_TRAJ_PATH='${EXPDIS_PRECOMPUTED_TRAJ_PATH:-}'
export EXPDIS_RESUME_TRAINING='${EXPDIS_RESUME_TRAINING:-0}'
export EXPDIS_ALLOW_XLA_GRPO_INTERMEDIATE_SAVE='${EXPDIS_ALLOW_XLA_GRPO_INTERMEDIATE_SAVE:-0}'
export EXPDIS_TRAJ_TARGET_ACCEPTED='${EXPDIS_TRAJ_TARGET_ACCEPTED:-2048}'
export EXPDIS_DEEPINFRA_API_KEY='${EXPDIS_DEEPINFRA_API_KEY:-}'
export EXPDIS_TRAJ_MAX_COMPLETION_LEN='${TRAJ_MAX_COMPLETION_LEN}'
export EXPDIS_GRPO_LOGGING_STRATEGY='${EXPDIS_GRPO_LOGGING_STRATEGY:-steps}'
export EXPDIS_GRPO_LOGGING_STEPS='${EXPDIS_GRPO_LOGGING_STEPS:-1}'
export EXPDIS_GRPO_SAVE_STRATEGY='${EXPDIS_GRPO_SAVE_STRATEGY:-steps}'
export EXPDIS_GRPO_SAVE_STEPS='${EXPDIS_GRPO_SAVE_STEPS:-4}'
export EXPDIS_GRPO_SAVE_TOTAL_LIMIT='${EXPDIS_GRPO_SAVE_TOTAL_LIMIT:-3}'
export EXPDIS_ACTUAL_GRPO_SAVE_STEPS='${EXPDIS_ACTUAL_GRPO_SAVE_STEPS:-25}'
export EXPDIS_ACTUAL_GRPO_SAVE_TOTAL_LIMIT='${EXPDIS_ACTUAL_GRPO_SAVE_TOTAL_LIMIT:-4}'
export EXPDIS_SKIP_INTERMEDIATE_EVALS='${EXPDIS_SKIP_INTERMEDIATE_EVALS:-0}'
export EXPDIS_FINAL_EVAL_ONLY_ACTUAL='${EXPDIS_FINAL_EVAL_ONLY_ACTUAL:-0}'
export EXPDIS_FINAL_EVAL_ROLES='${EXPDIS_FINAL_EVAL_ROLES:-}'
export EXPDIS_FINAL_EVAL_MAX_EXAMPLES='${EXPDIS_FINAL_EVAL_MAX_EXAMPLES:-}'
export EXPDIS_SKIP_FINAL_BENCHMARK_EVALS='${EXPDIS_SKIP_FINAL_BENCHMARK_EVALS:-0}'
export EXPDIS_FINAL_BENCHMARK_MAX_EXAMPLES='${EXPDIS_FINAL_BENCHMARK_MAX_EXAMPLES:-}'
export EXPDIS_STOP_AFTER_EXPLORER='${EXPDIS_STOP_AFTER_EXPLORER:-0}'
export EXPDIS_STOP_AFTER_TRAJECTORIES='${EXPDIS_STOP_AFTER_TRAJECTORIES:-0}'
export EXPDIS_ENABLE_THINKING='${EXPDIS_ENABLE_THINKING:-auto}'
export EXPDIS_OPTIM='${EXPDIS_OPTIM:-adamw_torch}'
export EXPDIS_DISTILL_EAGER_ATTN='${EXPDIS_DISTILL_EAGER_ATTN:-0}'
export EXPDIS_DISTILL_MAX_TOTAL_LEN='${EXPDIS_DISTILL_MAX_TOTAL_LEN:-0}'
export EXPDIS_DISTILL_OPTIM='${EXPDIS_DISTILL_OPTIM:-adafactor}'
export EXPDIS_DISTILL_GRAD_CKPT='${EXPDIS_DISTILL_GRAD_CKPT:-1}'
export EXPDIS_DISTILL_CE_CHUNK='${EXPDIS_DISTILL_CE_CHUNK:-256}'
export EXPDIS_ABORT_ON_DEGENERATE_EXPLORER='${EXPDIS_ABORT_ON_DEGENERATE_EXPLORER:-1}'
export EXPDIS_ABORT_ON_BAD_TRAJ_WINDOW='${EXPDIS_ABORT_ON_BAD_TRAJ_WINDOW:-1}'
export EXPDIS_FEW_SHOT_EXAMPLES='${EXPDIS_FEW_SHOT_EXAMPLES:-0}'
export EXPDIS_VLLM_SERVER_URLS='$VLLM_SERVER_URLS'
export EXPDIS_VLLM_MODEL_NAME='$SERVED_MODEL_NAME'
export EXPDIS_VLLM_CONCURRENCY='$EXPDIS_VLLM_CONCURRENCY_VALUE'
export EXPDIS_VLLM_MAX_RETRIES='${EXPDIS_VLLM_MAX_RETRIES:-3}'
export EXPDIS_GCS_CHECKPOINT_BASE='$RUN_ARTIFACT_GCS_PREFIX'
export EXPDIS_EXPLORER_VLLM_RELOAD_TPU_NAME='$INFER_TPU_NAME'
export EXPDIS_EXPLORER_VLLM_RELOAD_ZONE='$INFER_ZONE'
export EXPDIS_EXPLORER_VLLM_RELOAD_WORKERS='${EXPDIS_EXPLORER_VLLM_RELOAD_WORKERS:-all}'
export EXPDIS_EXPLORER_VLLM_MODEL_NAME='${EXPDIS_EXPLORER_VLLM_MODEL_NAME:-explorer_model}'
export EXPDIS_VLLM_RELOAD_TPU_NAME='$INFER_TPU_NAME'
export EXPDIS_VLLM_RELOAD_ZONE='$INFER_ZONE'
export EXPDIS_VLLM_RELOAD_WORKER='$RELOAD_WORKER'
export EXPDIS_VLLM_RELOAD_WORKER_URL='$reload_worker_url'
export EXPDIS_ACTUAL_VLLM_MODEL_NAME='${EXPDIS_ACTUAL_VLLM_MODEL_NAME:-actual_model}'
export PORT='$PORT'
export TP_SIZE='$TP_SIZE'
export MAX_MODEL_LEN='$MAX_MODEL_LEN'
export MAX_NUM_SEQS='$MAX_NUM_SEQS'
export MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS'
export VLLM_VENV_DIR='$VLLM_VENV_DIR'
export EXPDIS_GCS_CHECKPOINT_ROOT='${EXPDIS_GCS_CHECKPOINT_ROOT:-}'
# Pre-flight: download precomputed trajectories from GCS if needed
if [ -n "${EXPDIS_PRECOMPUTED_TRAJ_PATH:-}" ] && [ ! -f "${EXPDIS_PRECOMPUTED_TRAJ_PATH:-}" ]; then
  GCS_TRAJ="${EXPDIS_GCS_TRAJ_PATH:-}"
  if [ -n "\$GCS_TRAJ" ]; then
    echo "[pre-flight] Downloading trajectories from \$GCS_TRAJ"
    mkdir -p "\$(dirname "${EXPDIS_PRECOMPUTED_TRAJ_PATH:-}")"
    gsutil cp "\$GCS_TRAJ" "${EXPDIS_PRECOMPUTED_TRAJ_PATH:-}"
  fi
fi
# Pre-flight: restore XLA cache from GCS (skip 15-20 min recompile after preemption)
XLA_CACHE_DIR='${EXPDIS_XLA_PERSISTENT_CACHE_DIR:-/tmp/expdis_xla_cache_shared}'
XLA_CACHE_GCS="gs://llmvae-checkpoints/two-model-exploration/xla_cache/v6e-64"
mkdir -p "\$XLA_CACHE_DIR"
local_count=\$(find "\$XLA_CACHE_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)
if [ "\${EXPDIS_XLA_CACHE_RESTORE:-0}" = "1" ] && [ "\$local_count" -lt 10 ]; then
  echo "[xla-cache] restoring from \$XLA_CACHE_GCS (opt-in via EXPDIS_XLA_CACHE_RESTORE=1)"
  timeout 300 gsutil -m -q rsync -r "\$XLA_CACHE_GCS" "\$XLA_CACHE_DIR" 2>&1 | tail -3 || echo "[xla-cache] restore failed or empty; continuing"
  restored=\$(find "\$XLA_CACHE_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l)
  echo "[xla-cache] restored \$restored files"
else
  echo "[xla-cache] restore disabled (EXPDIS_XLA_CACHE_RESTORE!=1 or local already has \$local_count files); skipping — recompile will take 10-20 min but won't fill disk"
fi
# Background sync: push local cache to GCS every 90s while training runs
cat > /tmp/xla_cache_sync.sh <<'SYNC'
#!/bin/bash
XLA_CACHE_DIR='${EXPDIS_XLA_PERSISTENT_CACHE_DIR:-/tmp/expdis_xla_cache_shared}'
XLA_CACHE_GCS="gs://llmvae-checkpoints/two-model-exploration/xla_cache/v6e-64"
while true; do
  sleep 90
  timeout 120 gsutil -m -q rsync -r "\$XLA_CACHE_DIR" "\$XLA_CACHE_GCS" 2>/dev/null | tail -1 || true
done
SYNC
chmod +x /tmp/xla_cache_sync.sh
nohup bash /tmp/xla_cache_sync.sh > /tmp/xla_cache_sync.log 2>&1 &
echo "[xla-cache] background sync daemon launched PID=\$!"
mkdir -p '$TRAIN_REMOTE_ROOT/$RUN_DIR'
cat > '$TRAIN_REMOTE_ROOT/$RUN_DIR/launch_signature.env' <<'SIG'
RUN_NAME='$RUN_NAME'
TRAIN_TPU_NAME='$TRAIN_TPU_NAME'
TRAIN_ZONE='$TRAIN_ZONE'
TRAIN_WORKER='$TRAIN_WORKER'
SINGLE_SLICE_MODE='$SINGLE_SLICE_MODE'
TRAIN_LAUNCH_MODE='${TRAIN_LAUNCH_MODE:-python}'
DDP_VISIBLE_CHIPS='${DDP_VISIBLE_CHIPS:-}'
DDP_NUM_PROCESSES='${DDP_NUM_PROCESSES:-}'
MAX_PROMPT_LEN='$MAX_PROMPT_LEN'
MAX_COMPLETION_LEN='$MAX_COMPLETION_LEN'
EXPLORATION_MAX_COMPLETION_LEN='$EXPLORATION_MAX_COMPLETION_LEN'
EVAL_MAX_COMPLETION_LEN='$EVAL_MAX_COMPLETION_LEN'
MAX_TOTAL_LEN='$MAX_TOTAL_LEN'
MAX_TRAIN_EXAMPLES='$MAX_TRAIN_EXAMPLES'
MAX_EVAL_EXAMPLES='$MAX_EVAL_EXAMPLES'
GRPO_MAX_STEPS='$GRPO_MAX_STEPS'
GRPO_BATCH_SIZE='$GRPO_BATCH_SIZE'
GRPO_GRAD_ACCUM='$GRPO_GRAD_ACCUM'
GRPO_NUM_GENERATIONS='$GRPO_NUM_GENERATIONS'
GRPO_GENERATION_BATCH_SIZE='$GRPO_GENERATION_BATCH_SIZE'
NUM_SAMPLES_PER_PROMPT='$NUM_SAMPLES_PER_PROMPT'
TRAJECTORY_BATCH_SIZE='$TRAJECTORY_BATCH_SIZE'
EVAL_BATCH_SIZE='$EVAL_BATCH_SIZE'
EXPDIS_FSDP='${EXPDIS_FSDP:-1}'
EXPDIS_USE_XLA_WORLD_SIZE='${EXPDIS_USE_XLA_WORLD_SIZE:-1}'
EXPDIS_TRAJ_USE_VLLM='${EXPDIS_TRAJ_USE_VLLM:-0}'
EXPDIS_XLA_FLASH_ATTENTION='${EXPDIS_XLA_FLASH_ATTENTION:-1}'
EXPDIS_XLA_FORCE_EAGER_ATTN='${EXPDIS_XLA_FORCE_EAGER_ATTN:-0}'
EXPDIS_ENABLE_THINKING='${EXPDIS_ENABLE_THINKING:-auto}'
EXPDIS_STRICT_GRPO_GEOMETRY='${EXPDIS_STRICT_GRPO_GEOMETRY:-0}'
SIG
echo "[launch-signature] wrote $TRAIN_REMOTE_ROOT/$RUN_DIR/launch_signature.env"
cat '$TRAIN_REMOTE_ROOT/$RUN_DIR/launch_signature.env'
python - <<'PY' > '$TRAIN_REMOTE_ROOT/$RUN_DIR/environment_versions.txt'
import importlib
import platform
import sys

print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
for name in (
    "torch",
    "torch_xla",
    "transformers",
    "accelerate",
    "trl",
    "datasets",
    "peft",
    "wandb",
    "jax",
    "libtpu",
):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}={version}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{exc}")
PY
echo "[launch-signature] wrote $TRAIN_REMOTE_ROOT/$RUN_DIR/environment_versions.txt"
cat '$TRAIN_REMOTE_ROOT/$RUN_DIR/environment_versions.txt'
set +e
${train_entrypoint} 2>&1 | tee -a '$TRAIN_REMOTE_ROOT/$RUN_DIR/console.log'
status=\${PIPESTATUS[0]}
set -e
printf '%s\n' "\$status" > '$TRAIN_REMOTE_ROOT/$RUN_DIR/exit_status.txt'
printf '[train-exit] status=%s\n' "\$status" | tee -a '$TRAIN_REMOTE_ROOT/$RUN_DIR/console.log'
exit "\$status"
INNER
chmod +x /tmp/training_${RUN_NAME}.sh
tmux new-session -d -s 'training_$RUN_NAME' 'bash /tmp/training_${RUN_NAME}.sh'
tmux ls | grep 'training_$RUN_NAME'
EOF
)
log "launching TPU training on $TRAIN_TPU_NAME worker $TRAIN_WORKER"
train_ssh "$remote_cmd"

log "launched training run=$RUN_NAME"
if [ "$TRAIN_WORKER" = "0" ]; then
  log "monitor train: ssh $TRAIN_SSH_HOST 'tmux capture-pane -t training_$RUN_NAME -p -S -80'"
else
  log "monitor train: gcloud alpha compute tpus tpu-vm ssh $TRAIN_TPU_NAME --worker=$TRAIN_WORKER --zone=$TRAIN_ZONE --tunnel-through-iap --command=\"tmux capture-pane -t training_$RUN_NAME -p -S -80\""
fi
if [ "$SKIP_INFERENCE_SLICE" != "1" ]; then
  log "monitor infer: gcloud alpha compute tpus tpu-vm ssh $INFER_TPU_NAME --worker=0 --zone=$INFER_ZONE $INFER_TRANSPORT_FLAG --command='tmux ls || true'"
fi
