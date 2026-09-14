#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-us-east1-d}"
TPU_TRANSPORT_FLAG="$(tmx_tpu_transport_flag)"
TPU_PROXY_JUMP_ALIAS="${TPU_PROXY_JUMP_ALIAS:-}"
TPU_SSH_USER="${TPU_SSH_USER:-$USER}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
RUN_PREFIX="${RUN_PREFIX:-use1d_qwen3_1p7b_sharded_$(date +%Y%m%d_%H%M%S)}"

WORKERS="${WORKERS:-all}"
EXCLUDE_WORKERS="${EXCLUDE_WORKERS:-}"
ALLOW_DIRTY="${ALLOW_DIRTY:-1}"
SYNC_REPO="${SYNC_REPO:-1}"
BOOTSTRAP_VLLM="${BOOTSTRAP_VLLM:-1}"
WAIT_FOR_COMPLETION="${WAIT_FOR_COMPLETION:-1}"
CLEANUP_STALE="${CLEANUP_STALE:-1}"

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"

BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
BENCHMARK_DATASET_SPLIT_POLICY="${BENCHMARK_DATASET_SPLIT_POLICY:-first_available}"
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
EVAL_RESPONSE_STYLE="${EVAL_RESPONSE_STYLE:-reasoning_final}"
ANSWER_CONSTRAINT_MODE="${ANSWER_CONSTRAINT_MODE:-none}"
THINKING_MODE="${THINKING_MODE:-auto}"
SKIP_SERVER_LAUNCH="${SKIP_SERVER_LAUNCH:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_GROUP="${WANDB_GROUP:-use1d_vllm_aime_sharded}"
RUN_ARTIFACT_GCS_PREFIX="${RUN_ARTIFACT_GCS_PREFIX:-}"
RUN_SYNC_INTERVAL_SECONDS="${RUN_SYNC_INTERVAL_SECONDS:-90}"
VLLM_ENV_CACHE_GCS_URI="${VLLM_ENV_CACHE_GCS_URI:-}"
BOOTSTRAP_PARALLELISM="${BOOTSTRAP_PARALLELISM:-4}"

LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-runs/${RUN_PREFIX}_pulled}"
MERGED_RUN_DIR="${MERGED_RUN_DIR:-runs/${RUN_PREFIX}_merged}"

log() {
  printf '[sharded-aime] %s\n' "$*"
}

worker_list_csv() {
  python3 - "$TPU_NAME" "$ZONE" "$WORKERS" "$EXCLUDE_WORKERS" <<'PY'
import json
import subprocess
import sys

tpu_name, zone, workers_spec, exclude_spec = sys.argv[1:]
desc = json.loads(
    subprocess.check_output(
        ["gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe", tpu_name, "--zone", zone, "--format=json"]
    )
)
count = len(desc.get("networkEndpoints", []))

def parse(spec: str):
    spec = (spec or "").strip()
    if not spec:
        return []
    if spec == "all":
        return list(range(count))
    values = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values

workers = parse(workers_spec)
excluded = set(parse(exclude_spec))
workers = [w for w in workers if w not in excluded]
if not workers:
    raise SystemExit("no workers selected")
for worker in workers:
    if worker < 0 or worker >= count:
        raise SystemExit(f"worker {worker} out of range for {count} workers")
print(",".join(str(worker) for worker in workers))
PY
}

worker_ip() {
  local worker="$1"
  python3 - "$TPU_NAME" "$ZONE" "$worker" <<'PY'
import json
import subprocess
import sys

tpu_name, zone, worker = sys.argv[1], sys.argv[2], int(sys.argv[3])
desc = json.loads(
    subprocess.check_output(
        ["gcloud", "alpha", "compute", "tpus", "tpu-vm", "describe", tpu_name, "--zone", zone, "--format=json"]
    )
)
print(desc["networkEndpoints"][worker]["ipAddress"])
PY
}

remote_ssh() {
  local worker="$1"
  local command="$2"
  if [ -n "$TPU_PROXY_JUMP_ALIAS" ]; then
    local ip
    ip="$(worker_ip "$worker")"
    ssh \
      -o BatchMode=yes \
      -o StrictHostKeyChecking=no \
      -o ConnectTimeout=20 \
      -J "$TPU_PROXY_JUMP_ALIAS" \
      "${TPU_SSH_USER}@${ip}" \
      "$command"
  else
    gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --worker="$worker" \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG" \
      --command="$command"
  fi
}

remote_scp_pull() {
  local worker="$1"
  local remote_path="$2"
  local local_path="$3"
  if [ -n "$TPU_PROXY_JUMP_ALIAS" ]; then
    local ip
    ip="$(worker_ip "$worker")"
    scp \
      -o BatchMode=yes \
      -o StrictHostKeyChecking=no \
      -o ConnectTimeout=20 \
      -o ProxyJump="$TPU_PROXY_JUMP_ALIAS" \
      -r \
      "${TPU_SSH_USER}@${ip}:$remote_path" \
      "$local_path"
  else
    gcloud alpha compute tpus tpu-vm scp \
      "$TPU_NAME:$remote_path" \
      "$local_path" \
      --worker="$worker" \
      --recurse \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG"
  fi
}

remote_scp_push() {
  local worker="$1"
  local local_path="$2"
  local remote_path="$3"
  if [ -n "$TPU_PROXY_JUMP_ALIAS" ]; then
    local ip
    ip="$(worker_ip "$worker")"
    scp \
      -o BatchMode=yes \
      -o StrictHostKeyChecking=no \
      -o ConnectTimeout=20 \
      -o ProxyJump="$TPU_PROXY_JUMP_ALIAS" \
      "$local_path" \
      "${TPU_SSH_USER}@${ip}:$remote_path"
  else
    gcloud alpha compute tpus tpu-vm scp \
      "$local_path" \
      "${TPU_NAME}:$remote_path" \
      --worker="$worker" \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG"
  fi
}

build_sync_archive() {
  local have_git=1
  local dirty_tree=0
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    have_git=0
    dirty_tree=1
  fi

  if [ "$have_git" = "1" ]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
      dirty_tree=1
    fi
  fi

  if [ "$ALLOW_DIRTY" != "1" ] && [ "$dirty_tree" = "1" ]; then
    echo "Refusing to sync a dirty working tree. Commit or set ALLOW_DIRTY=1." >&2
    exit 1
  fi

  local commit
  if [ "$have_git" = "1" ]; then
    commit="$(git rev-parse --short HEAD)"
  else
    commit="nogit-$(date +%Y%m%d%H%M%S)"
  fi

  local archive="/tmp/two-model-exploration-${commit}.tar"
  if [ "$have_git" = "0" ] || [ "$dirty_tree" = "1" ]; then
    tar \
      --exclude-vcs \
      --exclude='./.git' \
      --exclude='./.venv' \
      --exclude='./.venv*' \
      --exclude='./runs' \
      --exclude='./wandb' \
      --exclude='./.hf' \
      --exclude='./.hf*' \
      --exclude='./__pycache__' \
      --exclude='./tests/__pycache__' \
      -cf "$archive" \
      -C "$ROOT_DIR" \
      .
  else
    git archive --format=tar -o "$archive" HEAD
  fi
  printf '%s\n' "$archive"
}

sync_repo_direct() {
  local archive="$1"
  local remote_archive="/tmp/$(basename "$archive")"
  local active_syncs=0
  for worker in "${worker_ids[@]}"; do
    (
      log "syncing repo to worker $worker via proxy jump"
      remote_scp_push "$worker" "$archive" "$remote_archive"
      remote_ssh "$worker" "mkdir -p '$REMOTE_ROOT' && find '$REMOTE_ROOT' -mindepth 1 -maxdepth 1 ! -name '.venv_tpu_slice' ! -name '.hf' ! -name '.hf*' ! -name 'runs' ! -name 'wandb' ! -name '.wandb_cache' ! -name '.wandb_data' ! -name '.env' ! -name '.env.*' -exec rm -rf {} + && tar -xf '$remote_archive' -C '$REMOTE_ROOT'"
    ) &
    active_syncs=$((active_syncs + 1))
    if [ "$active_syncs" -ge "$BOOTSTRAP_PARALLELISM" ]; then
      wait
      active_syncs=0
    fi
  done
  wait
}

cleanup_worker_runtime() {
  local worker="$1"
  remote_ssh "$worker" "tmux ls 2>/dev/null | sed 's/:.*//' | grep '^vllm_' | xargs -r -n1 tmux kill-session -t || true; pkill -f 'vllm serve' || true; pkill -f 'run_vllm_aime_benchmark.py' || true"
}

workers_csv="$(worker_list_csv)"
IFS=',' read -r -a worker_ids <<< "$workers_csv"
shard_count="${#worker_ids[@]}"

log "TPU=$TPU_NAME zone=$ZONE workers=$workers_csv shard_count=$shard_count"

if [ "$SYNC_REPO" = "1" ]; then
  if [ -n "$TPU_PROXY_JUMP_ALIAS" ]; then
    archive="$(build_sync_archive)"
    sync_repo_direct "$archive"
  else
    log "syncing repo to workers $workers_csv"
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    SYNC_WORKERS="$workers_csv" \
    ALLOW_DIRTY="$ALLOW_DIRTY" \
    bash scripts/sync_tpu_slice_repo.sh
  fi
fi

if [ "$BOOTSTRAP_VLLM" = "1" ]; then
  primary_worker="${worker_ids[0]}"
  if [ "$CLEANUP_STALE" = "1" ]; then
    log "cleaning stale vllm processes on worker $primary_worker"
    cleanup_worker_runtime "$primary_worker"
  fi
  log "bootstrapping primary vllm env on worker $primary_worker"
  remote_ssh "$primary_worker" "cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' bash scripts/bootstrap_vllm_tpu_env.sh"

  active_bootstraps=0
  for worker in "${worker_ids[@]:1}"; do
    if [ "$CLEANUP_STALE" = "1" ]; then
      log "cleaning stale vllm processes on worker $worker"
      cleanup_worker_runtime "$worker"
    fi
    log "bootstrapping cached vllm env on worker $worker"
    (
      remote_ssh "$worker" "cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' bash scripts/bootstrap_vllm_tpu_env.sh"
    ) &
    active_bootstraps=$((active_bootstraps + 1))
    if [ "$active_bootstraps" -ge "$BOOTSTRAP_PARALLELISM" ]; then
      wait
      active_bootstraps=0
    fi
  done
  wait
fi

run_names=()
for shard_index in "${!worker_ids[@]}"; do
  worker="${worker_ids[$shard_index]}"
  run_name="${RUN_PREFIX}_s${shard_index}_w${worker}"
  session_name="vllm_orch_${RUN_PREFIX}_s${shard_index}"
  run_names+=("$run_name")
  log "launching shard $shard_index/$shard_count on worker $worker run=$run_name"
  remote_ssh "$worker" "cd '$REMOTE_ROOT' && mkdir -p 'runs/$run_name' && tmux kill-session -t '$session_name' >/dev/null 2>&1 || true && tmux new-session -d -s '$session_name' \"cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' MODEL_NAME='$MODEL_NAME' SERVED_MODEL_NAME='$SERVED_MODEL_NAME' TOKENIZER_NAME='$TOKENIZER_NAME' RUN_NAME='$run_name' PORT='$PORT' TP_SIZE='$TP_SIZE' MAX_MODEL_LEN='$MAX_MODEL_LEN' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' MAX_NUM_SEQS='$MAX_NUM_SEQS' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' BENCHMARK_DATASET_NAME='$BENCHMARK_DATASET_NAME' BENCHMARK_DATASET_CONFIG_NAME='$BENCHMARK_DATASET_CONFIG_NAME' MAX_BENCHMARK='$MAX_BENCHMARK' BENCHMARK_NUM_ROLLOUTS='$BENCHMARK_NUM_ROLLOUTS' BENCHMARK_ROLLOUT_TEMPERATURE='$BENCHMARK_ROLLOUT_TEMPERATURE' BENCHMARK_ROLLOUT_TOP_P='$BENCHMARK_ROLLOUT_TOP_P' BENCHMARK_ROLLOUT_TOP_K='$BENCHMARK_ROLLOUT_TOP_K' BENCHMARK_ROLLOUT_PASS_K='$BENCHMARK_ROLLOUT_PASS_K' BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN='$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN' REQUEST_CONCURRENCY='$REQUEST_CONCURRENCY' REQUEST_TIMEOUT_SECONDS='$REQUEST_TIMEOUT_SECONDS' MAX_PROMPT_LEN='$MAX_PROMPT_LEN' EVAL_RESPONSE_STYLE='$EVAL_RESPONSE_STYLE' ANSWER_CONSTRAINT_MODE='$ANSWER_CONSTRAINT_MODE' THINKING_MODE='$THINKING_MODE' SKIP_SERVER_LAUNCH='$SKIP_SERVER_LAUNCH' PROBLEM_SHARD_INDEX='$shard_index' PROBLEM_SHARD_COUNT='$shard_count' USE_WANDB='$USE_WANDB' WANDB_GROUP='$WANDB_GROUP' RUN_ARTIFACT_GCS_PREFIX='$RUN_ARTIFACT_GCS_PREFIX' RUN_SYNC_INTERVAL_SECONDS='$RUN_SYNC_INTERVAL_SECONDS' bash scripts/run_use1d_vllm_remote_eval.sh > 'runs/$run_name/orchestrator.log' 2>&1\""
done

log "all shard launches issued"

if [ "$WAIT_FOR_COMPLETION" != "1" ]; then
  log "launch-only mode; remote shards will continue independently"
  exit 0
fi

for shard_index in "${!worker_ids[@]}"; do
  worker="${worker_ids[$shard_index]}"
  run_name="${run_names[$shard_index]}"
  summary_path="$REMOTE_ROOT/runs/$run_name/analysis/benchmark_summary.json"
  log "waiting for shard summary on worker $worker: $summary_path"
  until remote_ssh "$worker" "test -f '$summary_path'"; do
    sleep 60
  done
done

mkdir -p "$LOCAL_PULL_ROOT"
shard_run_dirs=()
for shard_index in "${!worker_ids[@]}"; do
  worker="${worker_ids[$shard_index]}"
  run_name="${run_names[$shard_index]}"
  log "pulling shard run $run_name from worker $worker"
  remote_scp_pull "$worker" "$REMOTE_ROOT/runs/$run_name" "$LOCAL_PULL_ROOT/"
  shard_run_dirs+=("$LOCAL_PULL_ROOT/$run_name")
done

mkdir -p "$MERGED_RUN_DIR"
python3 scripts/merge_vllm_aime_shards.py \
  --run-dir "$MERGED_RUN_DIR" \
  --role-name base \
  --shard-run-dirs "${shard_run_dirs[@]}"

log "merged summary: $MERGED_RUN_DIR/analysis/benchmark_summary.json"
