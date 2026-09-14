#!/usr/bin/env bash
# Restore cached vLLM-TPU envs and launch vLLM servers on a set of TPU workers.
set -euo pipefail

TPU_NAME="${TPU_NAME:?set TPU_NAME to the serving TPU VM (slice) name}"
TPU_PROJECT="${TPU_PROJECT:-${CLOUDSDK_CORE_PROJECT:-}}"
ZONE="${ZONE:?set ZONE to the TPU zone, e.g. europe-west4-b}"
WORKERS="${WORKERS:-4 5 6 7 8 9 10 11 12 13 14 15}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_TPU_BUCKET_PADDING_GAP="${VLLM_TPU_BUCKET_PADDING_GAP:-}"
VLLM_TPU_MOST_MODEL_LEN="${VLLM_TPU_MOST_MODEL_LEN:-}"
VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}"
VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-tp${TP}_pda${VLLM_V1_USE_PREFILL_DECODE_ATTENTION}_mlen${MAX_MODEL_LEN}_mbt${MAX_NUM_BATCHED_TOKENS}}"
MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-auto}"
EXPDIS_VLLM_SUPERVISE="${EXPDIS_VLLM_SUPERVISE:-1}"
EXPDIS_VLLM_RESTART_SLEEP_SECONDS="${EXPDIS_VLLM_RESTART_SLEEP_SECONDS:-5}"
VLLM_ENV_CACHE_GCS_URI="${VLLM_ENV_CACHE_GCS_URI:-}"
EXPDIS_ALLOW_GCS_ARTIFACTS="${EXPDIS_ALLOW_GCS_ARTIFACTS:-0}"
LAUNCH_RETRIES="${EXPDIS_VLLM_BOOTSTRAP_RETRIES:-4}"
LAUNCH_RETRY_SLEEP="${EXPDIS_VLLM_BOOTSTRAP_RETRY_SLEEP:-8}"
SKIP_FAILED_WORKERS="${EXPDIS_VLLM_SKIP_FAILED_WORKERS:-0}"
SSH_CONNECT_TIMEOUT_SECONDS="${EXPDIS_VLLM_SSH_CONNECT_TIMEOUT_SECONDS:-30}"
SSH_SERVER_ALIVE_INTERVAL="${EXPDIS_VLLM_SSH_SERVER_ALIVE_INTERVAL:-10}"
SSH_SERVER_ALIVE_COUNT_MAX="${EXPDIS_VLLM_SSH_SERVER_ALIVE_COUNT_MAX:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_LOCAL="$SCRIPT_DIR/bootstrap_vllm_worker.sh"
gcloud_cmd=(gcloud)
if [ -n "$TPU_PROJECT" ]; then
  gcloud_cmd+=(--project="$TPU_PROJECT")
fi
ssh_flags=(
  --ssh-flag="-o ConnectTimeout=${SSH_CONNECT_TIMEOUT_SECONDS}"
  --ssh-flag="-o ServerAliveInterval=${SSH_SERVER_ALIVE_INTERVAL}"
  --ssh-flag="-o ServerAliveCountMax=${SSH_SERVER_ALIVE_COUNT_MAX}"
)
scp_flags=(
  --scp-flag="-o ConnectTimeout=${SSH_CONNECT_TIMEOUT_SECONDS}"
)

if [ ! -f "$BOOTSTRAP_LOCAL" ]; then
  echo "ERROR: missing $BOOTSTRAP_LOCAL" >&2
  exit 1
fi

gcloud_worker_spec() {
  python3 - "$1" <<'PY'
import sys

workers = [int(x) for x in sys.argv[1].split()]
if not workers:
    raise SystemExit("empty worker list")
workers = sorted(workers)
ranges = []
start = prev = workers[0]
for worker in workers[1:]:
    if worker == prev + 1:
        prev = worker
        continue
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    start = prev = worker
ranges.append(f"{start}-{prev}" if start != prev else str(start))
print(",".join(ranges))
PY
}

if [ "${EXPDIS_VLLM_BATCHED_SLICE_BOOTSTRAP:-0}" != "0" ]; then
  worker_spec="$(gcloud_worker_spec "$WORKERS")"
  echo "[vllm-slice] batched bootstrap workers=[$WORKERS] worker_spec=$worker_spec batch_size=${EXPDIS_VLLM_BATCH_SIZE:-4}"
  "${gcloud_cmd[@]}" alpha compute tpus tpu-vm scp \
    "$BOOTSTRAP_LOCAL" \
    "$TPU_NAME:~/bootstrap_vllm_worker.sh" \
    --worker="$worker_spec" \
    --batch-size="${EXPDIS_VLLM_BATCH_SIZE:-4}" \
    --zone="$ZONE" \
    --tunnel-through-iap \
    "${scp_flags[@]}"
  "${gcloud_cmd[@]}" alpha compute tpus tpu-vm ssh \
    "$TPU_NAME" \
    --worker="$worker_spec" \
    --batch-size="${EXPDIS_VLLM_BATCH_SIZE:-4}" \
    --zone="$ZONE" \
    --tunnel-through-iap \
    "${ssh_flags[@]}" \
    --command="
      chmod +x ~/bootstrap_vllm_worker.sh
      MODEL='$MODEL' \
      PORT='$PORT' \
      TP='$TP' \
      MAX_MODEL_LEN='$MAX_MODEL_LEN' \
      MAX_NUM_SEQS='$MAX_NUM_SEQS' \
      MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' \
      GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' \
      VLLM_TPU_BUCKET_PADDING_GAP='$VLLM_TPU_BUCKET_PADDING_GAP' \
      VLLM_TPU_MOST_MODEL_LEN='$VLLM_TPU_MOST_MODEL_LEN' \
      VLLM_XLA_CACHE_PATH='$VLLM_XLA_CACHE_PATH' \
      DISABLE_LOG_REQUESTS='$DISABLE_LOG_REQUESTS' \
      VLLM_USE_V1='$VLLM_USE_V1' \
      VLLM_ENABLE_CHUNKED_PREFILL='$VLLM_ENABLE_CHUNKED_PREFILL' \
      VLLM_V1_USE_PREFILL_DECODE_ATTENTION='$VLLM_V1_USE_PREFILL_DECODE_ATTENTION' \
      VLLM_XLA_CACHE_TAG='$VLLM_XLA_CACHE_TAG' \
      MODEL_IMPL_TYPE='$MODEL_IMPL_TYPE' \
      EXPDIS_VLLM_SUPERVISE='$EXPDIS_VLLM_SUPERVISE' \
      EXPDIS_VLLM_RESTART_SLEEP_SECONDS='$EXPDIS_VLLM_RESTART_SLEEP_SECONDS' \
      VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' \
      bash ~/bootstrap_vllm_worker.sh
    "
  echo "[vllm-slice] batched bootstrap command completed workers=[$WORKERS]"
  exit 0
fi

launch_one_worker() {
  local w="$1"
  worker_failed=0
  echo "[vllm-slice] syncing bootstrap script to worker $w"
  for attempt in $(seq 1 "$LAUNCH_RETRIES"); do
    if "${gcloud_cmd[@]}" alpha compute tpus tpu-vm scp \
      "$BOOTSTRAP_LOCAL" \
      "$TPU_NAME:~/bootstrap_vllm_worker.sh" \
      --worker="$w" \
      --zone="$ZONE" \
      --tunnel-through-iap \
      "${scp_flags[@]}" >/dev/null; then
      break
    fi
    if [ "$attempt" -eq "$LAUNCH_RETRIES" ]; then
      if [ "$SKIP_FAILED_WORKERS" != "0" ]; then
        echo "[vllm-slice] WARN scp failed worker=$w after $LAUNCH_RETRIES attempts; skipping"
        return 20
      fi
      echo "[vllm-slice] ERROR scp failed worker=$w after $LAUNCH_RETRIES attempts" >&2
      return 1
    fi
    echo "[vllm-slice] WARN scp failed worker=$w attempt=$attempt; retrying"
    sleep "$LAUNCH_RETRY_SLEEP"
  done

  echo "[vllm-slice] launching worker $w"
  for attempt in $(seq 1 "$LAUNCH_RETRIES"); do
    if "${gcloud_cmd[@]}" alpha compute tpus tpu-vm ssh \
      "$TPU_NAME" \
      --worker="$w" \
      --zone="$ZONE" \
      --tunnel-through-iap \
      "${ssh_flags[@]}" \
      --command="
        chmod +x ~/bootstrap_vllm_worker.sh
        MODEL='$MODEL' \
        PORT='$PORT' \
        TP='$TP' \
        MAX_MODEL_LEN='$MAX_MODEL_LEN' \
        MAX_NUM_SEQS='$MAX_NUM_SEQS' \
        MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' \
        GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' \
        VLLM_TPU_BUCKET_PADDING_GAP='$VLLM_TPU_BUCKET_PADDING_GAP' \
        VLLM_TPU_MOST_MODEL_LEN='$VLLM_TPU_MOST_MODEL_LEN' \
        VLLM_XLA_CACHE_PATH='$VLLM_XLA_CACHE_PATH' \
        DISABLE_LOG_REQUESTS='$DISABLE_LOG_REQUESTS' \
        VLLM_USE_V1='$VLLM_USE_V1' \
        VLLM_ENABLE_CHUNKED_PREFILL='$VLLM_ENABLE_CHUNKED_PREFILL' \
        VLLM_V1_USE_PREFILL_DECODE_ATTENTION='$VLLM_V1_USE_PREFILL_DECODE_ATTENTION' \
        VLLM_XLA_CACHE_TAG='$VLLM_XLA_CACHE_TAG' \
        MODEL_IMPL_TYPE='$MODEL_IMPL_TYPE' \
        EXPDIS_VLLM_SUPERVISE='$EXPDIS_VLLM_SUPERVISE' \
        EXPDIS_VLLM_RESTART_SLEEP_SECONDS='$EXPDIS_VLLM_RESTART_SLEEP_SECONDS' \
        VLLM_ENV_CACHE_GCS_URI='$VLLM_ENV_CACHE_GCS_URI' \
        bash ~/bootstrap_vllm_worker.sh
      " >/dev/null; then
      break
    fi
    if [ "$attempt" -eq "$LAUNCH_RETRIES" ]; then
      if [ "$SKIP_FAILED_WORKERS" != "0" ]; then
        echo "[vllm-slice] WARN launch failed worker=$w after $LAUNCH_RETRIES attempts; skipping"
        return 20
      fi
      echo "[vllm-slice] ERROR launch failed worker=$w after $LAUNCH_RETRIES attempts" >&2
      return 1
    fi
    echo "[vllm-slice] WARN launch failed worker=$w attempt=$attempt; retrying"
    sleep "$LAUNCH_RETRY_SLEEP"
  done
  echo "[vllm-slice] worker $w launched"
}

launched_workers=""
if [ "${EXPDIS_VLLM_PARALLEL_LAUNCH:-1}" != "0" ]; then
  tmp_dir="$(mktemp -d /tmp/expdis_vllm_launch.XXXXXX)"
  pids=""
  for w in $WORKERS; do
    (
      if launch_one_worker "$w" >"$tmp_dir/worker_${w}.log" 2>&1; then
        echo "$w" >"$tmp_dir/worker_${w}.ok"
      else
        code="$?"
        echo "$code" >"$tmp_dir/worker_${w}.fail"
        exit "$code"
      fi
    ) &
    pids="$pids $!"
  done
  fail=0
  for pid in $pids; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  for w in $WORKERS; do
    cat "$tmp_dir/worker_${w}.log" 2>/dev/null || true
    if [ -f "$tmp_dir/worker_${w}.ok" ]; then
      launched_workers="$launched_workers $w"
    elif [ "${SKIP_FAILED_WORKERS:-0}" = "0" ]; then
      fail=1
    fi
  done
  rm -rf "$tmp_dir"
  if [ "$fail" != "0" ] && [ "${SKIP_FAILED_WORKERS:-0}" = "0" ]; then
    echo "[vllm-slice] ERROR one or more workers failed launch" >&2
    exit 1
  fi
else
  for w in $WORKERS; do
    if launch_one_worker "$w"; then
      launched_workers="$launched_workers $w"
    elif [ "${SKIP_FAILED_WORKERS:-0}" = "0" ]; then
      exit 1
    fi
  done
fi

launched_workers="$(printf '%s\n' "$launched_workers" | xargs || true)"
if [ -z "$launched_workers" ]; then
  echo "[vllm-slice] ERROR no workers launched" >&2
  exit 1
fi

echo "[vllm-slice] launched workers: $launched_workers"
