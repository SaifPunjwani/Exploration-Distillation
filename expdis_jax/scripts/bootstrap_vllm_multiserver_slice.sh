#!/usr/bin/env bash
# Launch chip-local TP=1 vLLM servers across a TPU slice.
set -euo pipefail

TPU_NAME="${TPU_NAME:?Must set TPU_NAME}"
TPU_PROJECT="${TPU_PROJECT:-${CLOUDSDK_CORE_PROJECT:-}}"
ZONE="${ZONE:?Must set ZONE}"
WORKERS="${WORKERS:?Must set WORKERS}"
CHIPS="${CHIPS:-0 1 2 3}"
PORT_BASE="${PORT_BASE:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-22528}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}"
VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
LAUNCH_RETRIES="${EXPDIS_VLLM_BOOTSTRAP_RETRIES:-4}"
LAUNCH_RETRY_SLEEP="${EXPDIS_VLLM_BOOTSTRAP_RETRY_SLEEP:-8}"
SKIP_FAILED_WORKERS="${EXPDIS_VLLM_SKIP_FAILED_WORKERS:-0}"
SSH_CONNECT_TIMEOUT_SECONDS="${EXPDIS_VLLM_SSH_CONNECT_TIMEOUT_SECONDS:-30}"
SSH_SERVER_ALIVE_INTERVAL="${EXPDIS_VLLM_SSH_SERVER_ALIVE_INTERVAL:-10}"
SSH_SERVER_ALIVE_COUNT_MAX="${EXPDIS_VLLM_SSH_SERVER_ALIVE_COUNT_MAX:-3}"
POLL_SECONDS="${POLL_SECONDS:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="$SCRIPT_DIR/vllm_multiserver_worker.sh"
if [ ! -f "$WORKER_SCRIPT" ]; then
  echo "[vllm-multi-slice] ERROR: missing $WORKER_SCRIPT" >&2
  exit 1
fi

gcloud_cmd=(gcloud)
if [ -n "$TPU_PROJECT" ]; then
  gcloud_cmd+=(--project="$TPU_PROJECT")
fi

for worker in $WORKERS; do
  echo "[vllm-multi-slice] syncing worker script worker=$worker"
  launched=0
  for attempt in $(seq 1 "$LAUNCH_RETRIES"); do
    if "${gcloud_cmd[@]}" alpha compute tpus tpu-vm scp "$WORKER_SCRIPT" \
      "$TPU_NAME:~/vllm_multiserver_worker.sh" \
      --worker="$worker" --zone="$ZONE" --tunnel-through-iap \
      --scp-flag="-o ConnectTimeout=${SSH_CONNECT_TIMEOUT_SECONDS}" \
      --scp-flag="-o ServerAliveInterval=${SSH_SERVER_ALIVE_INTERVAL}" \
      --scp-flag="-o ServerAliveCountMax=${SSH_SERVER_ALIVE_COUNT_MAX}" >/dev/null; then
      launched=1
      break
    fi
    echo "[vllm-multi-slice] WARN scp failed worker=$worker attempt=$attempt"
    sleep "$LAUNCH_RETRY_SLEEP"
  done
  if [ "$launched" != "1" ]; then
    echo "[vllm-multi-slice] ERROR scp failed worker=$worker" >&2
    exit 1
  fi

  echo "[vllm-multi-slice] launching worker=$worker chips=[$CHIPS]"
  launched=0
  for attempt in $(seq 1 "$LAUNCH_RETRIES"); do
    if "${gcloud_cmd[@]}" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --worker="$worker" --zone="$ZONE" --tunnel-through-iap \
      --ssh-flag="-o ConnectTimeout=${SSH_CONNECT_TIMEOUT_SECONDS}" \
      --ssh-flag="-o ServerAliveInterval=${SSH_SERVER_ALIVE_INTERVAL}" \
      --ssh-flag="-o ServerAliveCountMax=${SSH_SERVER_ALIVE_COUNT_MAX}" \
      --command="
        chmod +x ~/vllm_multiserver_worker.sh
        nohup env \
          WORKER_ID='$worker' \
          CHIPS='$CHIPS' \
          PORT_BASE='$PORT_BASE' \
          MODEL='$MODEL' \
          SERVED_MODEL_NAME='$SERVED_MODEL_NAME' \
          MAX_MODEL_LEN='$MAX_MODEL_LEN' \
          MAX_NUM_SEQS='$MAX_NUM_SEQS' \
          MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' \
          GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' \
          VLLM_ENABLE_CHUNKED_PREFILL='$VLLM_ENABLE_CHUNKED_PREFILL' \
          VLLM_V1_USE_PREFILL_DECODE_ATTENTION='$VLLM_V1_USE_PREFILL_DECODE_ATTENTION' \
          VLLM_XLA_CACHE_TAG='$VLLM_XLA_CACHE_TAG' \
          HEALTH_TIMEOUT_SECONDS='$HEALTH_TIMEOUT_SECONDS' \
          POLL_SECONDS='$POLL_SECONDS' \
          LOG_FILE=\"\$HOME/vllm_multiserver_worker_${worker}.log\" \
          bash ~/vllm_multiserver_worker.sh > \"\$HOME/vllm_multiserver_worker_${worker}.nohup\" 2>&1 &
        echo launched
      " >/dev/null; then
      launched=1
      break
    fi
    echo "[vllm-multi-slice] WARN launch failed worker=$worker attempt=$attempt"
    sleep "$LAUNCH_RETRY_SLEEP"
  done
  if [ "$launched" != "1" ]; then
    echo "[vllm-multi-slice] ERROR launch failed worker=$worker" >&2
    exit 1
  fi
done

deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
pending="$WORKERS"
failed_workers=""
while [ -n "$pending" ] && [ "$(date +%s)" -lt "$deadline" ]; do
  next_pending=""
  for worker in $pending; do
    check_cmd="
      ok=1
      for chip in $CHIPS; do
        port=\$(( $PORT_BASE + chip ))
        if ! curl -fsS --connect-timeout 3 http://127.0.0.1:\${port}/health >/dev/null; then
          ok=0
        fi
      done
      if [ \"\$ok\" = 1 ]; then
        echo HEALTHY
      elif grep -q 'ERROR:' \$HOME/vllm_multiserver_worker_${worker}.log 2>/dev/null; then
        echo FAILED
        tail -120 \$HOME/vllm_multiserver_worker_${worker}.log 2>/dev/null || true
      else
        echo PENDING
      fi
    "
    status="$("${gcloud_cmd[@]}" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --worker="$worker" --zone="$ZONE" --tunnel-through-iap \
      --ssh-flag="-o ConnectTimeout=${SSH_CONNECT_TIMEOUT_SECONDS}" \
      --ssh-flag="-o ServerAliveInterval=${SSH_SERVER_ALIVE_INTERVAL}" \
      --ssh-flag="-o ServerAliveCountMax=${SSH_SERVER_ALIVE_COUNT_MAX}" \
      --command="$check_cmd" 2>&1 || true)"
    if printf '%s\n' "$status" | grep -q '^HEALTHY'; then
      echo "[vllm-multi-slice] worker=$worker HEALTHY"
    elif printf '%s\n' "$status" | grep -q '^FAILED'; then
      echo "[vllm-multi-slice] worker=$worker FAILED" >&2
      printf '%s\n' "$status" >&2
      if [ "$SKIP_FAILED_WORKERS" = "1" ]; then
        failed_workers="$failed_workers $worker"
      else
        exit 1
      fi
    else
      next_pending="$next_pending $worker"
    fi
  done
  pending="$(printf '%s\n' "$next_pending" | xargs || true)"
  [ -z "$pending" ] && break
  echo "[vllm-multi-slice] waiting workers: $pending"
  sleep "$POLL_SECONDS"
done

if [ -n "$pending" ]; then
  echo "[vllm-multi-slice] ERROR: timed out waiting workers: $pending" >&2
  if [ "$SKIP_FAILED_WORKERS" = "1" ]; then
    failed_workers="$failed_workers $pending"
  else
    exit 1
  fi
fi

if [ -n "$failed_workers" ]; then
  echo "[vllm-multi-slice] WARN: continuing despite failed workers:$(printf '%s\n' "$failed_workers" | xargs || true)" >&2
fi

echo "[vllm-multi-slice] all workers healthy workers=[$WORKERS] chips=[$CHIPS]"
