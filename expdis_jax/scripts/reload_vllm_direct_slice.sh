#!/usr/bin/env bash
# Reload vLLM-TPU workers from an HF model bundle in Hugging Face or GCS.
#
# Large model copies can outlive or destabilize a single long TPU SSH command.
# This script therefore installs a small detached worker-side reloader, starts it
# with nohup, and polls worker health through short SSH calls.
set -euo pipefail

GCS_MODEL_PATH="${GCS_MODEL_PATH:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-}"
HF_REVISION="${HF_REVISION:-}"
MODEL_TAR_URL="${MODEL_TAR_URL:-}"
HF_TOKEN="${HF_TOKEN:-}"
TPU_NAME="${TPU_NAME:?Must set TPU_NAME}"
ZONE="${ZONE:?Must set ZONE}"
TPU_PROJECT="${TPU_PROJECT:-${CLOUDSDK_CORE_PROJECT:-}}"
WORKERS="${WORKERS:?Must set WORKERS, e.g. '1 2 3' or '1,2,3'}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-1.7B}"
LOCAL_MODEL_DIR_BASENAME="${LOCAL_MODEL_DIR_BASENAME:-expdis_vllm_model}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-}"  # empty = the worker's default ($HOME/vllm_tpu_env on the worker)
VLLM_TPU_BUCKET_PADDING_GAP="${VLLM_TPU_BUCKET_PADDING_GAP:-}"
VLLM_TPU_MOST_MODEL_LEN="${VLLM_TPU_MOST_MODEL_LEN:-}"
VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}"
VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-}"
PARALLELISM="${PARALLELISM:-8}"
LAUNCH_RETRIES="${EXPDIS_VLLM_RELOAD_LAUNCH_RETRIES:-4}"
LAUNCH_RETRY_SLEEP="${EXPDIS_VLLM_RELOAD_LAUNCH_RETRY_SLEEP:-8}"
LAUNCH_SSH_TIMEOUT_SECONDS="${EXPDIS_VLLM_RELOAD_LAUNCH_SSH_TIMEOUT_SECONDS:-180}"
POLL_SSH_TIMEOUT_SECONDS="${EXPDIS_VLLM_RELOAD_POLL_SSH_TIMEOUT_SECONDS:-45}"
TPU_TRANSPORT_FLAG="${TPU_TRANSPORT_FLAG:-direct}"
transport_args=()
case "$TPU_TRANSPORT_FLAG" in
  ""|"none"|"NONE"|"direct"|"DIRECT")
    # Use direct TPU VM SSH. This is required when running gcloud from one TPU
    # VM to another TPU VM whose external IPs are available; IAP can resolve
    # stale backing VM names after TPU recreation/preemption.
    ;;
  *)
    transport_args+=("$TPU_TRANSPORT_FLAG")
    ;;
esac
MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-vllm}"
RELOAD_TIMEOUT_SECONDS="${RELOAD_TIMEOUT_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-10}"
MANUAL_RELOAD_MARKER="/tmp/expdis_manual_reload_${LOCAL_MODEL_DIR_BASENAME}.done"
SKIP_FAILED_WORKERS="${EXPDIS_VLLM_SKIP_FAILED_WORKERS:-0}"
MIN_HEALTHY_WORKERS="${EXPDIS_VLLM_RELOAD_MIN_HEALTHY_WORKERS:-0}"

if [ -z "$MODEL_TAR_URL" ] && [ -z "$GCS_MODEL_PATH" ] && { [ -z "$HF_REPO_ID" ] || [ -z "$HF_PATH_IN_REPO" ]; }; then
  echo "[reload-direct] ERROR: set MODEL_TAR_URL, or GCS_MODEL_PATH, or HF_REPO_ID+HF_PATH_IN_REPO" >&2
  exit 2
fi

if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
fi

if [ "${EXPDIS_VLLM_RELOAD_ALLOW_PARALLEL:-0}" != "1" ] && [ "${PARALLELISM}" -gt 1 ]; then
  # Parallel IAP SSH to TPU workers is flaky enough to strand full pipeline
  # runs. Launch reload workers serially; the detached worker-side copies still
  # proceed concurrently after launch.
  PARALLELISM=1
fi

if [ -f "$MANUAL_RELOAD_MARKER" ]; then
  echo "[reload-direct] manual external reload marker present: $MANUAL_RELOAD_MARKER"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
worker_script_b64="$(base64 < "$SCRIPT_DIR/reload_vllm_worker_detached.sh" | tr -d '\n')"
worker_list="$(printf '%s\n' "$WORKERS" | tr ',' ' ')"
tmp_dir="$(mktemp -d)"
gcloud_cmd=(gcloud)
if [ -n "$TPU_PROJECT" ]; then
  gcloud_cmd+=(--project="$TPU_PROJECT")
fi
run_with_timeout() {
  local seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$seconds" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$seconds" "$@"
  else
    python3 - "$seconds" "$@" <<'PY'
import subprocess
import sys

seconds = float(sys.argv[1])
cmd = sys.argv[2:]
try:
    raise SystemExit(subprocess.run(cmd, timeout=seconds).returncode)
except subprocess.TimeoutExpired:
    print(f"[reload-direct] command timed out after {seconds:.0f}s: {' '.join(cmd)}", file=sys.stderr)
    raise SystemExit(124)
PY
  fi
}
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

if [ "${EXPDIS_VLLM_MULTI_SERVER:-0}" = "1" ]; then
  multi_worker_script_b64="$(base64 < "$SCRIPT_DIR/vllm_multiserver_worker.sh" | tr -d '\n')"
  multi_chips="${CHIPS:-${EXPDIS_VLLM_MULTI_SERVER_CHIPS:-0 1 2 3}}"
  port_base="${PORT_BASE:-${EXPDIS_VLLM_MULTI_SERVER_PORT_BASE:-8000}}"
  failures=0
  pids=()
  pid_workers=()

  wait_multi_launch_batch() {
    local i
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "[reload-direct] multi-server launch failed worker=${pid_workers[$i]}" >&2
        failures=$((failures + 1))
      fi
    done
    pids=()
    pid_workers=()
  }

  for worker in $worker_list; do
    (
      log_file="\$HOME/vllm_multiserver_worker_${worker}.log"  # expanded on the worker
      read -r -d '' remote_cmd <<EOF || true
set -euo pipefail
printf '%s' '${multi_worker_script_b64}' | base64 -d > /tmp/vllm_multiserver_worker.sh
chmod +x /tmp/vllm_multiserver_worker.sh
rm -f ${log_file}
nohup env \\
  GCS_MODEL_PATH='${GCS_MODEL_PATH}' \\
  HF_REPO_ID='${HF_REPO_ID}' \\
  HF_REPO_TYPE='${HF_REPO_TYPE}' \\
  HF_PATH_IN_REPO='${HF_PATH_IN_REPO}' \\
  HF_REVISION='${HF_REVISION}' \\
  MODEL_TAR_URL='${MODEL_TAR_URL}' \\
  HF_TOKEN='${HF_TOKEN}' \\
  WORKER_ID='${worker}' \\
  CHIPS='${multi_chips}' \\
  PORT_BASE='${port_base}' \\
  LOCAL_MODEL_DIR_BASENAME='${LOCAL_MODEL_DIR_BASENAME}' \\
  MODEL='${SERVED_MODEL_NAME}' \\
  SERVED_MODEL_NAME='${SERVED_MODEL_NAME}' \\
  MAX_MODEL_LEN='${MAX_MODEL_LEN}' \\
  MAX_NUM_SEQS='${MAX_NUM_SEQS}' \\
  MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}' \\
  VLLM_VENV_DIR='${VLLM_VENV_DIR}' \\
  VLLM_TPU_BUCKET_PADDING_GAP='${VLLM_TPU_BUCKET_PADDING_GAP}' \\
  VLLM_TPU_MOST_MODEL_LEN='${VLLM_TPU_MOST_MODEL_LEN}' \\
  VLLM_XLA_CACHE_PATH='${VLLM_XLA_CACHE_PATH}' \\
  DISABLE_LOG_REQUESTS='${DISABLE_LOG_REQUESTS}' \\
  VLLM_USE_V1='${VLLM_USE_V1}' \\
  VLLM_ENABLE_CHUNKED_PREFILL='${VLLM_ENABLE_CHUNKED_PREFILL}' \\
  VLLM_V1_USE_PREFILL_DECODE_ATTENTION='${VLLM_V1_USE_PREFILL_DECODE_ATTENTION}' \\
  VLLM_XLA_CACHE_TAG='${VLLM_XLA_CACHE_TAG}' \\
  MODEL_IMPL_TYPE='${MODEL_IMPL_TYPE}' \\
  LOG_FILE="${log_file}" \\
  HEALTH_TIMEOUT_SECONDS='${RELOAD_TIMEOUT_SECONDS}' \\
  bash /tmp/vllm_multiserver_worker.sh > ${log_file}.nohup 2>&1 &
echo "launched multi-server worker=${worker}"
EOF
      for launch_attempt in $(seq 1 "$LAUNCH_RETRIES"); do
        if run_with_timeout "$LAUNCH_SSH_TIMEOUT_SECONDS" "${gcloud_cmd[@]}" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
          --worker="$worker" \
          --zone="$ZONE" \
          "${transport_args[@]}" \
          --command="$remote_cmd" >"$tmp_dir/launch_${worker}.log" 2>&1; then
          exit 0
        fi
        echo "[reload-direct] multi worker=${worker} launch attempt ${launch_attempt}/${LAUNCH_RETRIES} failed" >>"$tmp_dir/launch_${worker}.log"
        sleep "$LAUNCH_RETRY_SLEEP"
      done
      exit 1
    ) &
    pids+=("$!")
    pid_workers+=("$worker")
    if [ "${#pids[@]}" -ge "$PARALLELISM" ]; then
      wait_multi_launch_batch
    fi
  done

  wait_multi_launch_batch
  if [ "$failures" -ne 0 ]; then
    for worker in $worker_list; do
      [ -f "$tmp_dir/launch_${worker}.log" ] && tail -50 "$tmp_dir/launch_${worker}.log" || true
    done
    echo "[reload-direct] ERROR: one or more multi-server launches failed" >&2
    exit 1
  fi

  deadline=$(( $(date +%s) + RELOAD_TIMEOUT_SECONDS ))
  pending="$worker_list"
  while [ -n "$pending" ] && [ "$(date +%s)" -lt "$deadline" ]; do
    next_pending=""
    for worker in $pending; do
      check_cmd="
        ok=1
        for chip in ${multi_chips}; do
          port=\$(( ${port_base} + chip ))
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
      status="$(
        run_with_timeout "$POLL_SSH_TIMEOUT_SECONDS" "${gcloud_cmd[@]}" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
          --worker="$worker" \
          --zone="$ZONE" \
          "${transport_args[@]}" \
          --command="$check_cmd" 2>&1 || true
      )"
      if printf '%s\n' "$status" | grep -q '^HEALTHY'; then
        echo "[reload-direct] multi worker=${worker} HEALTHY"
      elif printf '%s\n' "$status" | grep -q '^FAILED'; then
        echo "[reload-direct] multi worker=${worker} FAILED" >&2
        printf '%s\n' "$status" >&2
        failures=$((failures + 1))
      else
        next_pending="${next_pending} ${worker}"
      fi
    done
    pending="$(printf '%s\n' "$next_pending" | xargs || true)"
    [ -z "$pending" ] && break
    sleep "$POLL_SECONDS"
  done

  if [ -n "$pending" ]; then
    echo "[reload-direct] ERROR: timed out waiting for multi-server workers:$pending" >&2
    failures=$((failures + 1))
  fi
  if [ "$failures" -ne 0 ]; then
    echo "[reload-direct] ERROR: one or more multi-server workers failed" >&2
    exit 1
  fi
  echo "[reload-direct] all requested multi-server workers healthy"
  exit 0
fi

failures=0
pids=()
pid_workers=()
failed_workers=""
healthy_workers=""

wait_launch_batch() {
  local i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[reload-direct] launch failed worker=${pid_workers[$i]}" >&2
      failures=$((failures + 1))
    fi
  done
  pids=()
  pid_workers=()
}

for worker in $worker_list; do
  (
    log_file="\$HOME/vllm_reload_worker_${worker}.log"  # expanded on the worker
    read -r -d '' remote_cmd <<EOF || true
set -euo pipefail
# Free stale per-checkpoint model exports before writing the detached worker
# script. Avoid pkill patterns here: this SSH command text itself contains the
# worker script name and can otherwise kill its own launch.
pkill -9 -f "[v]llm serve" 2>/dev/null || true
tmux kill-session -t vllm_server 2>/dev/null || true
rm -rf /tmp/expdis_external_*_w${worker} /tmp/expdis_vllm_reload_* 2>/dev/null || true
printf '%s' '${worker_script_b64}' | base64 -d > /tmp/reload_vllm_worker_detached.sh
chmod +x /tmp/reload_vllm_worker_detached.sh
rm -f ${log_file}
nohup env \\
  GCS_MODEL_PATH='${GCS_MODEL_PATH}' \\
  HF_REPO_ID='${HF_REPO_ID}' \\
  HF_REPO_TYPE='${HF_REPO_TYPE}' \\
  HF_PATH_IN_REPO='${HF_PATH_IN_REPO}' \\
  HF_REVISION='${HF_REVISION}' \\
  MODEL_TAR_URL='${MODEL_TAR_URL}' \\
  HF_TOKEN='${HF_TOKEN}' \\
  WORKER_ID='${worker}' \\
  LOCAL_MODEL_DIR_BASENAME='${LOCAL_MODEL_DIR_BASENAME}' \\
  SERVED_MODEL_NAME='${SERVED_MODEL_NAME}' \\
  PORT='${PORT}' \\
  TP_SIZE='${TP_SIZE}' \\
  MAX_MODEL_LEN='${MAX_MODEL_LEN}' \\
  MAX_NUM_SEQS='${MAX_NUM_SEQS}' \\
  MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}' \\
  VLLM_VENV_DIR='${VLLM_VENV_DIR}' \\
  VLLM_TPU_BUCKET_PADDING_GAP='${VLLM_TPU_BUCKET_PADDING_GAP}' \\
  VLLM_TPU_MOST_MODEL_LEN='${VLLM_TPU_MOST_MODEL_LEN}' \\
  VLLM_XLA_CACHE_PATH='${VLLM_XLA_CACHE_PATH}' \\
  DISABLE_LOG_REQUESTS='${DISABLE_LOG_REQUESTS}' \\
  VLLM_USE_V1='${VLLM_USE_V1}' \\
  VLLM_ENABLE_CHUNKED_PREFILL='${VLLM_ENABLE_CHUNKED_PREFILL}' \\
  VLLM_V1_USE_PREFILL_DECODE_ATTENTION='${VLLM_V1_USE_PREFILL_DECODE_ATTENTION}' \\
  VLLM_XLA_CACHE_TAG='${VLLM_XLA_CACHE_TAG}' \\
  MODEL_IMPL_TYPE='${MODEL_IMPL_TYPE}' \\
  LOG_FILE="${log_file}" \\
  bash /tmp/reload_vllm_worker_detached.sh > ${log_file}.nohup 2>&1 &
echo "launched worker=${worker}"
EOF
    for launch_attempt in $(seq 1 "$LAUNCH_RETRIES"); do
      if run_with_timeout "$LAUNCH_SSH_TIMEOUT_SECONDS" "${gcloud_cmd[@]}" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
        --worker="$worker" \
        --zone="$ZONE" \
        "${transport_args[@]}" \
        --command="$remote_cmd" >"$tmp_dir/launch_${worker}.log" 2>&1; then
        exit 0
      fi
      echo "[reload-direct] worker=${worker} launch attempt ${launch_attempt}/${LAUNCH_RETRIES} failed" >>"$tmp_dir/launch_${worker}.log"
      sleep "$LAUNCH_RETRY_SLEEP"
    done
    exit 1
  ) &
  pids+=("$!")
  pid_workers+=("$worker")
  if [ "${#pids[@]}" -ge "$PARALLELISM" ]; then
    wait_launch_batch
  fi
done

wait_launch_batch
if [ "$failures" -ne 0 ]; then
  for worker in $worker_list; do
    [ -f "$tmp_dir/launch_${worker}.log" ] && tail -50 "$tmp_dir/launch_${worker}.log" || true
  done
  echo "[reload-direct] ERROR: one or more worker reload launches failed" >&2
  exit 1
fi

deadline=$(( $(date +%s) + RELOAD_TIMEOUT_SECONDS ))
pending="$worker_list"
while [ -n "$pending" ] && [ "$(date +%s)" -lt "$deadline" ]; do
  next_pending=""
  for worker in $pending; do
    check_cmd="
      if curl -fsS --connect-timeout 3 http://127.0.0.1:${PORT}/health >/dev/null; then
        echo HEALTHY
      elif grep -q 'ERROR:' \$HOME/vllm_reload_worker_${worker}.log 2>/dev/null; then
        echo FAILED
        tail -80 \$HOME/vllm_reload_worker_${worker}.log 2>/dev/null || true
      else
        echo PENDING
      fi
    "
    status="$(
      run_with_timeout "$POLL_SSH_TIMEOUT_SECONDS" "${gcloud_cmd[@]}" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
        --worker="$worker" \
        --zone="$ZONE" \
        "${transport_args[@]}" \
        --command="$check_cmd" 2>&1 || true
    )"
    if printf '%s\n' "$status" | grep -q '^HEALTHY'; then
      echo "[reload-direct] worker=${worker} HEALTHY"
      healthy_workers="${healthy_workers} ${worker}"
    elif printf '%s\n' "$status" | grep -q '^FAILED'; then
      echo "[reload-direct] worker=${worker} FAILED" >&2
      printf '%s\n' "$status" >&2
      if [ "$SKIP_FAILED_WORKERS" = "1" ]; then
        failed_workers="${failed_workers} ${worker}"
      else
        failures=$((failures + 1))
      fi
    else
      next_pending="${next_pending} ${worker}"
    fi
  done
  if [ "$SKIP_FAILED_WORKERS" = "1" ] && [ "$MIN_HEALTHY_WORKERS" != "0" ]; then
    healthy_workers_compact="$(printf '%s\n' "$healthy_workers" | xargs || true)"
    healthy_count="$(printf '%s\n' "$healthy_workers_compact" | wc -w | tr -d ' ')"
    if [ "$healthy_count" -ge "$MIN_HEALTHY_WORKERS" ]; then
      failed_workers_compact="$(printf '%s\n' "$failed_workers" | xargs || true)"
      pending_compact="$(printf '%s\n' "$next_pending" | xargs || true)"
      if [ -n "$failed_workers_compact$pending_compact" ]; then
        echo "[reload-direct] WARN continuing with healthy workers=[$healthy_workers_compact] failed=[$failed_workers_compact] pending=[$pending_compact]"
      fi
      echo "[reload-direct] healthy threshold met count=$healthy_count/$MIN_HEALTHY_WORKERS"
      exit 0
    fi
  fi
  pending="$(printf '%s\n' "$next_pending" | xargs || true)"
  [ -z "$pending" ] && break
  sleep "$POLL_SECONDS"
done

if [ -n "$pending" ]; then
  echo "[reload-direct] ERROR: timed out waiting for workers:$pending" >&2
  if [ "$SKIP_FAILED_WORKERS" = "1" ]; then
    failed_workers="${failed_workers} ${pending}"
  else
    failures=$((failures + 1))
  fi
fi

if [ "$SKIP_FAILED_WORKERS" = "1" ]; then
  requested_count="$(printf '%s\n' "$worker_list" | wc -w | tr -d ' ')"
  if [ "$MIN_HEALTHY_WORKERS" = "0" ]; then
    MIN_HEALTHY_WORKERS="$requested_count"
  fi
  healthy_workers="$(printf '%s\n' "$healthy_workers" | xargs || true)"
  healthy_count="$(printf '%s\n' "$healthy_workers" | wc -w | tr -d ' ')"
  if [ "$healthy_count" -ge "$MIN_HEALTHY_WORKERS" ]; then
    if [ -n "$failed_workers" ]; then
      echo "[reload-direct] WARN continuing with healthy workers=[$healthy_workers] failed=[$(printf '%s\n' "$failed_workers" | xargs || true)]"
    fi
    echo "[reload-direct] healthy threshold met count=$healthy_count/$MIN_HEALTHY_WORKERS"
    exit 0
  fi
  echo "[reload-direct] ERROR healthy threshold not met count=$healthy_count/$MIN_HEALTHY_WORKERS healthy=[$healthy_workers] failed=[$(printf '%s\n' "$failed_workers" | xargs || true)]" >&2
  exit 1
fi

if [ "$failures" -ne 0 ]; then
  echo "[reload-direct] ERROR: one or more workers failed" >&2
  exit 1
fi

echo "[reload-direct] all requested workers healthy"
