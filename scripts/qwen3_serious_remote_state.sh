#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
. scripts/tpu_target_lib.sh

MODE="${1:-check}"
TARGET="${TARGET:-all}"

TRAIN_HOST="${TRAIN_HOST:?set TRAIN_HOST to the ssh host alias of the training TPU VM}"
INFER_HOST="${INFER_HOST:?set INFER_HOST to the ssh host alias of the inference (vLLM) TPU VM}"
INFER_TPU_NAME="${INFER_TPU_NAME:?set INFER_TPU_NAME to the inference (vLLM) TPU VM (slice) name}"
INFER_ZONE="${INFER_ZONE:-europe-west4-b}"
INFER_WORKERS="${INFER_WORKERS:-all}"
INFER_TRANSPORT_FLAG="$(tmx_tpu_transport_flag)"

show_usage() {
  cat <<'EOF'
Usage:
  bash scripts/qwen3_serious_remote_state.sh check
  CONFIRM_REMOTE_CLEANUP=1 bash scripts/qwen3_serious_remote_state.sh clean

Environment:
  TARGET=train|infer|all    default: all
  TRAIN_HOST=...            default: (required)
  INFER_HOST=...            default: (required)
  INFER_TPU_NAME=...        default: (required)
  INFER_ZONE=...            default: europe-west4-b
  INFER_WORKERS=all|list    default: all

Notes:
  - `check` is read-only.
  - `clean` is destructive on the dedicated Qwen3 serious-run hosts:
    - stops tmux sessions
    - stops local Ray/vLLM processes
    - releases TPU device holders on ew4a
    - removes stale train-host `/tmp` XLA/HF caches
EOF
}

run_ssh() {
  local host="$1"
  shift
  ssh "$host" "$@"
}

infer_worker_count() {
  local describe_json
  describe_json="$(tmx_gcloud alpha compute tpus tpu-vm describe "$INFER_TPU_NAME" --zone "$INFER_ZONE" --format=json)"
  python3 - "$describe_json" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
for key in ("networkEndpoints", "workerNetworkEndpoints"):
    value = payload.get(key)
    if isinstance(value, list) and value:
        print(len(value))
        break
else:
    print(1)
PY
}

infer_worker_list() {
  if [ "$INFER_WORKERS" = "all" ]; then
    seq 0 "$(( $(infer_worker_count) - 1 ))"
  else
    printf '%s\n' "$INFER_WORKERS" | tr ', ' '\n\n' | sed '/^$/d'
  fi
}

infer_worker_ssh() {
  local worker="$1"
  local command="$2"
  tmx_gcloud alpha compute tpus tpu-vm ssh "$INFER_TPU_NAME" \
    --worker="$worker" \
    --zone="$INFER_ZONE" \
    "$INFER_TRANSPORT_FLAG" \
    --command="$command"
}

remote_process_cleanup() {
  local host="$1"
  local patterns_csv="$2"
  local encoded
  encoded="$(printf '%s' "$patterns_csv" | base64 | tr -d '\n')"
  run_ssh "$host" "python3 - '$encoded' <<'PY'
import base64
import os
import signal
import sys

uid = os.getuid()
me = os.getpid()
patterns = [p for p in base64.b64decode(sys.argv[1]).decode('utf-8').splitlines() if p.strip()]
killed = []

for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    pid_i = int(pid)
    if pid_i == me:
        continue
    try:
        st = os.stat(f'/proc/{pid}')
        if st.st_uid != uid:
            continue
        with open(f'/proc/{pid}/cmdline', 'rb') as handle:
            cmd = handle.read().replace(b'\\x00', b' ').decode('utf-8', errors='ignore')
    except Exception:
        continue
    if not cmd:
        continue
    if any(p in cmd for p in patterns):
        try:
            os.kill(pid_i, signal.SIGKILL)
            killed.append((pid_i, cmd))
        except ProcessLookupError:
            pass
        except Exception:
            pass

for pid_i, cmd in killed:
    print(f'killed {pid_i}: {cmd}')
PY"
}

check_host() {
  local label="$1"
  local host="$2"
  echo "=== $label: $host ==="
  run_ssh "$host" "
    echo '[tmux]';
    tmux ls 2>/dev/null || true;
    echo '---';
    echo '[proc]';
    ps aux | grep -E 'main.py|vllm|ray|accelerate' | grep -v grep | head -40 || true;
    echo '---';
    echo '[disk]';
    df -h / | tail -1;
  "
}

check_infer_slice() {
  echo "=== infer slice: ${INFER_TPU_NAME} (${INFER_ZONE}) workers=${INFER_WORKERS} ==="
  local worker
  while IFS= read -r worker; do
    [ -n "$worker" ] || continue
    echo "--- infer worker $worker ---"
    infer_worker_ssh "$worker" "
      echo '[tmux]';
      tmux ls 2>/dev/null || true;
      echo '---';
      echo '[proc]';
      ps aux | grep -E 'main.py|vllm|ray|accelerate' | grep -v grep | head -40 || true;
      echo '---';
      echo '[disk]';
      df -h / | tail -1;
    "
  done < <(infer_worker_list)
}

clean_train_host() {
  local host="$1"
  echo "=== clean train host: $host ==="
  run_ssh "$host" "tmux ls 2>/dev/null | sed 's/:.*//' | grep -E '^(training_|dual_train|eval|c[0-9]+|think_pipe)' | xargs -r -n1 tmux kill-session -t || true"
  remote_process_cleanup "$host" $'python -u main.py --device tpu\npython main.py --device tpu\naccelerate launch\nvllm serve\nsite-packages/ray/\nray::'
  run_ssh "$host" "
    fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
    rm -rf /tmp/tmx_xla_cache_* /tmp/hf_cache 2>/dev/null || true
    echo '[post-clean]'
    tmux ls 2>/dev/null || true
    echo '---'
    ps aux | grep -E 'main.py|vllm|ray|accelerate' | grep -v grep | head -20 || true
    echo '---'
    df -h / | tail -1
  "
}

clean_infer_host() {
  echo "=== clean infer slice: ${INFER_TPU_NAME} (${INFER_ZONE}) workers=${INFER_WORKERS} ==="
  local worker
  while IFS= read -r worker; do
    [ -n "$worker" ] || continue
    echo "--- clean infer worker $worker ---"
    infer_worker_ssh "$worker" "
      tmux ls 2>/dev/null | sed 's/:.*//' | grep -E '^(vllm_|explorer_manual_|training_|dual_train|eval)' | xargs -r -n1 tmux kill-session -t || true
      pkill -9 -f '[v]llm serve' || true
      pkill -9 -f 'site-packages/r[a]y/' || true
      pkill -9 -f 'ra[y]::' || true
      fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
      echo '[post-clean]'
      tmux ls 2>/dev/null || true
      echo '---'
      ps aux | grep -E 'main.py|vllm|ray|accelerate' | grep -v grep | head -20 || true
    "
  done < <(infer_worker_list)
}

needs_target() {
  local want="$1"
  [ "$TARGET" = "all" ] || [ "$TARGET" = "$want" ]
}

case "$MODE" in
  check)
    if needs_target train; then
      check_host "train" "$TRAIN_HOST"
    fi
    if needs_target infer; then
      check_infer_slice
    fi
    ;;
  clean)
    if [ "${CONFIRM_REMOTE_CLEANUP:-0}" != "1" ]; then
      echo "Refusing remote cleanup without CONFIRM_REMOTE_CLEANUP=1" >&2
      exit 2
    fi
    if needs_target train; then
      clean_train_host "$TRAIN_HOST"
    fi
    if needs_target infer; then
      clean_infer_host
    fi
    ;;
  -h|--help|help)
    show_usage
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    show_usage >&2
    exit 2
    ;;
esac
