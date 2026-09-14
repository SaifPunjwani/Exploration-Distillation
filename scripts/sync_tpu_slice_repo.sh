#!/usr/bin/env bash
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
SYNC_WORKERS="${SYNC_WORKERS:-all}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
DIRECT_SSH_HOST="${DIRECT_SSH_HOST:-}"
DIRECT_SSH_PROXY_JUMP="${DIRECT_SSH_PROXY_JUMP:-}"
FANOUT_SEED_WORKER="${FANOUT_SEED_WORKER:-0}"
FANOUT_PARALLELISM="${FANOUT_PARALLELISM:-8}"
FANOUT_HTTP_PORT="${FANOUT_HTTP_PORT:-18765}"

ensure_gcloud_ssh_key() {
  local key_path="${HOME}/.ssh/google_compute_engine"
  if [ ! -f "$key_path" ]; then
    return
  fi
  if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l >/dev/null 2>&1; then
    eval "$(ssh-agent -s)" >/dev/null
  fi
  ssh-add "$key_path" >/dev/null 2>&1 || true
}

ensure_gcloud_ssh_key

use_direct_ssh=0
if [ -n "$DIRECT_SSH_HOST" ] && [ "$SYNC_WORKERS" = "0" ]; then
  use_direct_ssh=1
fi

copy_to_direct_ssh() {
  local local_path="$1"
  local remote_path="$2"
  local attempt=1
  local max_attempts=3
  while true; do
    if scp \
      -C \
      -o BatchMode=yes \
      -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no \
      ${DIRECT_SSH_PROXY_JUMP:+-o ProxyJump="$DIRECT_SSH_PROXY_JUMP"} \
      "$local_path" \
      "${DIRECT_SSH_HOST}:${remote_path}"
    then
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "[sync] ERROR: direct scp to ${DIRECT_SSH_HOST} failed after ${max_attempts} attempts" >&2
      return 1
    fi
    echo "[sync] WARN: direct scp attempt ${attempt}/${max_attempts} failed; retrying..." >&2
    attempt=$((attempt + 1))
    sleep 5
  done
}

describe_worker_ips() {
  local desc
  desc="$(expdis_gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format=json)"
  python3 - "$desc" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
eps = payload.get("networkEndpoints") or payload.get("workerNetworkEndpoints") or []
for idx, ep in enumerate(eps):
    ip = ep.get("ipAddress", "")
    if ip:
        print(f"{idx} {ip}")
PY
}

run_remote_command() {
  local command="$1"
  if [ "$use_direct_ssh" = "1" ]; then
    local attempt=1
    local max_attempts=3
    while true; do
      if ssh \
        -o BatchMode=yes \
        -o ConnectTimeout=20 \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o StrictHostKeyChecking=no \
        ${DIRECT_SSH_PROXY_JUMP:+-o ProxyJump="$DIRECT_SSH_PROXY_JUMP"} \
        "$DIRECT_SSH_HOST" \
        "$command"
      then
        return 0
      fi
      if [ "$attempt" -ge "$max_attempts" ]; then
        echo "[sync] ERROR: direct ssh command on ${DIRECT_SSH_HOST} failed after ${max_attempts} attempts" >&2
        return 1
      fi
      echo "[sync] WARN: direct ssh command attempt ${attempt}/${max_attempts} failed; retrying..." >&2
      attempt=$((attempt + 1))
      sleep 5
    done
  else
    expdis_gcloud alpha compute tpus tpu-vm ssh \
      "$TPU_NAME" \
      --worker="$SYNC_WORKERS" \
      --batch-size=all \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG" \
      --command="$command"
  fi
}

copy_to_remote() {
  local local_path="$1"
  local remote_path="$2"
  if [ "$use_direct_ssh" = "1" ]; then
    copy_to_direct_ssh "$local_path" "$remote_path"
  else
    expdis_gcloud alpha compute tpus tpu-vm scp \
      "$local_path" \
      "${TPU_NAME}:${remote_path}" \
      --worker="$SYNC_WORKERS" \
      --batch-size=all \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG"
  fi
}

seed_worker_ip_from_map() {
  local worker_ip_map="$1"
  local seed_worker="$2"
  python3 - "$worker_ip_map" "$seed_worker" <<'PY'
import sys

worker_ip_map = sys.argv[1].splitlines()
seed_worker = sys.argv[2].strip()
for entry in worker_ip_map:
    entry = entry.strip()
    if not entry:
        continue
    worker, ip = entry.split(" ", 1)
    if worker == seed_worker:
        print(ip.strip())
        raise SystemExit(0)
raise SystemExit(1)
PY
}

start_seed_archive_server() {
  local remote_path="$1"
  local seed_worker="$2"
  local http_port="$3"
  local remote_dir
  local remote_file
  remote_dir="$(dirname "$remote_path")"
  remote_file="$(basename "$remote_path")"
  expdis_gcloud alpha compute tpus tpu-vm ssh \
    "$TPU_NAME" \
    --worker="$seed_worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="
      set -euo pipefail
      remote_dir='$remote_dir'
      remote_file='$remote_file'
      http_port='$http_port'
      cd \"\$remote_dir\"
      pkill -f \"python3 -m http.server \$http_port\" 2>/dev/null || true
      nohup python3 -m http.server \"\$http_port\" --bind 0.0.0.0 >/tmp/expdis_fanout_http_${http_port}.log 2>&1 &
      server_pid=\$!
      echo \"\$server_pid\" >/tmp/expdis_fanout_http_${http_port}.pid
      for _ in \$(seq 1 20); do
        if curl -fsS \"http://127.0.0.1:\$http_port/\$remote_file\" >/dev/null 2>&1; then
          exit 0
        fi
        sleep 1
      done
      echo '[sync] ERROR: seed archive server failed to start' >&2
      exit 1
    "
}

stop_seed_archive_server() {
  local seed_worker="$1"
  local http_port="$2"
  expdis_gcloud alpha compute tpus tpu-vm ssh \
    "$TPU_NAME" \
    --worker="$seed_worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="
      set -euo pipefail
      if [ -f /tmp/expdis_fanout_http_${http_port}.pid ]; then
        kill \$(cat /tmp/expdis_fanout_http_${http_port}.pid) 2>/dev/null || true
        rm -f /tmp/expdis_fanout_http_${http_port}.pid
      fi
      pkill -f 'python3 -m http.server ${http_port}' 2>/dev/null || true
    " >/dev/null 2>&1 || true
}

have_git=1
dirty_tree=0
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
  echo "Refusing to sync a dirty working tree. Commit or set ALLOW_DIRTY=1."
  exit 1
fi

if [ "$have_git" = "1" ]; then
  commit="$(git rev-parse --short HEAD)"
else
  commit="nogit-$(date +%Y%m%d%H%M%S)"
fi
archive="/tmp/two-model-exploration-${commit}.tar"
remote_archive="/tmp/$(basename "$archive")"

if [ "$have_git" = "0" ]; then
  echo "[sync] No git metadata detected; archiving current working tree."
  COPYFILE_DISABLE=1 tar \
    --exclude-vcs \
    --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./.venv*' \
    --exclude='./runs' \
    --exclude='./wandb' \
    --exclude='./.wandb_cache' \
    --exclude='./.wandb_data' \
    --exclude='./.hf' \
    --exclude='./.hf*' \
    --exclude='./.gcloud_codex' \
    --exclude='./._*' \
    --exclude='./__pycache__' \
    --exclude='./tests/__pycache__' \
    -cf "$archive" \
    -C "$ROOT_DIR" \
    .
elif [ "$dirty_tree" = "1" ]; then
  echo "[sync] ALLOW_DIRTY=1 with local modifications; archiving current working tree."
  COPYFILE_DISABLE=1 tar \
    --exclude-vcs \
    --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./.venv*' \
    --exclude='./runs' \
    --exclude='./wandb' \
    --exclude='./.wandb_cache' \
    --exclude='./.wandb_data' \
    --exclude='./.hf' \
    --exclude='./.hf*' \
    --exclude='./.gcloud_codex' \
    --exclude='./._*' \
    --exclude='./__pycache__' \
    --exclude='./tests/__pycache__' \
    -cf "$archive" \
    -C "$ROOT_DIR" \
    .
else
  git archive --format=tar -o "$archive" HEAD
fi

FANOUT_DISABLED="${FANOUT_DISABLED:-0}"
if [ "$use_direct_ssh" = "0" ] && [ "$SYNC_WORKERS" = "all" ] && [ "$FANOUT_DISABLED" = "0" ]; then
  worker_ip_map="$(describe_worker_ips)"
  worker_count="$(printf '%s\n' "$worker_ip_map" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [ "${worker_count:-0}" -gt 1 ]; then
    echo "[sync] full-slice sync detected; copying archive once to worker $FANOUT_SEED_WORKER and fanning out internally."
    if [ -n "$DIRECT_SSH_HOST" ]; then
      copy_to_direct_ssh "$archive" "$remote_archive"
    else
      expdis_gcloud alpha compute tpus tpu-vm scp \
        "$archive" \
        "${TPU_NAME}:${remote_archive}" \
        --worker="$FANOUT_SEED_WORKER" \
        --zone="$ZONE" \
        "$TPU_TRANSPORT_FLAG"
    fi
    seed_worker_ip="$(seed_worker_ip_from_map "$worker_ip_map" "$FANOUT_SEED_WORKER")"
    start_seed_archive_server "$remote_archive" "$FANOUT_SEED_WORKER" "$FANOUT_HTTP_PORT"
    trap 'stop_seed_archive_server "$FANOUT_SEED_WORKER" "$FANOUT_HTTP_PORT"' EXIT
    run_remote_command "
      set -euo pipefail
      mkdir -p '$REMOTE_ROOT'
      if hostname -I 2>/dev/null | tr ' ' '\n' | grep -Fxq '${seed_worker_ip}'; then
        :
      else
        rm -f '$remote_archive'
        curl -fsS 'http://${seed_worker_ip}:${FANOUT_HTTP_PORT}/$(basename "$remote_archive")' -o '$remote_archive'
      fi
      find '$REMOTE_ROOT' -mindepth 1 -maxdepth 1 ! -name '.venv_tpu_slice' ! -name '.hf' ! -name '.hf*' ! -name 'runs' ! -name 'wandb' ! -name '.wandb_cache' ! -name '.wandb_data' ! -name '.env' ! -name '.env.*' -exec rm -rf {} +
      tar -xf '$remote_archive' -C '$REMOTE_ROOT'
    "
    stop_seed_archive_server "$FANOUT_SEED_WORKER" "$FANOUT_HTTP_PORT"
    trap - EXIT
  else
    copy_to_remote "$archive" "$remote_archive"
    run_remote_command "mkdir -p '$REMOTE_ROOT' && find '$REMOTE_ROOT' -mindepth 1 -maxdepth 1 ! -name '.venv_tpu_slice' ! -name '.hf' ! -name '.hf*' ! -name 'runs' ! -name 'wandb' ! -name '.wandb_cache' ! -name '.wandb_data' ! -name '.env' ! -name '.env.*' -exec rm -rf {} + && tar -xf '$remote_archive' -C '$REMOTE_ROOT'"
  fi
else
  copy_to_remote "$archive" "$remote_archive"
  run_remote_command "mkdir -p '$REMOTE_ROOT' && find '$REMOTE_ROOT' -mindepth 1 -maxdepth 1 ! -name '.venv_tpu_slice' ! -name '.hf' ! -name '.hf*' ! -name 'runs' ! -name 'wandb' ! -name '.wandb_cache' ! -name '.wandb_data' ! -name '.env' ! -name '.env.*' -exec rm -rf {} + && tar -xf '$remote_archive' -C '$REMOTE_ROOT'"
fi

echo "Synced commit $commit to TPU workers $SYNC_WORKERS at $REMOTE_ROOT"
