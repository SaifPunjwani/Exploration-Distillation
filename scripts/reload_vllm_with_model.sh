#!/usr/bin/env bash
# Reload a vLLM server on a specific TPU worker with a new model from GCS.
# RESTART-ONLY. Mirrors the proven bootstrap pattern from
# launch_dual_slice_training.sh (standalone /tmp/session.sh + tmux new-session +
# fuser -k on vfio devices to release TPU chips).
#
# Required:
#   GCS_MODEL_PATH=gs://...
#   TPU_NAME=...
#   ZONE=...
#
# Optional:
#   WORKER=0
#   SERVED_MODEL_NAME=actual_model
#   LOCAL_MODEL_DIR_BASENAME=vllm_model
#   PORT=8000
#   TP_SIZE=4
#   MAX_MODEL_LEN=4096
#   MAX_NUM_SEQS=32
#   MAX_NUM_BATCHED_TOKENS=65536
#   VLLM_VENV_DIR=/home/<user>/vllm_tpu_env
#   REMOTE_ROOT=/home/<user>/expdis
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT_DIR/scripts/tpu_target_lib.sh"

GCS_MODEL_PATH="${GCS_MODEL_PATH:?Must set GCS_MODEL_PATH}"
TPU_NAME="${TPU_NAME:?Must set TPU_NAME}"
ZONE="${ZONE:?Must set ZONE}"
WORKER="${WORKER:-0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-actual_model}"
LOCAL_MODEL_DIR_BASENAME="${LOCAL_MODEL_DIR_BASENAME:-vllm_model}"
SESSION_NAME="${SESSION_NAME:-vllm_reload_${SERVED_MODEL_NAME//[^A-Za-z0-9_]/_}_w${WORKER}}"
TPU_TRANSPORT_FLAG="${TPU_TRANSPORT_FLAG:-$(expdis_tpu_transport_flag)}"
LOCAL_MODEL_DIR="/tmp/${LOCAL_MODEL_DIR_BASENAME}_w${WORKER}"

echo "[reload] worker=${WORKER} served_model_name=${SERVED_MODEL_NAME} gcs=${GCS_MODEL_PATH}"

read -r -d '' REMOTE_CMD <<EOF || true
set -euo pipefail
# 1. Download new model from GCS.
mkdir -p '${LOCAL_MODEL_DIR}'
rm -rf '${LOCAL_MODEL_DIR}'
mkdir -p '${LOCAL_MODEL_DIR}'
gsutil -m cp -r '${GCS_MODEL_PATH}'/* '${LOCAL_MODEL_DIR}/'
cd '${REMOTE_ROOT}'

# 2. Write a standalone launcher that does teardown + relaunch inside itself,
#    so fuser -k / pkill can't nuke our SSH session. The launcher runs detached
#    via tmux; the SSH command returns quickly, and then we poll /health.
cat > /tmp/${SESSION_NAME}.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd '${REMOTE_ROOT}'
# Teardown of any previous vLLM and TPU chip holders — must happen inside this
# detached script, not in the SSH remote command, because fuser -k would
# otherwise SIGKILL processes that include the SSH shell itself.
tmux kill-server 2>/dev/null || true
pkill -9 -f '[v]llm serve' 2>/dev/null || true
fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
sleep 3
export VLLM_VENV_DIR='${VLLM_VENV_DIR}'
export MODEL_NAME='${LOCAL_MODEL_DIR}'
export SERVED_MODEL_NAME='${SERVED_MODEL_NAME}'
export PORT='${PORT}'
export TP_SIZE='${TP_SIZE}'
export MAX_MODEL_LEN='${MAX_MODEL_LEN}'
export MAX_NUM_SEQS='${MAX_NUM_SEQS}'
export MAX_NUM_BATCHED_TOKENS='${MAX_NUM_BATCHED_TOKENS}'
export EXPDIS_VLLM_HOT_RELOAD=0
bash scripts/run_vllm_server.sh > /tmp/${SESSION_NAME}.log 2>&1
INNER
chmod +x /tmp/${SESSION_NAME}.sh

# 3. Detach via setsid + nohup so the script survives SSH disconnect even if
#    some in-script kill removes our bash. Kill the existing vllm tmux session
#    cleanly first (NOT kill-server — that would also kill this SSH's tmux if
#    one exists on the worker).
for s in \$(tmux ls 2>/dev/null | awk -F: '{print \$1}' | grep -E 'vllm_(server|reload)_' || true); do
  tmux kill-session -t "\$s" 2>/dev/null || true
done
setsid nohup bash /tmp/${SESSION_NAME}.sh >/tmp/${SESSION_NAME}.driver.log 2>&1 </dev/null &
disown || true
sleep 5  # give the backgrounded launcher a head start

# 4. Wait for /health to become 200 OK — up to 5 min.
for attempt in \$(seq 1 60); do
  if curl -s --connect-timeout 3 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "HEALTHY after \${attempt} attempts"
    exit 0
  fi
  sleep 5
done
echo "ERROR: vLLM reload did not become healthy after 5 min"
echo "--- last 40 lines of /tmp/${SESSION_NAME}.log ---"
tail -40 "/tmp/${SESSION_NAME}.log" 2>&1 || true
exit 1
EOF

expdis_gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --worker="$WORKER" \
  --zone="$ZONE" \
  "$TPU_TRANSPORT_FLAG" \
  --command="$REMOTE_CMD"
