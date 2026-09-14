#!/usr/bin/env bash
# Launch DAPO+Dr.GRPO+novelty training on all 256 chips of trc2-v5lite256-ew4b-r1.
#
# Architecture:
#   - All 64 workers run ONE Python process each.
#   - All 64 processes JAX-init together via the libtpu coordinator on worker 0.
#   - JAX sees all 256 chips; FSDP shards the actor across them.
#   - Tunix runs vLLM in-process inside each worker's Python — no SSH tunnels,
#     no cross-pod data plane, no libtpu sub-slice fight (every worker
#     participates in the full 8x8 fabric, which is what libtpu wants).
#   - tmx.orchestration.tunix_workload dispatches to tunix_main programmatic
#     entry when lambda_novelty > 0 (which builds DAPOConfig+Dr.GRPO+novelty).
#
# Pre-reqs:
#   - bootstrap_tunix_env.sh has installed Tunix+MaxText into the env on
#     ALL 64 workers (idempotent; takes ~20 min on first run).
#   - HF_TOKEN + WANDB_API_KEY are set in the env this script is run in;
#     they're forwarded to all workers via gcloud ssh --command.
#
# This script is a parallel sibling to the codex raw-tpu path; it does NOT
# touch /home/<user>/two-model-exploration or the tmx_jax conda env.
# It uses /home/<user>/tmx-tunix and the tmx_tunix_py312 env, and
# writes its checkpoints under HF prefix dapo16k_drgrpo_*_tunix_v256_*.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}"
TPU_ZONE="${TPU_ZONE:-europe-west4-b}"
TPU_NAME="${TPU_NAME:-trc2-v5lite256-ew4b-r1}"
ENV_NAME="${ENV_NAME:-tmx_tunix_py312}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"
WORK_DIR="${WORK_DIR:-${HOME_DIR}/tmx-tunix}"
SPEC="${SPEC:?set SPEC=path/to/tunix_spec.json}"
COORD_PORT="${COORD_PORT:-12345}"
SESSION_PREFIX="${SESSION_PREFIX:-tmx_tunix_v256}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "${SPEC}" ]; then
    echo "[launch] ERROR: spec ${SPEC} not found" >&2
    exit 2
fi
if [ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "[launch] ERROR: set HF_TOKEN before launching" >&2
    exit 2
fi

# Discover the leader (worker 0) internal IP for JAX coordinator.
LEADER_IP="$(
    gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm describe "${TPU_NAME}" \
        --zone="${TPU_ZONE}" --format=json \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['networkEndpoints'][0]['ipAddress'])"
)"
NUM_WORKERS="$(
    gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm describe "${TPU_NAME}" \
        --zone="${TPU_ZONE}" --format=json \
      | python3 -c "import json,sys;d=json.load(sys.stdin);print(len(d['networkEndpoints']))"
)"

RUN_NAME="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "${SPEC}")"
SPEC_B64="$(python3 -c "
import base64,json,sys
spec=json.load(open(sys.argv[1]))
print(base64.b64encode(json.dumps(spec,separators=(',', ':')).encode()).decode())
" "${SPEC}")"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
SESSION="${SESSION_PREFIX}_${RUN_NAME}_${STAMP}"
RUN_LOG_BASE="${HOME_DIR}/runs_logs/${RUN_NAME}_${STAMP}"

echo "[launch] tpu=${TPU_NAME}@${TPU_PROJECT}/${TPU_ZONE}"
echo "[launch] num_workers=${NUM_WORKERS} leader_ip=${LEADER_IP}:${COORD_PORT}"
echo "[launch] run_name=${RUN_NAME}"
echo "[launch] tmux session per worker: ${SESSION}_w<i>"
echo "[launch] log dir on each worker: ${RUN_LOG_BASE}/"

if [ "${DRY_RUN}" = "1" ]; then
    echo "[launch] DRY_RUN=1 — printing remote command and exiting"
fi

# 1) Sync the tmx package to every worker, into ${WORK_DIR}/.
#    Each worker runs Python from ${WORK_DIR}, so PYTHONPATH naturally finds tmx.
echo "[launch] syncing tmx package to all ${NUM_WORKERS} workers"
gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${TPU_ZONE}" --worker=all \
    --command="mkdir -p '${WORK_DIR}'" >/dev/null

gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm scp \
    --recurse "${ROOT_DIR}/tmx" "${ROOT_DIR}/configs" \
    "${TPU_NAME}:${WORK_DIR}/" \
    --zone="${TPU_ZONE}" --worker=all >/dev/null

# 2) Build the per-worker remote launch script. ${TPU_WORKER_ID} is auto-set
#    by gcloud on each worker, giving us a stable 0..N-1 process id.
REMOTE_SCRIPT=$(cat <<REMOTE
set -euo pipefail

mkdir -p "${RUN_LOG_BASE}"
WORKER_LOG="${RUN_LOG_BASE}/w\${TPU_WORKER_ID}.log"
SESSION_NAME="${SESSION}_w\${TPU_WORKER_ID}"

# Don't double-launch: kill any previous tmux session of the same name and
# any leftover tmx-tunix python process. Does NOT touch codex's tmx_jax env
# or tmux sessions (different name prefix).
tmux kill-session -t "\${SESSION_NAME}" 2>/dev/null || true
pkill -f "tmx.orchestration.tunix_main" 2>/dev/null || true

# Verify the tmx_tunix_py312 env exists (bootstrap pre-req).
if [ ! -d "${HOME_DIR}/miniforge3/envs/${ENV_NAME}" ]; then
    echo "[w\${TPU_WORKER_ID}] ERROR: env ${ENV_NAME} not installed; run bootstrap_tunix_env.sh first" | tee "\${WORKER_LOG}"
    exit 3
fi

cat > /tmp/tmx_tunix_run_w\${TPU_WORKER_ID}.sh <<INNER
#!/usr/bin/env bash
set -euo pipefail
cd "${WORK_DIR}"

source "${HOME_DIR}/miniforge3/bin/activate" "${ENV_NAME}"

# Multi-host JAX coordinator. Worker 0 is the rendezvous point; all 64
# Python processes call jax.distributed.initialize() with these env vars
# during tunix_main.run() -> apply_runtime_env().
export JAX_NUM_PROCESSES=${NUM_WORKERS}
export JAX_PROCESS_ID=\${TPU_WORKER_ID}
export JAX_COORDINATOR_ADDRESS=${LEADER_IP}:${COORD_PORT}
export TPU_NAME=${TPU_NAME}
export PYTHONUNBUFFERED=1

# DAPO + Dr.GRPO + novelty contract — gated by both ack flags.
export TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1
export TMX_ALLOW_EXPERIMENTAL_TUNIX_NOVELTY=1
export TMX_EXPERIMENT_SPEC_JSON_B64='${SPEC_B64}'

# HF + W&B
export HF_TOKEN='${HF_TOKEN:-}'
export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN:-}'
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_API_KEY='${WANDB_API_KEY:-}'
export WANDB_PROJECT=two-model-explore
export WANDB_ENTITY="${WANDB_ENTITY:-}"

# Worker 0 prints describe before launching for an audit trail.
if [ "\${TPU_WORKER_ID}" = "0" ]; then
    python -m tmx.orchestration.tunix_workload describe 2>&1 | head -200
fi
exec python -m tmx.orchestration.tunix_workload run
INNER
chmod +x /tmp/tmx_tunix_run_w\${TPU_WORKER_ID}.sh

if [ "${DRY_RUN}" = "1" ]; then
    echo "[w\${TPU_WORKER_ID}] DRY_RUN — would launch:"
    cat /tmp/tmx_tunix_run_w\${TPU_WORKER_ID}.sh
    exit 0
fi

# Detach in tmux so the process survives our SSH disconnecting.
tmux new-session -d -s "\${SESSION_NAME}" \\
    "bash /tmp/tmx_tunix_run_w\${TPU_WORKER_ID}.sh 2>&1 | tee '\${WORKER_LOG}'"
echo "[w\${TPU_WORKER_ID}] launched session=\${SESSION_NAME} log=\${WORKER_LOG}"
REMOTE
)

# 3) Fan out to all workers in parallel via gcloud --worker=all.
echo "[launch] fanning out to all ${NUM_WORKERS} workers"
gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${TPU_ZONE}" --worker=all \
    --batch-size="${TPU_BATCH_SIZE:-64}" \
    --command="bash -s" <<< "${REMOTE_SCRIPT}"

if [ "${DRY_RUN}" = "1" ]; then
    echo "[launch] DRY_RUN done"
    exit 0
fi

echo
echo "[launch] OK — training launched across ${NUM_WORKERS} workers"
echo "[launch] tmux sessions: ${SESSION}_w0 .. ${SESSION}_w$((NUM_WORKERS-1))"
echo "[launch] tail leader log:"
echo "  gcloud --project=${TPU_PROJECT} alpha compute tpus tpu-vm ssh ${TPU_NAME} \\"
echo "    --zone=${TPU_ZONE} --worker=0 \\"
echo "    --command='tail -f ${RUN_LOG_BASE}/w0.log'"
echo
echo "[launch] kill all sessions for this run:"
echo "  gcloud --project=${TPU_PROJECT} alpha compute tpus tpu-vm ssh ${TPU_NAME} \\"
echo "    --zone=${TPU_ZONE} --worker=all \\"
echo "    --command='tmux kill-session -t ${SESSION}_w\${TPU_WORKER_ID} 2>/dev/null || true'"
