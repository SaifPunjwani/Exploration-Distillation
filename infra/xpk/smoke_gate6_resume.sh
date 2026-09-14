#!/usr/bin/env bash
# Gate 6: production resume from λ=0.5 step_000100.
#
# Pre-reqs: gates 1–4 PASS, contract verified at gate 3, max-context confirmed
# at gate 4. This launches the actual paper run continuation.
#
# Reads the production spec configs/tunix/tmx_dapo_drgrpo_resume_step100.json
# and dispatches via tmx.orchestration.tunix_workload. For λ>0 this uses the
# programmatic Tunix/MaxText DAPO+Dr.GRPO adapter, not the plain Tunix CLI.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT}"
TPU_ZONE="${TPU_ZONE:?set TPU_ZONE}"
TPU_NAME="${TPU_NAME:?set TPU_NAME}"
TPU_WORKER="${TPU_WORKER:-all}"
ENV_NAME="${ENV_NAME:-tmx_jax}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"
SPEC="${SPEC:-${ROOT_DIR}/configs/tunix/tmx_dapo_drgrpo_resume_step100.json}"
RUN_LOG="${RUN_LOG:-${HOME_DIR}/runs_logs/dapo16k_drgrpo_novelty05_tunix_resume100_$(date -u +%Y%m%d_%H%M%S).log}"

if [ ! -f "${SPEC}" ]; then
    echo "[gate6] ERROR: ${SPEC} not found" >&2
    exit 2
fi

# Validate locally before remote launch.
python3 scripts/tmx_submit.py validate --spec "${SPEC}" --show-spec >/dev/null \
    && echo "[gate6] local spec.validate(): PASS" \
    || { echo "[gate6] local validate FAIL" >&2; exit 3; }

# Pack the spec into TMX_EXPERIMENT_SPEC_JSON_B64.
SPEC_B64="$(python3 -c "
import base64, json, sys
spec = json.load(open(sys.argv[1]))
print(base64.b64encode(json.dumps(spec, separators=(',', ':')).encode()).decode())
" "${SPEC}")"

# Sync our orchestration package to the TPU VM.
echo "[gate6] syncing tmx package to ${TPU_NAME}:${TPU_WORKER}"
gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm scp \
    --recurse "${ROOT_DIR}/tmx" \
    "${TPU_NAME}:${HOME_DIR}/two-model-exploration/" \
    --worker="${TPU_WORKER}" \
    --zone="${TPU_ZONE}"

REMOTE=$(cat <<REMOTE
set -euo pipefail
source "${HOME_DIR}/miniforge3/bin/activate" "${ENV_NAME}"

cd "${HOME_DIR}/two-model-exploration"
mkdir -p "$(dirname "${RUN_LOG}")"

# Hand TMX_EXPERIMENT_SPEC_JSON_B64 + ack flag to the workload.
export TMX_EXPERIMENT_SPEC_JSON_B64="${SPEC_B64}"
export TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1
export PYTHONUNBUFFERED=1

# describe first (dry-run) so we see the command before it launches.
python -m tmx.orchestration.tunix_workload describe | tee "${RUN_LOG}.describe"

# Launch in remote tmux so the run survives laptop sleep/disconnect.
tmux kill-session -t tmx_tunix_resume100 2>/dev/null || true
tmux new-session -d -s tmx_tunix_resume100 "cd '${HOME_DIR}/two-model-exploration' && \
  source '${HOME_DIR}/miniforge3/bin/activate' '${ENV_NAME}' && \
  export TMX_EXPERIMENT_SPEC_JSON_B64='${SPEC_B64}' && \
  export TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1 && \
  export PYTHONUNBUFFERED=1 && \
  python -m tmx.orchestration.tunix_workload run 2>&1 | tee '${RUN_LOG}'"
echo "[gate6] remote tmux session: tmx_tunix_resume100"
echo "[gate6] monitor: tail -f ${RUN_LOG}"
REMOTE
)

echo "[gate6] launching on ${TPU_NAME}:${TPU_WORKER}; log -> ${RUN_LOG}"
gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --worker="${TPU_WORKER}" \
    --batch-size="${TPU_BATCH_SIZE:-64}" \
    --zone="${TPU_ZONE}" \
    --command="bash -s" <<< "${REMOTE}"

echo "[gate6] OK (resume run launched)"
