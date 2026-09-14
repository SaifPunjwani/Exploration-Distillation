#!/usr/bin/env bash
# Gate 2: tiny correctness-only training smoke through the real TMX Tunix
# dispatcher. This uses configs/tunix/tmx_smoke_tiny.json and verifies that
# the direct Tunix/MaxText/vLLM entrypoint starts without obsolete CLI keys.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT}"
TPU_ZONE="${TPU_ZONE:?set TPU_ZONE}"
TPU_NAME="${TPU_NAME:?set TPU_NAME}"
TPU_WORKER="${TPU_WORKER:-all}"
ENV_NAME="${ENV_NAME:-tmx_jax}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"
SMOKE_OUT="${SMOKE_OUT:-${HOME_DIR}/runs/smoke_gate2_$(date -u +%Y%m%d_%H%M%S)}"
SPEC="${SPEC_GATE2:-${ROOT_DIR}/configs/tunix/tmx_smoke_tiny.json}"

SPEC_B64="$(python3 -c "
import base64, json, sys
spec = json.load(open(sys.argv[1]))
print(base64.b64encode(json.dumps(spec, separators=(',', ':')).encode()).decode())
" "${SPEC}")"

echo "[gate2] tpu=${TPU_NAME} worker=${TPU_WORKER} out=${SMOKE_OUT}"
echo "[gate2] syncing tmx package to ${TPU_NAME}:${TPU_WORKER}"
gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm scp \
    --recurse "${ROOT_DIR}/tmx" \
    "${TPU_NAME}:${HOME_DIR}/two-model-exploration/" \
    --worker="${TPU_WORKER}" \
    --zone="${TPU_ZONE}"

REMOTE=$(cat <<REMOTE
set -euo pipefail
source "${HOME_DIR}/miniforge3/bin/activate" "${ENV_NAME}"
cd "${HOME_DIR}/two-model-exploration"
mkdir -p "${SMOKE_OUT}"
export TMX_EXPERIMENT_SPEC_JSON_B64="${SPEC_B64}"
export TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1
export PYTHONUNBUFFERED=1
python -m tmx.orchestration.tunix_workload describe | tee "${SMOKE_OUT}/describe.log"
python -m tmx.orchestration.tunix_workload run 2>&1 | tee "${SMOKE_OUT}/run.log"
echo "[gate2] PASS"
REMOTE
)

gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --worker="${TPU_WORKER}" \
    --batch-size="${TPU_BATCH_SIZE:-64}" \
    --zone="${TPU_ZONE}" \
    --command="bash -s" <<< "${REMOTE}"

echo "[gate2] OK"
