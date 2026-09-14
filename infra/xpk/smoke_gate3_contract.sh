#!/usr/bin/env bash
# Gate 3: DAPO/Dr.GRPO contract smoke.
#
# This is intentionally a dry-run of the exact production spec through
# tmx.orchestration.tunix_workload. It verifies the real launcher emits the
# contract values without using obsolete Tunix CLI key names.
set -euo pipefail

TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT}"
TPU_ZONE="${TPU_ZONE:?set TPU_ZONE}"
TPU_NAME="${TPU_NAME:?set TPU_NAME}"
TPU_WORKER="${TPU_WORKER:-all}"
ENV_NAME="${ENV_NAME:-tmx_jax}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"
SMOKE_OUT="${SMOKE_OUT:-${HOME_DIR}/runs/smoke_gate3_$(date -u +%Y%m%d_%H%M%S)}"
SPEC="${SPEC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/tunix/tmx_dapo_drgrpo_resume_step100.json}"

SPEC_B64="${TMX_EXPERIMENT_SPEC_JSON_B64:-$(python3 -c "
import base64, json, sys
spec = json.load(open(sys.argv[1]))
print(base64.b64encode(json.dumps(spec, separators=(',', ':')).encode()).decode())
" "${SPEC}")}"

echo "[gate3] tpu=${TPU_NAME} worker=${TPU_WORKER} out=${SMOKE_OUT}"

echo "[gate3] syncing tmx package to ${TPU_NAME}:${TPU_WORKER}"
gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm scp \
    --recurse "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tmx" \
    "${TPU_NAME}:${HOME_DIR}/two-model-exploration/" \
    --worker="${TPU_WORKER}" \
    --zone="${TPU_ZONE}"

REMOTE=$(cat <<REMOTE
set -euo pipefail
source "${HOME_DIR}/miniforge3/bin/activate" "${ENV_NAME}"
mkdir -p "${SMOKE_OUT}"
cd "${HOME_DIR}/two-model-exploration"

export TMX_EXPERIMENT_SPEC_JSON_B64="${SPEC_B64}"
export TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1
export TMX_DRY_RUN_WORKLOAD=1
python -m tmx.orchestration.tunix_workload run 2>&1 | tee "${SMOKE_OUT}/gate3.log"

contract_ok=true
grep -q "tunix_maxtext_programmatic_dapo_drgrpo" "${SMOKE_OUT}/gate3.log" || contract_ok=false
grep -q "grpo_config.advantage_estimator=drgrpo" "${SMOKE_OUT}/gate3.log" || contract_ok=false
grep -q "grpo_config.epsilon=0.2" "${SMOKE_OUT}/gate3.log" || contract_ok=false
grep -q "grpo_config.epsilon_high=0.28" "${SMOKE_OUT}/gate3.log" || contract_ok=false
grep -q "grpo_config.beta=null" "${SMOKE_OUT}/gate3.log" || contract_ok=false
grep -q "rollout_config.total_generation_steps=16384" "${SMOKE_OUT}/gate3.log" || contract_ok=false
grep -q "grpo_config.dynamic_sampling=true" "${SMOKE_OUT}/gate3.log" || contract_ok=false

if \${contract_ok}; then
    echo "[gate3] DAPO contract observed in log: PASS"
else
    echo "[gate3] DAPO contract NOT confirmed in log: FAIL" >&2
    exit 3
fi
REMOTE
)

gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --worker="${TPU_WORKER}" \
    --batch-size="${TPU_BATCH_SIZE:-64}" \
    --zone="${TPU_ZONE}" \
    --command="bash -s" <<< "${REMOTE}"

echo "[gate3] OK"
