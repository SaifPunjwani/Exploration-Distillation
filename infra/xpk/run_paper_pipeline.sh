#!/usr/bin/env bash
# Encapsulated paper-run launcher.
#
# Single-command driver that walks the gates 1→4 then launches the actual
# resume run (gate 6). On any gate failure, stops with non-zero rc and
# preserves the remote logs for inspection.
#
# Required env (or pass as args):
#   TPU_PROJECT  GCP project (e.g. my-gcp-project)
#   TPU_ZONE     zone (e.g. europe-west4-b)
#   TPU_NAME     existing TPU VM name (e.g. trc2-v5lite256-ew4b-r1)
# Optional env:
#   TPU_WORKER         default all
#   GATES              default "1 2 3 4 6"; set "1" for import-only
#   SKIP_BOOTSTRAP     default 0; set 1 if Tunix is already installed
#   SPEC               default configs/tunix/tmx_dapo_drgrpo_resume_step100.json
#   HOME_DIR           (required) home directory on the TPU VM
#   ENV_NAME           default tmx_tunix_py312
#
# Behavior:
#   1. If SKIP_BOOTSTRAP=0: run bootstrap_tunix_env.sh on TPU_WORKER/WORKERS (idempotent).
#   2. For each gate in GATES:
#       gate1: smoke_gate1_imports.sh
#       gate2: smoke_gate2_training.sh
#       gate3: smoke_gate3_contract.sh
#       gate4: smoke_gate4_rollout.sh   (TIERS=1 by default; bump for full run)
#       gate6: smoke_gate6_resume.sh    (the actual paper-run continuation)
#   3. Each gate writes a .ok or .fail file under ${RESULTS_DIR}.
#
# Logs land at runs_logs/paper_pipeline_<stamp>/ on the TPU and locally.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT}"
TPU_ZONE="${TPU_ZONE:?set TPU_ZONE}"
TPU_NAME="${TPU_NAME:?set TPU_NAME}"
TPU_WORKER="${TPU_WORKER:-all}"
GATES="${GATES:-1 2 3 4 6}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
SPEC="${SPEC:-${ROOT_DIR}/configs/tunix/tmx_dapo_drgrpo_resume_step100.json}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"
ENV_NAME="${ENV_NAME:-tmx_tunix_py312}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RESULTS_DIR="${ROOT_DIR}/runs_logs/paper_pipeline_${STAMP}"
mkdir -p "${RESULTS_DIR}"

echo "[pipeline] tpu=${TPU_NAME}@${TPU_PROJECT}/${TPU_ZONE} worker=${TPU_WORKER}"
echo "[pipeline] gates=[${GATES}] spec=${SPEC}"
echo "[pipeline] results=${RESULTS_DIR}"

if [ -n "${TMX_EXPERIMENT_SPEC_JSON_B64:-}" ]; then
    SPEC_B64="${TMX_EXPERIMENT_SPEC_JSON_B64}"
elif [ -f "${SPEC}" ]; then
    SPEC_B64="$(python3 -c "
import base64, json, sys
spec = json.load(open(sys.argv[1]))
print(base64.b64encode(json.dumps(spec, separators=(',', ':')).encode()).decode())
" "${SPEC}")"
    export TMX_EXPERIMENT_SPEC_JSON_B64="${SPEC_B64}"
else
    echo "[pipeline] ERROR: no SPEC file and no TMX_EXPERIMENT_SPEC_JSON_B64" >&2
    exit 2
fi

export TPU_PROJECT TPU_ZONE TPU_NAME TPU_WORKER HOME_DIR ENV_NAME SPEC TMX_EXPERIMENT_SPEC_JSON_B64
export WORKERS="${WORKERS:-${TPU_WORKER}}"

run_step() {
    local label="$1"
    local script="$2"
    local log="${RESULTS_DIR}/${label}.log"
    echo "[pipeline] >>> ${label}"
    if bash "${script}" 2>&1 | tee "${log}"; then
        touch "${RESULTS_DIR}/${label}.ok"
        echo "[pipeline] <<< ${label} PASS"
    else
        touch "${RESULTS_DIR}/${label}.fail"
        echo "[pipeline] <<< ${label} FAIL (log: ${log})" >&2
        return 3
    fi
}

if [ "${SKIP_BOOTSTRAP}" != "1" ]; then
    run_step bootstrap "${ROOT_DIR}/infra/xpk/bootstrap_tunix_env.sh" || exit 3
fi

for g in ${GATES}; do
    case "${g}" in
        1) run_step gate1 "${ROOT_DIR}/infra/xpk/smoke_gate1_imports.sh" || exit 3 ;;
        2) run_step gate2 "${ROOT_DIR}/infra/xpk/smoke_gate2_training.sh" || exit 3 ;;
        3) run_step gate3 "${ROOT_DIR}/infra/xpk/smoke_gate3_contract.sh" || exit 3 ;;
        4) TIERS="${TIERS:-1}" run_step gate4 "${ROOT_DIR}/infra/xpk/smoke_gate4_rollout.sh" || exit 3 ;;
        6) run_step gate6 "${ROOT_DIR}/infra/xpk/smoke_gate6_resume.sh" || exit 3 ;;
        *) echo "[pipeline] unknown gate: ${g}; valid: 1 2 3 4 6" >&2; exit 2 ;;
    esac
done

echo "[pipeline] ALL GATES PASS"
echo "[pipeline] results: ${RESULTS_DIR}"
