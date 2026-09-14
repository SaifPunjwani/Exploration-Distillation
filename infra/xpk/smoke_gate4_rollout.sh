#!/usr/bin/env bash
# Gate 4: rollout scale-up smoke through the real TMX Tunix dispatcher.
#
# Tiers:
#   1: 4 generations, 2K completion
#   2: 8 generations, 8K completion
#   3: 16 generations, 16K completion
#
# This intentionally avoids raw/obsolete Tunix CLI override names. Each tier
# is a small correctness-only GRPO run through tmx.orchestration.tunix_workload,
# so vLLM-on-TPU is exercised through the same dispatcher as production.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT}"
TPU_ZONE="${TPU_ZONE:?set TPU_ZONE}"
TPU_NAME="${TPU_NAME:?set TPU_NAME}"
TPU_WORKER="${TPU_WORKER:-all}"
ENV_NAME="${ENV_NAME:-tmx_jax}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"
SMOKE_OUT="${SMOKE_OUT:-${HOME_DIR}/runs/smoke_gate4_$(date -u +%Y%m%d_%H%M%S)}"
TIERS="${TIERS:-1 2 3}"
BASE_SPEC="${BASE_SPEC:-${ROOT_DIR}/configs/tunix/tmx_smoke_tiny.json}"

make_spec_b64() {
    local tier="$1"
    local n_gen="$2"
    local max_comp="$3"
    python3 -c "
import base64, json, sys
path, tier, n_gen, max_comp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
spec = json.load(open(path))
spec['name'] = f\"tmx_gate4_tier{tier}_{max_comp}\"
spec['storage']['artifact_prefix'] = spec['name']
spec['storage']['checkpoint_every_steps'] = 1
alg = spec['algorithm']
alg['phase'] = 'explorer_correctness'
alg['max_prompt_len'] = 2048
alg['max_completion_len'] = max_comp
alg['max_total_len'] = 2048 + max_comp
alg['grpo_num_generations'] = n_gen
alg['grpo_max_steps'] = 1
alg['lambda_novelty'] = 0.0
alg['incorrect_novelty_scale'] = 0.0
alg['dynamic_sampling_accept_reward_variance'] = False
alg['soft_overlong_expected_len'] = int(max_comp * 0.8)
alg['soft_overlong_cache_len'] = max_comp - alg['soft_overlong_expected_len']
spec['backend']['maxtext_base_output_directory'] = f\"/tmp/{spec['name']}\"
print(base64.b64encode(json.dumps(spec, separators=(',', ':')).encode()).decode())
" "${BASE_SPEC}" "${tier}" "${n_gen}" "${max_comp}"
}

echo "[gate4] tpu=${TPU_NAME} worker=${TPU_WORKER} tiers=[${TIERS}]"
echo "[gate4] syncing tmx package to ${TPU_NAME}:${TPU_WORKER}"
gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm scp \
    --recurse "${ROOT_DIR}/tmx" \
    "${TPU_NAME}:${HOME_DIR}/two-model-exploration/" \
    --worker="${TPU_WORKER}" \
    --zone="${TPU_ZONE}"

for t in ${TIERS}; do
    case "${t}" in
        1) n_gen=4; max_comp=2048 ;;
        2) n_gen=8; max_comp=8192 ;;
        3) n_gen=16; max_comp=16384 ;;
        *) echo "[gate4] unknown tier ${t}; valid: 1 2 3" >&2; exit 2 ;;
    esac
    SPEC_B64="$(make_spec_b64 "${t}" "${n_gen}" "${max_comp}")"
    REMOTE=$(cat <<REMOTE
set -euo pipefail
source "${HOME_DIR}/miniforge3/bin/activate" "${ENV_NAME}"
cd "${HOME_DIR}/two-model-exploration"
mkdir -p "${SMOKE_OUT}/tier${t}"
export TMX_EXPERIMENT_SPEC_JSON_B64="${SPEC_B64}"
export TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1
export PYTHONUNBUFFERED=1
echo "[gate4] tier=${t} n_gen=${n_gen} max_completion=${max_comp}"
python -m tmx.orchestration.tunix_workload describe | tee "${SMOKE_OUT}/tier${t}/describe.log"
python -m tmx.orchestration.tunix_workload run 2>&1 | tee "${SMOKE_OUT}/tier${t}/run.log"
echo "[gate4] tier ${t} PASS"
REMOTE
)
    gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
        --worker="${TPU_WORKER}" \
        --batch-size="${TPU_BATCH_SIZE:-64}" \
        --zone="${TPU_ZONE}" \
        --command="bash -s" <<< "${REMOTE}"
done

echo "[gate4] OK"
