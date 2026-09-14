#!/usr/bin/env bash
# Install Tunix + MaxText into a TPU VM's existing miniforge env.
#
# This is the no-Docker, no-XPK path: SSH to a TPU VM, run this once per
# allocation, then run the smoke gates / production launcher directly with
# `python -m tunix.cli.grpo_main` or `python -m tmx.orchestration.tunix_workload`.
#
# Required env (or pass as args):
#   TPU_PROJECT       e.g. my-gcp-project
#   TPU_ZONE          e.g. europe-west4-b
#   TPU_NAME          e.g. trc2-v5lite256-ew4b-r1
# Optional env:
#   TPU_WORKER        default 0 (worker to bootstrap; sweep with WORKERS=...)
#   WORKERS           if set, space-separated list of workers to bootstrap in parallel
#   ENV_NAME          default tmx_tunix_py312
#   TUNIX_REF         default main
#   MAXTEXT_REF       default main
#
# What it does on the worker:
#   1. Ensure miniforge exists; install it if this TPU VM is fresh.
#   2. Ensure the env exists.
#   3. Clone tunix + maxtext into /home/<user>/{tunix,maxtext} if missing.
#   4. VLLM_TARGET_DEVICE=tpu pip install -r tunix/requirements/requirements.txt
#      (gets the pinned vllm + tpu-inference commits — DO NOT replace with PyPI vllm).
#   5. pip install tunix + maxtext (editable).
#   6. Smoke import without initializing JAX TPU backend.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT}"
TPU_ZONE="${TPU_ZONE:?set TPU_ZONE}"
TPU_NAME="${TPU_NAME:?set TPU_NAME}"
WORKERS="${WORKERS:-${TPU_WORKER:-0}}"
ENV_NAME="${ENV_NAME:-tmx_tunix_py312}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TUNIX_REF="${TUNIX_REF:-main}"
MAXTEXT_REF="${MAXTEXT_REF:-main}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"
BOOTSTRAP_JAX_SMOKE="${BOOTSTRAP_JAX_SMOKE:-0}"
BOOTSTRAP_REINSTALL="${BOOTSTRAP_REINSTALL:-0}"

if [ "${WORKERS}" = "all" ]; then
    # Discover worker count from gcloud.
    WORKER_COUNT="$(
        gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm describe "${TPU_NAME}" \
            --zone="${TPU_ZONE}" --format='json(networkEndpoints)' \
          | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("networkEndpoints", [])))'
    )"
    WORKERS="$(seq 0 $((WORKER_COUNT - 1)))"
fi

echo "[bootstrap-tunix] project=${TPU_PROJECT} zone=${TPU_ZONE} tpu=${TPU_NAME}"
echo "[bootstrap-tunix] workers=[${WORKERS}] env=${ENV_NAME} python=${PYTHON_VERSION} tunix_ref=${TUNIX_REF}"

# Refuse forbidden CLI flags.
for arg in "$@"; do
    case "${arg}" in
        --internal-ip|--internal-ips)
            echo "[bootstrap-tunix] ERROR: refusing ${arg} (TRC requires external IPs)" >&2
            exit 2
            ;;
    esac
done

REMOTE_SCRIPT=$(cat <<'REMOTE'
set -euo pipefail
ENV_NAME="${ENV_NAME:-tmx_jax}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"
TUNIX_REF="${TUNIX_REF:-main}"
MAXTEXT_REF="${MAXTEXT_REF:-main}"
BOOTSTRAP_REINSTALL="${BOOTSTRAP_REINSTALL:-0}"

echo "[remote] host=$(hostname) date=$(date -Is)"

if [ ! -x "${HOME_DIR}/miniforge3/bin/conda" ]; then
    echo "[remote] installing miniforge under ${HOME_DIR}/miniforge3"
    tmp_installer="/tmp/miniforge.sh"
    curl -fsSL -o "${tmp_installer}" \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "${tmp_installer}" -b -p "${HOME_DIR}/miniforge3"
fi

source "${HOME_DIR}/miniforge3/bin/activate"
if [ "${BOOTSTRAP_REINSTALL}" = "1" ]; then
    echo "[remote] removing existing conda env ${ENV_NAME}"
    conda env remove -y -n "${ENV_NAME}" || true
fi
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[remote] creating conda env ${ENV_NAME} python=${PYTHON_VERSION}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

source "${HOME_DIR}/miniforge3/bin/activate" "${ENV_NAME}"
python --version

# Tunix from source — must use tunix's pinned vllm + tpu-inference commits.
if [ ! -d "${HOME_DIR}/tunix/.git" ]; then
    git clone --depth 1 --branch "${TUNIX_REF}" https://github.com/google/tunix.git "${HOME_DIR}/tunix"
else
    git -C "${HOME_DIR}/tunix" pull --ff-only || true
fi

CONSTRAINTS="/tmp/tmx_tunix_tpu_constraints.txt"
cat > "${CONSTRAINTS}" <<'EOF'
# Keep the Tunix/vLLM TPU stack compatible. vLLM's pinned tpu-inference
# package requires this JAX/flax/qwix stack; letting Tunix/MaxText pull latest
# JAX breaks TPU serving.
jax==0.8.0
jaxlib==0.8.0
flax==0.11.1
qwix==0.1.1
EOF

# CRITICAL: VLLM_TARGET_DEVICE=tpu so pip builds the TPU vllm wheel.
export VLLM_TARGET_DEVICE=tpu
pip install --upgrade pip
# Preinstall CPU PyTorch wheels before vLLM dependency resolution. Without
# this, pip satisfies vLLM's broad torch/torchvision requirements with CUDA
# wheels, wasting many GB per TPU worker and slowing fresh-pod bootstrap.
pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.9.0+cpu" \
  "torchvision==0.24.0+cpu"
pip install -r "${HOME_DIR}/tunix/requirements/requirements.txt" -c "${CONSTRAINTS}"
pip install -e "${HOME_DIR}/tunix" -c "${CONSTRAINTS}"

# MaxText from source — needed for model_source=maxtext. Install editable
# without dependencies because its generated "latest" dependency set currently
# pulls JAX versions newer than vLLM TPU supports.
if [ ! -d "${HOME_DIR}/maxtext/.git" ]; then
    git clone --depth 1 --branch "${MAXTEXT_REF}" https://github.com/AI-Hypercomputer/maxtext.git "${HOME_DIR}/maxtext"
else
    git -C "${HOME_DIR}/maxtext" pull --ff-only || true
fi
pip install -e "${HOME_DIR}/maxtext" --no-deps

# Smoke import. Avoid initializing the TPU backend here unless explicitly
# requested; multi-host JAX init must be coordinated across all 64 workers.
python - <<'PY'
import maxtext
print("[remote] maxtext OK")
import tunix
print("[remote] tunix OK")
from tunix.rl.grpo.drgrpo_learner import DrGRPOConfig
print("[remote] DrGRPOConfig OK")
from tunix.cli.config import obtain_reward_fn
print("[remote] reward_fn loader OK")
print("[remote] PASS")
PY

if [ "${BOOTSTRAP_JAX_SMOKE:-0}" = "1" ]; then
python - <<'PY'
import jax
print(f"[remote] jax {jax.__version__} devices={len(jax.devices())}")
PY
fi
REMOTE
)

bootstrap_one() {
    local w="$1"
    echo "[bootstrap-tunix] worker ${w}: starting"
    gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
        --worker="${w}" \
        --zone="${TPU_ZONE}" \
        --command="HOME_DIR='${HOME_DIR}' ENV_NAME='${ENV_NAME}' PYTHON_VERSION='${PYTHON_VERSION}' TUNIX_REF='${TUNIX_REF}' MAXTEXT_REF='${MAXTEXT_REF}' BOOTSTRAP_REINSTALL='${BOOTSTRAP_REINSTALL}' BOOTSTRAP_JAX_SMOKE='${BOOTSTRAP_JAX_SMOKE}' bash -s" <<< "${REMOTE_SCRIPT}"
}

# Parallel bootstrap if multiple workers requested.
pids=()
for w in ${WORKERS}; do
    bootstrap_one "${w}" 2>&1 | sed "s/^/[w${w}] /" &
    pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
    wait "${pid}" || rc=$?
done

if [ "${rc}" -eq 0 ]; then
    echo "[bootstrap-tunix] OK"
else
    echo "[bootstrap-tunix] FAIL (rc=${rc})" >&2
fi
exit "${rc}"
