#!/usr/bin/env bash
# Bootstrap the JAX/TPU environment on a fresh TPU VM.
# Uses Miniforge (no Anaconda ToS, no system python3-venv dep).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-tmx_jax}"
JAX_VERSION="${JAX_VERSION:-0.7.1}"

if [ ! -d "$ENV_DIR" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.11 pip setuptools wheel
fi
conda activate "$ENV_NAME"

if ! python -m pip --version >/dev/null 2>&1; then
  conda install -y -n "$ENV_NAME" pip setuptools wheel
fi
python -m pip install --upgrade pip setuptools wheel

# JAX on TPU (PJRT). Match the known-good worker-0 stack by default.
pip install --upgrade "jax[tpu]==${JAX_VERSION}" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install -r "$REPO_ROOT/tmx_jax/requirements.txt"

# Avoid hanging on pod-wide discovery during setup. We only need a local
# single-host sanity check here; multi-host launch supplies the full runtime env.
PJRT_DEVICE="${PJRT_DEVICE:-TPU}" \
TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}" \
TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0,1,2,3}" \
TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}" \
TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}" \
TPU_WORKER_ID="${TPU_WORKER_ID:-0}" \
TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}" \
python -c "import jax; devs=jax.devices(); print('devices:', len(devs), devs[0].platform if devs else 'none')"
echo "[setup] OK"
