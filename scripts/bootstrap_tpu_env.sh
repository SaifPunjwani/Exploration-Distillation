#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

USE_TPU="${USE_TPU:-1}"
VENV_DIR="${VENV_DIR:-.venv_tpu}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-}"

if [ ! -x "$PYTHON_BIN" ] || [ ! -f "$VENV_DIR/bin/activate" ] || ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  rm -rf "$VENV_DIR"
  if ! python3 -m venv "$VENV_DIR"; then
    echo "[WARN] python3 -m venv failed; falling back to user-space virtualenv."
    python3 -m pip install --user virtualenv
    python3 -m virtualenv "$VENV_DIR"
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

if [ "$SKIP_INSTALL" != "1" ]; then
  if [ -n "$WHEELHOUSE_DIR" ] && [ -d "$WHEELHOUSE_DIR" ]; then
    python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" --upgrade pip
  else
    python -m pip install --upgrade pip
  fi
fi

if [ "$SKIP_INSTALL" != "1" ]; then
  if [ "$USE_TPU" = "1" ]; then
    TORCH_VERSION="${TORCH_VERSION:-2.6.0+cpu}"
    TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.6.0}"
    LIBTPU_VERSION="${LIBTPU_VERSION:-}"
    LIBTPU_WHEEL_URL="${LIBTPU_WHEEL_URL:-https://storage.googleapis.com/libtpu-releases/index.html}"
    TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    if [ -n "$WHEELHOUSE_DIR" ] && [ -d "$WHEELHOUSE_DIR" ]; then
      python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" \
        "torch==${TORCH_VERSION}"
      python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" \
        "torch_xla==${TORCH_XLA_VERSION}"
      shopt -s nullglob
      libtpu_wheels=("$WHEELHOUSE_DIR"/libtpu*.whl)
      if [ "${#libtpu_wheels[@]}" -gt 0 ]; then
        python -m pip install --no-index "${libtpu_wheels[@]}"
      fi
      shopt -u nullglob
      if [ -n "$LIBTPU_VERSION" ]; then
        python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" \
          "libtpu==${LIBTPU_VERSION}" tpu-info
      fi
      python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" "numpy<2"
    else
      if [[ "${TORCH_VERSION}" == *+cpu ]]; then
        python -m pip install \
          --extra-index-url "${TORCH_CPU_INDEX_URL}" \
          "torch==${TORCH_VERSION}"
      else
        python -m pip install "torch==${TORCH_VERSION}"
      fi
      python -m pip install \
        "torch_xla[tpu]==${TORCH_XLA_VERSION}" \
        -f "${LIBTPU_WHEEL_URL}"
      python -m pip install "numpy<2"
    fi
  else
    if [ -n "$WHEELHOUSE_DIR" ] && [ -d "$WHEELHOUSE_DIR" ]; then
      python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" torch torchvision torchaudio
    else
      python -m pip install torch torchvision torchaudio
    fi
  fi
  # JAX is required for Pallas flash attention kernels on TPU.
  if [ "$USE_TPU" = "1" ]; then
    JAX_VERSION="${JAX_VERSION:-0.4.38}"
    python -m pip install "jax[tpu]==${JAX_VERSION}" \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  fi
  # v6e chips need libtpu-nightly to avoid NaN after XLA graph caching.
  # Set LIBTPU_NIGHTLY_VERSION to override the libtpu installed by jax[tpu]/torch_xla[tpu].
  if [ "$USE_TPU" = "1" ] && [ -n "${LIBTPU_NIGHTLY_VERSION:-}" ]; then
    python -m pip install --force-reinstall \
      "libtpu-nightly==${LIBTPU_NIGHTLY_VERSION}" \
      -f https://storage.googleapis.com/libtpu-releases/index.html
  fi
  if [ "$USE_TPU" = "1" ]; then
    FILTERED_REQS="$(mktemp)"
    grep -Ev '^(torch|torchvision|torchaudio)([<>=].*)?$' requirements.txt >"$FILTERED_REQS"
    if [ -n "$WHEELHOUSE_DIR" ] && [ -d "$WHEELHOUSE_DIR" ]; then
      python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" -r "$FILTERED_REQS"
    else
      python -m pip install -r "$FILTERED_REQS"
    fi
    rm -f "$FILTERED_REQS"
  else
    if [ -n "$WHEELHOUSE_DIR" ] && [ -d "$WHEELHOUSE_DIR" ]; then
      python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" -r requirements.txt
    else
      python -m pip install -r requirements.txt
    fi
  fi
fi

if [ "$USE_TPU" = "1" ]; then
  export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
  export USE_TORCH_XLA=1
  export EXPDIS_REQUIRE_TPU="${EXPDIS_REQUIRE_TPU:-1}"
  export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v4-8}"
  export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
fi

echo "Environment ready."
echo "python: $(command -v python)"
python -c 'import sys; print(f"python_version={sys.version.split()[0]}")'
python -c 'import torch; print(f"torch={torch.__version__}")' || true
python -c 'import torch_xla; print(f"torch_xla={torch_xla.__version__}")' || true
python -c 'import numpy; print(f"numpy={numpy.__version__}")' || true
python -c 'import wandb; print(f"wandb={wandb.__version__}")' || true
