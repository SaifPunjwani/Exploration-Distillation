#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.13.3}"
PYTHON_BIN_CANDIDATE="${PYTHON_BIN_CANDIDATE:-}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
VLLM_ENV_CACHE_GCS_URI="${VLLM_ENV_CACHE_GCS_URI:-}"
VLLM_SKIP_PIP_UPGRADE="${VLLM_SKIP_PIP_UPGRADE:-1}"

log() {
  printf '[bootstrap-vllm] %s\n' "$*"
}

resolve_python_bin() {
  if [ -n "$PYTHON_BIN_CANDIDATE" ] && command -v "$PYTHON_BIN_CANDIDATE" >/dev/null 2>&1; then
    command -v "$PYTHON_BIN_CANDIDATE"
    return 0
  fi

  local candidate=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  echo "No suitable python interpreter found" >&2
  exit 1
}

PYTHON_BIN="$(resolve_python_bin)"

verify_env() {
  [ -x "$VLLM_VENV_DIR/bin/python" ] || return 1
  "$VLLM_VENV_DIR/bin/python" - <<'PY' >/dev/null
import importlib.metadata
import vllm

assert vllm.__version__
assert importlib.metadata.version("tpu_inference")
PY
}

restore_cached_env() {
  [ -n "$VLLM_ENV_CACHE_GCS_URI" ] || return 1
  gcloud storage ls "$VLLM_ENV_CACHE_GCS_URI" >/dev/null 2>&1 || return 1

  log "restoring cached env from $VLLM_ENV_CACHE_GCS_URI"
  local archive_path
  archive_path="$(mktemp /tmp/vllm_tpu_env.XXXXXX.tar.gz)"
  rm -rf "$VLLM_VENV_DIR"
  mkdir -p "$(dirname "$VLLM_VENV_DIR")"
  gcloud storage cp "$VLLM_ENV_CACHE_GCS_URI" "$archive_path" >/dev/null
  tar -xzf "$archive_path" -C "$(dirname "$VLLM_VENV_DIR")"
  rm -f "$archive_path"
  verify_env
}

upload_cached_env() {
  [ -n "$VLLM_ENV_CACHE_GCS_URI" ] || return 0
  log "uploading cached env to $VLLM_ENV_CACHE_GCS_URI"
  local archive_path
  archive_path="$(mktemp /tmp/vllm_tpu_env.XXXXXX.tar.gz)"
  tar -czf "$archive_path" -C "$(dirname "$VLLM_VENV_DIR")" "$(basename "$VLLM_VENV_DIR")"
  gcloud storage cp "$archive_path" "$VLLM_ENV_CACHE_GCS_URI" >/dev/null
  rm -f "$archive_path"
}

if [ "$FORCE_REINSTALL" = "1" ]; then
  rm -rf "$VLLM_VENV_DIR"
fi

if verify_env; then
  log "ready: ${VLLM_VENV_DIR}"
  exit 0
fi

if restore_cached_env; then
  log "ready from cache: ${VLLM_VENV_DIR}"
  exit 0
fi

if [ ! -x "$VLLM_VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VLLM_VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VLLM_VENV_DIR/bin/activate"

if [ "$VLLM_SKIP_PIP_UPGRADE" != "1" ]; then
  python -m pip install --upgrade pip setuptools wheel
fi
python -m pip install "vllm-tpu==${VLLM_TPU_VERSION}" datasets wandb transformers sentencepiece tiktoken

log "verifying env"
python - <<'PY'
import importlib.metadata
import platform
import sys

import vllm

print(f"python={platform.python_version()}")
print(f"vllm={vllm.__version__}")
print(f"tpu_inference={importlib.metadata.version('tpu_inference')}")
print(f"executable={sys.executable}")
PY

upload_cached_env

log "ready: ${VLLM_VENV_DIR}"
