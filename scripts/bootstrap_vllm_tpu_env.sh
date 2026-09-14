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
TMX_ALLOW_GCS_ARTIFACTS="${TMX_ALLOW_GCS_ARTIFACTS:-0}"

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

patch_vllm_tpu_transformers_compat() {
  [ -x "$VLLM_VENV_DIR/bin/python" ] || return 1
  "$VLLM_VENV_DIR/bin/python" - <<'PY'
import py_compile
import sysconfig
from pathlib import Path

purelib = Path(sysconfig.get_paths()["purelib"])
replacements = {
    purelib / "tpu_inference/models/jax/qwen2.py": {
        "from transformers import Qwen2Config, modeling_flax_utils":
            "from transformers import Qwen2Config\nfrom tpu_inference.layers.jax.layers import modeling_flax_utils",
    },
    purelib / "tpu_inference/models/jax/qwen2_5_vl.py": {
        "from transformers import modeling_flax_utils\n":
            "from tpu_inference.layers.jax.layers import modeling_flax_utils\n",
    },
    purelib / "tpu_inference/models/jax/llama3.py": {
        "from transformers import LlamaConfig, modeling_flax_utils":
            "from transformers import LlamaConfig\nfrom tpu_inference.layers.jax.layers import modeling_flax_utils",
    },
}

changed = []
for path, path_replacements in replacements.items():
    if not path.exists():
        continue
    text = path.read_text()
    updated = text
    for old, new in path_replacements.items():
        updated = updated.replace(old, new)
    if updated != text:
        backup = path.with_suffix(path.suffix + ".tmx_transformers_compat_bak")
        if not backup.exists():
            backup.write_text(text)
        path.write_text(updated)
        changed.append(str(path))
    py_compile.compile(str(path), doraise=True)

transformers_base = purelib / "vllm/model_executor/models/transformers/base.py"
if transformers_base.exists():
    text = transformers_base.read_text()
    original_text = text
    marker = "# TMX text-only Gemma compatibility: ignore unsupported audio tower weights."
    suffix_line = "            self.ignore_unexpected_suffixes.append(\".layer_scalar\")\n"
    if marker not in text:
        needle = (
            "        if self.quant_config:\n"
            "            quant_method_name = self.quant_config.get_name()\n"
        )
        patch = (
            f"        {marker}\n"
            "        if getattr(self.config, \"model_type\", \"\") in {\"gemma4\", \"gemma3n\"}:\n"
            "            self.ignore_unexpected_prefixes.extend([\n"
            "                \"model.audio_tower.\",\n"
            "                \"audio_tower.\",\n"
            "            ])\n"
            "            self.ignore_unexpected_suffixes.append(\".layer_scalar\")\n\n"
        )
        if needle not in text:
            raise RuntimeError(f"cannot patch {transformers_base}: quant_config marker not found")
        text = text.replace(needle, patch + needle)
    elif suffix_line not in text:
        suffix_needle = (
            "            self.ignore_unexpected_prefixes.extend([\n"
            "                \"model.audio_tower.\",\n"
            "                \"audio_tower.\",\n"
            "            ])\n"
        )
        if suffix_needle not in text:
            raise RuntimeError(f"cannot patch {transformers_base}: Gemma audio prefix block not found")
        text = text.replace(suffix_needle, suffix_needle + suffix_line)
    if text != original_text:
        backup = transformers_base.with_suffix(transformers_base.suffix + ".tmx_gemma_audio_compat_bak")
        if not backup.exists():
            backup.write_text(original_text)
        transformers_base.write_text(text)
        changed.append(str(transformers_base))
    py_compile.compile(str(transformers_base), doraise=True)

for module in (
    "tpu_inference.models.jax.qwen2",
    "tpu_inference.models.jax.qwen2_5_vl",
    "tpu_inference.models.jax.llama3",
):
    __import__(module)

if changed:
    print("[bootstrap-vllm] patched transformers modeling_flax_utils compatibility: " + ", ".join(changed))
else:
    print("[bootstrap-vllm] transformers modeling_flax_utils compatibility already ok")
PY
}

verify_env() {
  [ -x "$VLLM_VENV_DIR/bin/python" ] || return 1
  patch_vllm_tpu_transformers_compat >/dev/null || return 1
  "$VLLM_VENV_DIR/bin/python" - <<'PY' >/dev/null
import importlib.metadata
import vllm

assert vllm.__version__
assert importlib.metadata.version("tpu_inference")
PY
}

restore_cached_env() {
  [ -n "$VLLM_ENV_CACHE_GCS_URI" ] || return 1
  if [ "$TMX_ALLOW_GCS_ARTIFACTS" != "1" ]; then
    log "skipping GCS env cache restore; set TMX_ALLOW_GCS_ARTIFACTS=1 only for explicit legacy use"
    return 1
  fi
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
  if [ "$TMX_ALLOW_GCS_ARTIFACTS" != "1" ]; then
    log "skipping GCS env cache upload; Hugging Face is the artifact backend"
    return 0
  fi
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
patch_vllm_tpu_transformers_compat

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
