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

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/offline_tpu_assets_$(date +%Y%m%d_%H%M%S)}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$OUTPUT_ROOT/wheelhouse}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$OUTPUT_ROOT/hf_cache}"
WHEELHOUSE_TAR="${WHEELHOUSE_TAR:-$OUTPUT_ROOT/wheelhouse.tar}"
HF_CACHE_TAR="${HF_CACHE_TAR:-$OUTPUT_ROOT/hf_cache.tar}"
BUILD_WHEELHOUSE="${BUILD_WHEELHOUSE:-1}"
BUILD_HF_CACHE="${BUILD_HF_CACHE:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.6.0}"
TORCH_WHEEL_URL="${TORCH_WHEEL_URL:-}"
TORCH_XLA_WHEEL_URL="${TORCH_XLA_WHEEL_URL:-https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-${TORCH_XLA_VERSION}-cp310-cp310-manylinux_2_28_x86_64.whl}"
PYTHON_VERSION="${PYTHON_VERSION:-310}"
PYTHON_ABI="${PYTHON_ABI:-cp310}"
PLATFORM_TAG="${PLATFORM_TAG:-manylinux2014_x86_64}"
INCLUDE_TPU_INFO="${INCLUDE_TPU_INFO:-1}"
INCLUDE_LIBTPU_VERSION="${INCLUDE_LIBTPU_VERSION:-}"
LIBTPU_NIGHTLY_WHEEL_URL="${LIBTPU_NIGHTLY_WHEEL_URL:-}"
MODEL_SPECS="${MODEL_SPECS:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
DATASET_SPECS="${DATASET_SPECS:-openai/gsm8k::main::train}"

mkdir -p "$OUTPUT_ROOT" "$WHEELHOUSE_DIR" "$HF_CACHE_ROOT"

FILTERED_REQS="$(mktemp)"
grep -Ev '^(torch|torchvision|torchaudio)([<>=].*)?$' requirements.txt >"$FILTERED_REQS"

download_wheels() {
  "$PYTHON_BIN" -m pip download \
    --dest "$WHEELHOUSE_DIR" \
    --only-binary=:all: \
    --platform "$PLATFORM_TAG" \
    --implementation cp \
    --python-version "$PYTHON_VERSION" \
    --abi "$PYTHON_ABI" \
    "$@"
}

download_requires_from_wheel() {
  "$PYTHON_BIN" - <<'PY' "$1" "$PYTHON_BIN" "$WHEELHOUSE_DIR" "$PLATFORM_TAG" "$PYTHON_VERSION" "$PYTHON_ABI"
import pathlib
import re
import subprocess
import sys
import zipfile
from packaging.requirements import Requirement

wheel_path = pathlib.Path(sys.argv[1])
python_bin = sys.argv[2]
wheelhouse_dir = sys.argv[3]
platform_tag = sys.argv[4]
python_version = sys.argv[5]
python_abi = sys.argv[6]

requirements = []
with zipfile.ZipFile(wheel_path) as zf:
    meta_name = next(name for name in zf.namelist() if name.endswith("METADATA"))
    metadata = zf.read(meta_name).decode("utf-8", errors="replace")

for line in metadata.splitlines():
    if not line.startswith("Requires-Dist:"):
        continue
    req_line = line.split(":", 1)[1].strip()
    req = Requirement(req_line)
    if req.marker:
        marker = str(req.marker)
        if 'extra ==' in marker:
            continue
        if 'python_version < "3.10"' in marker or "python_version < '3.10'" in marker:
            continue
    normalized = req.name + str(req.specifier)
    if normalized:
        requirements.append(normalized)

if requirements:
    cmd = [
        python_bin, "-m", "pip", "download",
        "--dest", wheelhouse_dir,
        "--only-binary=:all:",
        "--platform", platform_tag,
        "--implementation", "cp",
        "--python-version", python_version,
        "--abi", python_abi,
        *requirements,
    ]
    subprocess.check_call(cmd)
PY
}

resolve_wheelhouse_closure() {
  "$PYTHON_BIN" - <<'PY' "$PYTHON_BIN" "$WHEELHOUSE_DIR" "$PLATFORM_TAG" "$PYTHON_VERSION" "$PYTHON_ABI"
import email
import os
import pathlib
import subprocess
import sys
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

python_bin = sys.argv[1]
wheelhouse_dir = pathlib.Path(sys.argv[2])
platform_tag = sys.argv[3]
python_version = sys.argv[4]
python_abi = sys.argv[5]

env = default_environment()
env.update({
    "python_version": f"{python_version[0]}.{python_version[1:]}",
    "python_full_version": f"{python_version[0]}.{python_version[1:]}.0",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "implementation_name": "cpython",
})

def parse_name_version(filename: str):
    stem = filename[:-4] if filename.endswith(".whl") else filename
    parts = stem.split("-")
    if len(parts) < 2:
        return None, None
    return canonicalize_name(parts[0]), parts[1]

def installed_names():
    names = set()
    for wheel in wheelhouse_dir.glob("*.whl"):
        name, _ = parse_name_version(wheel.name)
        if name:
            names.add(name)
    return names

for _ in range(12):
    present = installed_names()
    missing = []
    seen = set()
    for wheel in wheelhouse_dir.glob("*.whl"):
        with zipfile.ZipFile(wheel) as zf:
            meta_name = next((n for n in zf.namelist() if n.endswith("METADATA")), None)
            if not meta_name:
                continue
            metadata = email.message_from_bytes(zf.read(meta_name))
        for raw_req in metadata.get_all("Requires-Dist", []):
            req = Requirement(raw_req)
            if req.marker and not req.marker.evaluate(env):
                continue
            dep_name = canonicalize_name(req.name)
            if dep_name in present or dep_name in seen:
                continue
            missing.append(req.name + str(req.specifier))
            seen.add(dep_name)
    if not missing:
        break
    cmd = [
        python_bin, "-m", "pip", "download",
        "--dest", str(wheelhouse_dir),
        "--only-binary=:all:",
        "--platform", platform_tag,
        "--implementation", "cp",
        "--python-version", python_version,
        "--abi", python_abi,
        *missing,
    ]
    subprocess.check_call(cmd)
else:
    raise SystemExit("wheelhouse closure did not converge")
PY
}

decoded_wheel_name() {
  "$PYTHON_BIN" - <<'PY' "$1"
import os
import sys
from urllib.parse import unquote, urlparse

url = sys.argv[1]
path = urlparse(url).path or url
print(unquote(os.path.basename(path)))
PY
}

echo "[assets] building wheelhouse at $WHEELHOUSE_DIR"
if [ "$BUILD_WHEELHOUSE" = "1" ]; then
  if [ -n "$TORCH_WHEEL_URL" ]; then
    torch_wheel_name="$(decoded_wheel_name "$TORCH_WHEEL_URL")"
    curl -L --fail -o "$WHEELHOUSE_DIR/$torch_wheel_name" "$TORCH_WHEEL_URL"
  else
    download_wheels "torch==${TORCH_VERSION}"
  fi
  torch_xla_wheel_name="$(decoded_wheel_name "$TORCH_XLA_WHEEL_URL")"
  curl -L --fail -o "$WHEELHOUSE_DIR/$torch_xla_wheel_name" "$TORCH_XLA_WHEEL_URL"
  download_requires_from_wheel "$WHEELHOUSE_DIR/$torch_xla_wheel_name"
  download_wheels -r "$FILTERED_REQS"
  if [ -n "${torch_wheel_name:-}" ]; then
    find "$WHEELHOUSE_DIR" -maxdepth 1 -type f -name 'torch-*.whl' ! -name "$torch_wheel_name" -delete
  fi
  if [ "$INCLUDE_TPU_INFO" = "1" ]; then
    download_wheels tpu-info
  fi
  if [ -n "$INCLUDE_LIBTPU_VERSION" ]; then
    download_wheels -f https://storage.googleapis.com/libtpu-wheels/index.html "libtpu==${INCLUDE_LIBTPU_VERSION}"
  fi
  if [ -n "$LIBTPU_NIGHTLY_WHEEL_URL" ]; then
    libtpu_nightly_wheel_name="$(decoded_wheel_name "$LIBTPU_NIGHTLY_WHEEL_URL")"
    curl -L --fail -o "$WHEELHOUSE_DIR/$libtpu_nightly_wheel_name" "$LIBTPU_NIGHTLY_WHEEL_URL"
  fi
  resolve_wheelhouse_closure
  echo "[assets] packing wheelhouse to $WHEELHOUSE_TAR"
  tar -cf "$WHEELHOUSE_TAR" -C "$WHEELHOUSE_DIR" .
fi
rm -f "$FILTERED_REQS"

if [ "$BUILD_HF_CACHE" = "1" ]; then
  echo "[assets] building huggingface cache at $HF_CACHE_ROOT"
  HF_HOME="$HF_CACHE_ROOT" HF_HUB_CACHE="$HF_CACHE_ROOT/hub" HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets" TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers" \
    "$PYTHON_BIN" - <<'PY' "$MODEL_SPECS" "$DATASET_SPECS"
import os
import sys

from datasets import load_dataset
from huggingface_hub import snapshot_download

model_specs = [s for s in sys.argv[1].split(";") if s]
dataset_specs = [s for s in sys.argv[2].split(";") if s]
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")

for model_id in model_specs:
    print(f"[assets] model {model_id}", flush=True)
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        token=hf_token,
        resume_download=True,
        local_files_only=False,
    )

for spec in dataset_specs:
    parts = spec.split("::")
    while len(parts) < 3:
        parts.append("")
    name, config, split = parts[:3]
    kwargs = {}
    if config:
        kwargs["name"] = config
    if split:
        kwargs["split"] = split
    print(f"[assets] dataset {name} config={config or '-'} split={split or '-'}", flush=True)
    load_dataset(name, **kwargs)
PY
  echo "[assets] packing hf cache to $HF_CACHE_TAR"
  tar -cf "$HF_CACHE_TAR" -C "$HF_CACHE_ROOT" .
fi

echo "[assets] complete"
echo "[assets] wheelhouse_dir=$WHEELHOUSE_DIR"
echo "[assets] wheelhouse_tar=$WHEELHOUSE_TAR"
echo "[assets] hf_cache_root=$HF_CACHE_ROOT"
echo "[assets] hf_cache_tar=$HF_CACHE_TAR"
