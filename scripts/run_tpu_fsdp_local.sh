#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  source ".env"
  set +a
fi

VENV_DIR="${VENV_DIR:-.venv_tpu_slice}"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

default_visible_chips() {
  python3 - <<'PY'
import glob
import os

nodes = sorted(glob.glob("/dev/accel*"))
if not nodes:
    vfio = []
    for path in sorted(glob.glob("/dev/vfio/*")):
        name = os.path.basename(path)
        if name.isdigit():
            vfio.append(path)
    nodes = vfio
count = len(nodes) or 4
print(",".join(str(i) for i in range(count)))
PY
}

count_visible_chips() {
  python3 - "$1" <<'PY'
import sys

items = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
print(len(items) if items else 0)
PY
}

infer_accelerator_type() {
  python3 - <<'PY'
from tmx.devices import _infer_tpu_accelerator_type

print(_infer_tpu_accelerator_type() or "")
PY
}

normalize_chip_list() {
  python3 - "$1" <<'PY'
import sys

items = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
print(",".join(items))
PY
}

infer_process_bounds() {
  python3 - "$1" <<'PY'
import sys

count = int(sys.argv[1])
mapping = {
    1: "1,1,1",
    2: "2,1,1",
    3: "3,1,1",
    4: "2,2,1",
}
print(mapping.get(count, f"{count},1,1"))
PY
}

FSDP_MODE="${FSDP_MODE:-train}"
FSDP_CONFIG_TEMPLATE="${FSDP_CONFIG_TEMPLATE:-configs/accelerate_fsdp_tpu4.yaml}"
FSDP_MIXED_PRECISION="${FSDP_MIXED_PRECISION:-no}"
FSDP_DOWNCAST_BF16="${FSDP_DOWNCAST_BF16:-no}"
FSDP_VISIBLE_CHIPS="${FSDP_VISIBLE_CHIPS:-${TPU_VISIBLE_CHIPS:-}}"
if [ -z "$FSDP_VISIBLE_CHIPS" ]; then
  FSDP_VISIBLE_CHIPS="$(default_visible_chips)"
fi
FSDP_NUM_PROCESSES="${FSDP_NUM_PROCESSES:-}"
if [ -z "$FSDP_NUM_PROCESSES" ]; then
  FSDP_NUM_PROCESSES="$(count_visible_chips "$FSDP_VISIBLE_CHIPS")"
fi
normalized_chip_list="$(normalize_chip_list "$FSDP_VISIBLE_CHIPS")"
expected_full_host_list="$(python3 - "$FSDP_NUM_PROCESSES" <<'PY'
import sys

count = int(sys.argv[1])
print(",".join(str(i) for i in range(count)))
PY
)"

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
if [ -z "${TPU_ACCELERATOR_TYPE:-}" ]; then
  inferred_accelerator_type="$(infer_accelerator_type)"
  if [ -n "$inferred_accelerator_type" ]; then
    export TPU_ACCELERATOR_TYPE="$inferred_accelerator_type"
  fi
fi
export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-$(infer_process_bounds "$FSDP_NUM_PROCESSES")}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS:-1,1,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-$TPU_PROCESS_BOUNDS}"
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TMX_TPU_SLICE_LAUNCHED="${TMX_TPU_SLICE_LAUNCHED:-1}"
export TMX_REQUIRE_TPU="${TMX_REQUIRE_TPU:-1}"
export TMX_FSDP="${TMX_FSDP:-1}"
export TMX_USE_XLA_WORLD_SIZE="${TMX_USE_XLA_WORLD_SIZE:-1}"
export TMX_DISABLE_BF16="${TMX_DISABLE_BF16:-1}"

if [ "$FSDP_NUM_PROCESSES" -gt 1 ] && [ "$normalized_chip_list" != "$expected_full_host_list" ]; then
  echo "[tpu-fsdp] partial/non-zero-based multi-process chip list is not supported by this launcher: $normalized_chip_list"
  echo "[tpu-fsdp] expected full-host contiguous mapping for $FSDP_NUM_PROCESSES processes: $expected_full_host_list"
  exit 2
fi
if [ "$FSDP_NUM_PROCESSES" -gt 1 ] && [ "$normalized_chip_list" = "$expected_full_host_list" ]; then
  unset TPU_VISIBLE_CHIPS
else
  export TPU_VISIBLE_CHIPS="$FSDP_VISIBLE_CHIPS"
fi

if [ -z "${TMX_XLA_PERSISTENT_CACHE_DIR:-}" ]; then
  sanitized_chips="${FSDP_VISIBLE_CHIPS//,/}"
  export TMX_XLA_PERSISTENT_CACHE_DIR="/tmp/tmx_xla_cache_fsdp_${sanitized_chips}"
fi

run_cfg="$(mktemp /tmp/tmx_accelerate_fsdp_XXXX.yaml)"
python3 - "$FSDP_CONFIG_TEMPLATE" "$run_cfg" "$FSDP_NUM_PROCESSES" "$FSDP_MIXED_PRECISION" "$FSDP_DOWNCAST_BF16" <<'PY'
from pathlib import Path
import re
import sys

src, dst, num_processes, mixed_precision, downcast_bf16 = sys.argv[1:]
text = Path(src).read_text()
text = re.sub(r"^num_processes:.*$", f"num_processes: {num_processes}", text, flags=re.MULTILINE)
text = re.sub(r"^mixed_precision:.*$", f"mixed_precision: '{mixed_precision}'", text, flags=re.MULTILINE)
text = re.sub(r"^downcast_bf16:.*$", f"downcast_bf16: '{downcast_bf16}'", text, flags=re.MULTILINE)
Path(dst).write_text(text)
PY
trap 'rm -f "$run_cfg"' EXIT

echo "[tpu-fsdp] mode=$FSDP_MODE chips=$FSDP_VISIBLE_CHIPS processes=$FSDP_NUM_PROCESSES config=$run_cfg"
echo "[tpu-fsdp] PJRT_DEVICE=$PJRT_DEVICE TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS:-<rank-mapped>} cache=${TMX_XLA_PERSISTENT_CACHE_DIR:-}"

entrypoint="${1:-main.py}"
if [ "$#" -gt 0 ]; then
  shift
fi
if [ "$FSDP_MODE" = "smoke" ]; then
  entrypoint="scripts/tpu_fsdp_smoke.py"
  set --
fi

exec accelerate launch \
  --config_file "$run_cfg" \
  --num_processes "$FSDP_NUM_PROCESSES" \
  --tpu \
  --mixed_precision "$FSDP_MIXED_PRECISION" \
  "$entrypoint" \
  "$@"
