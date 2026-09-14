#!/usr/bin/env bash
# TPU DDP launcher: 4-chip data-parallel via accelerate (no FSDP, no SPMD).
# Usage: bash scripts/run_tpu_ddp_local.sh run_expdis_torch.py [args...]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then set -a; source ".env"; set +a; fi

VENV_DIR="${VENV_DIR:-.venv_tpu_slice}"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ── chip discovery ──────────────────────────────────────────────────
DDP_VISIBLE_CHIPS="${DDP_VISIBLE_CHIPS:-${TPU_VISIBLE_CHIPS:-}}"
if [ -z "$DDP_VISIBLE_CHIPS" ]; then
  DDP_VISIBLE_CHIPS="$(python3 -c "
import glob, os
nodes = sorted(glob.glob('/dev/accel*'))
if not nodes:
    vfio = [p for p in sorted(glob.glob('/dev/vfio/*')) if os.path.basename(p).isdigit()]
    nodes = vfio
count = len(nodes) or 4
print(','.join(str(i) for i in range(count)))
")"
fi
DDP_NUM_PROCESSES="${DDP_NUM_PROCESSES:-$(python3 -c "print(len('$DDP_VISIBLE_CHIPS'.split(',')))")}"

# ── TPU env ─────────────────────────────────────────────────────────
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
# Let torch_xla query GCE metadata for accelerator type (required for tpu.version())
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-0}"

# Process bounds for multi-chip
BOUNDS_MAP="$(python3 -c "
m={1:'1,1,1',2:'2,1,1',3:'3,1,1',4:'2,2,1'}
print(m.get($DDP_NUM_PROCESSES,'$DDP_NUM_PROCESSES,1,1'))
")"
export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-$BOUNDS_MAP}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS:-1,1,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-$TPU_PROCESS_BOUNDS}"
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export EXPDIS_TPU_SLICE_LAUNCHED="${EXPDIS_TPU_SLICE_LAUNCHED:-1}"
export EXPDIS_REQUIRE_TPU="${EXPDIS_REQUIRE_TPU:-1}"
export EXPDIS_USE_XLA_WORLD_SIZE="${EXPDIS_USE_XLA_WORLD_SIZE:-1}"
export EXPDIS_DISABLE_BF16="${EXPDIS_DISABLE_BF16:-1}"

# NOT setting EXPDIS_FSDP — this is pure DDP, not FSDP

# Unset TPU_VISIBLE_CHIPS for multi-process so rank mapping works
if [ "$DDP_NUM_PROCESSES" -gt 1 ]; then
  unset TPU_VISIBLE_CHIPS
fi

if [ -z "${EXPDIS_XLA_PERSISTENT_CACHE_DIR:-}" ]; then
  sanitized="${DDP_VISIBLE_CHIPS//,/}"
  export EXPDIS_XLA_PERSISTENT_CACHE_DIR="/tmp/expdis_xla_cache_ddp_${sanitized}"
fi

# ── accelerate config ───────────────────────────────────────────────
DDP_CONFIG_TEMPLATE="${DDP_CONFIG_TEMPLATE:-configs/accelerate_tpu_ddp4.yaml}"
run_cfg="$(mktemp /tmp/expdis_accelerate_ddp_XXXX.yaml)"
python3 -c "
from pathlib import Path; import re, sys
text = Path('$DDP_CONFIG_TEMPLATE').read_text()
text = re.sub(r'^num_processes:.*$', 'num_processes: $DDP_NUM_PROCESSES', text, flags=re.MULTILINE)
Path('$run_cfg').write_text(text)
"
trap 'rm -f "$run_cfg"' EXIT

echo "[tpu-ddp] chips=$DDP_VISIBLE_CHIPS processes=$DDP_NUM_PROCESSES config=$run_cfg"
echo "[tpu-ddp] PJRT=$PJRT_DEVICE BOUNDS=$TPU_PROCESS_BOUNDS cache=${EXPDIS_XLA_PERSISTENT_CACHE_DIR:-}"

entrypoint="${1:-run_expdis_torch.py}"
shift || true

exec accelerate launch \
  --config_file "$run_cfg" \
  --num_processes "$DDP_NUM_PROCESSES" \
  --tpu \
  --mixed_precision no \
  "$entrypoint" \
  "$@"
