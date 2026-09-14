#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-a}"
RUN_NAME="${RUN_NAME:-jax_lambda0_v6e_singlehost_$(date +%Y%m%d_%H%M%S)}"

export TPU_NAME="$TPU_NAME"
export ZONE="$ZONE"
export TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
export SERVE_WORKERS="${SERVE_WORKERS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-64}"
export RUN_NAME="$RUN_NAME"

exec "$SCRIPT_DIR/launch_jax_multihost_v5lite.sh"
