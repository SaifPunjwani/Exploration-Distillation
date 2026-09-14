#!/usr/bin/env bash
# Lambda=0.25 Explorer production entrypoint. Set TRAIN_TPU_* to an owned
# nanny-created train TPU to use it; otherwise runs on the TRC v5litepod-256
# train/serve partition.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.25}"
export SOURCE_STEP="${SOURCE_STEP:-000100}"
export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-300}"
export RUN_NAME="${RUN_NAME:-dapo16k_drgrpo_novelty025_jaxrepo_$(date +%Y%m%d_%H%M%S)}"

exec "$SCRIPT_DIR/launch_dapo_drgrpo_tpu.sh" "$@"
