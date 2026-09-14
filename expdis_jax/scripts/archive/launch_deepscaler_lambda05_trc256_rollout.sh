#!/usr/bin/env bash
# Lambda=0.5 method run on the DeepScaleR data path for comparison to vanilla
# GRPO. The train/serve TPU selection is inherited from launch_dapo_drgrpo_tpu.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.5}"
export DATASET_NAME="${DATASET_NAME:-deepscaler}"
export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-300}"
export RUN_NAME="${RUN_NAME:-deepscaler_drgrpo_novelty05_jaxrepo_$(date +%Y%m%d_%H%M%S)}"

exec "$SCRIPT_DIR/../launch_dapo_drgrpo_tpu.sh" "$@"
