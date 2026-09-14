#!/usr/bin/env bash
# Convenience entrypoint for the lambda=0.5 Explorer continuation from HF step
# 100. Set TRAIN_TPU_* to an owned nanny TPU to train there; otherwise this
# falls back to a TRC v5litepod-256 train/serve partition.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.5}"
export SOURCE_RUN="${SOURCE_RUN:-dapo16k_drgrpo_novelty05_hf_resume49_20260429_174651_wandbresume_parallelvllm}"
export SOURCE_STEP="${SOURCE_STEP:-000100}"
export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-300}"
export RUN_NAME="${RUN_NAME:-dapo16k_drgrpo_novelty05_jaxrepo_resume100_$(date +%Y%m%d_%H%M%S)}"

exec "$SCRIPT_DIR/launch_dapo_drgrpo_tpu.sh" "$@"
