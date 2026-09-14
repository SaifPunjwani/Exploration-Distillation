#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
export ZONE="${ZONE:-europe-west4-b}"
export TPU_PROXY_JUMP_ALIAS="${TPU_PROXY_JUMP_ALIAS:-none}"
export RUN_PREFIX="${RUN_PREFIX:-ew4b_qwen3_1p7b_avg32_resume}"
export WORKERS_THINK="${WORKERS_THINK:-1,2,3,4,5,6,7}"
export WORKERS_NOTHINK="${WORKERS_NOTHINK:-8,9,10,11,12,13,14,15}"
export CLEANUP_STALE="${CLEANUP_STALE:-0}"

exec bash scripts/launch_ew4b_qwen3_avg32_baselines.sh
