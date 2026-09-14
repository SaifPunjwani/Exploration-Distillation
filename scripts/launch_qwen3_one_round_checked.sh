#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ALLOW_DIRTY="${ALLOW_DIRTY:-1}"
CLEAN_BEFORE_LAUNCH="${CLEAN_BEFORE_LAUNCH:-1}"
REMOTE_STATE_TARGET="${REMOTE_STATE_TARGET:-all}"

echo "[qwen3-serious] validating local serious-run path"
bash scripts/validate_qwen3_serious_run.sh

echo "[qwen3-serious] current remote state"
TARGET="$REMOTE_STATE_TARGET" bash scripts/qwen3_serious_remote_state.sh check

if [ "$CLEAN_BEFORE_LAUNCH" = "1" ]; then
  echo "[qwen3-serious] cleaning dedicated remote hosts before launch"
  CONFIRM_REMOTE_CLEANUP=1 TARGET="$REMOTE_STATE_TARGET" bash scripts/qwen3_serious_remote_state.sh clean
  echo "[qwen3-serious] remote state after cleanup"
  TARGET="$REMOTE_STATE_TARGET" bash scripts/qwen3_serious_remote_state.sh check
fi

echo "[qwen3-serious] launching one-round full DAPO run"
ALLOW_DIRTY="$ALLOW_DIRTY" bash scripts/launch_qwen3_one_round_full_dapo.sh
