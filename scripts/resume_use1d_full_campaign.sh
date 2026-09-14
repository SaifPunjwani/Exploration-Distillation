#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CAMPAIGN_TAG="${CAMPAIGN_TAG:-use1d_six_model_campaign_20260331_211041}"
export RESUME_AFTER_CANARY="${RESUME_AFTER_CANARY:-1}"
export TRAIN_WORKERS="${TRAIN_WORKERS:-1 2 3 4 5 6 7 8}"
export BOOTSTRAP_CONCURRENCY="${BOOTSTRAP_CONCURRENCY:-4}"
export DISPATCH_ONLY="${DISPATCH_ONLY:-1}"
export TRAIN_INLINE_AIME_BENCHMARK="${TRAIN_INLINE_AIME_BENCHMARK:-0}"

bash scripts/run_use1d_full_campaign.sh "$@"
