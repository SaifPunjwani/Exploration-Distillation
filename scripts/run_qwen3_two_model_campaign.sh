#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAMPAIGN_TAG="${CAMPAIGN_TAG:-qwen3_two_model_campaign_$(date +%Y%m%d_%H%M%S)}"

export CAMPAIGN_TAG
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-configs/qwen3_two_model_campaign.tsv}"
export RAW_HERD_MANIFEST="${RAW_HERD_MANIFEST:-configs/qwen3_aime24_raw.tsv}"
export TRAIN_INLINE_AIME_BENCHMARK="${TRAIN_INLINE_AIME_BENCHMARK:-1}"
export RESUME_AFTER_CANARY="${RESUME_AFTER_CANARY:-1}"
export TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-dapo_math_17k}"
export AIME24_MAX_EXAMPLES="${AIME24_MAX_EXAMPLES:-30}"
export AIME_NUM_ROLLOUTS="${AIME_NUM_ROLLOUTS:-32}"
export AIME_TEMPERATURE="${AIME_TEMPERATURE:-1.0}"
export AIME_TOP_P="${AIME_TOP_P:-1.0}"
export AIME_PASS_K="${AIME_PASS_K:-1,4,8,16,32}"
export AIME_MAX_COMPLETION_LEN="${AIME_MAX_COMPLETION_LEN:-16384}"
export RAW_HERD_WORKER="${RAW_HERD_WORKER:-8}"
export TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3 4 5 6 7 8}"
export BOOTSTRAP_CONCURRENCY="${BOOTSTRAP_CONCURRENCY:-1}"

echo "[qwen3-campaign] launching with CAMPAIGN_TAG=$CAMPAIGN_TAG"
echo "[qwen3-campaign] train manifest: $TRAIN_MANIFEST"
echo "[qwen3-campaign] raw herd manifest: $RAW_HERD_MANIFEST"
exec bash scripts/run_use1d_full_campaign.sh
