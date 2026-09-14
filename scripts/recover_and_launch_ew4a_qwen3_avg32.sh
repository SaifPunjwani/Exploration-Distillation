#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-a}"
RUN_PREFIX="${RUN_PREFIX:-ew4a_qwen3_1p7b_avg32_resume_20260401_2300}"
POLL_SECONDS="${POLL_SECONDS:-20}"
PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"

ensure_qr_exists() {
  local qr_state
  local node_state
  qr_state="$(gcloud alpha compute tpus queued-resources describe "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --format='value(state.state)' 2>/dev/null || true)"
  node_state="$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --format='value(state)' 2>/dev/null || true)"
  if [ -z "$qr_state" ] && [ -z "$node_state" ]; then
    ACCELERATOR_TYPE=v6e-64 PROJECT="$PROJECT" \
      gcloud alpha compute tpus queued-resources create "$TPU_NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --accelerator-type=v6e-64 \
      --runtime-version=v2-alpha-tpuv6e \
      --node-id="$TPU_NAME" \
      --provisioning-model=SPOT \
      --internal-ips \
      --network=default \
      --subnetwork=default
  fi
}

ensure_qr_exists

while true; do
  qr_state="$(gcloud alpha compute tpus queued-resources describe "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --format='value(state.state)' 2>/dev/null || true)"
  state="$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --format='value(state)' 2>/dev/null || true)"
  health="$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --format='value(health)' 2>/dev/null || true)"
  printf '[%s] qr_state=%s state=%s health=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${qr_state:-NONE}" "${state:-NONE}" "${health:-NONE}"
  if [ "$state" = "READY" ] && [ "$health" = "HEALTHY" ]; then
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    RUN_PREFIX="$RUN_PREFIX" \
    MODEL_NAME='Qwen/Qwen3-1.7B' \
    SERVED_MODEL_NAME='Qwen/Qwen3-1.7B' \
    TOKENIZER_NAME='Qwen/Qwen3-1.7B' \
    BENCHMARK_DATASET_NAME='aime_2024' \
    MAX_BENCHMARK=30 \
    BENCHMARK_NUM_ROLLOUTS=32 \
    BENCHMARK_ROLLOUT_TEMPERATURE=1.0 \
    BENCHMARK_ROLLOUT_TOP_P=1.0 \
    BENCHMARK_ROLLOUT_TOP_K=0 \
    BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN=16384 \
    BENCHMARK_ROLLOUT_PASS_K='1,4,8,16,32' \
    REQUEST_CONCURRENCY=32 \
    USE_WANDB=1 \
    ALLOW_DIRTY=1 \
    bash scripts/launch_ew4a_qwen3_avg32_baselines.sh
    break
  fi
  ensure_qr_exists
  sleep "$POLL_SECONDS"
done
