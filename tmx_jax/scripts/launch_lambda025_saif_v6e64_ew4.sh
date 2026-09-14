#!/usr/bin/env bash
# Lambda=0.25 side run: owned saif v6e-64 train + owned saif v5lite64 rollout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_TPU_PROJECT="${TRAIN_TPU_PROJECT:?set TRAIN_TPU_PROJECT to the GCP project id of the training TPU}"
export TRAIN_ZONE="${TRAIN_ZONE:-europe-west4-a}"
export TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
export TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-64}"
export TMX_TPU_HOST_BOUNDS="${TMX_TPU_HOST_BOUNDS:-4,4,1}"
export TMX_EXPECTED_TRAIN_ACCELERATOR="${TMX_EXPECTED_TRAIN_ACCELERATOR:-v6e-64}"

export SERVE_TPU_PROJECT="${SERVE_TPU_PROJECT:?set SERVE_TPU_PROJECT to the GCP project id of the serving TPU}"
export SERVE_ZONE="${SERVE_ZONE:-europe-west4-b}"
export SERVE_TPU_NAME="${SERVE_TPU_NAME:?set SERVE_TPU_NAME to the serving (vLLM) TPU VM (slice) name}"
export SERVE_WORKERS="${SERVE_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
export TMX_EXPECTED_SERVE_ACCELERATOR="${TMX_EXPECTED_SERVE_ACCELERATOR:-v5litepod-64}"

export LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.25}"
export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-300}"
export GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-4}"
export GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
export RUN_NAME="${RUN_NAME:-dapo16k_drgrpo_novelty025_saif_v6e64_ew4_$(date +%Y%m%d_%H%M%S)}"

# Fanout must produce independent samples; otherwise DAPO sees duplicate
# completions, zero reward variance, and wastes full attempts before fallback.
export TMX_VLLM_RANDOMIZE_SEEDS="${TMX_VLLM_RANDOMIZE_SEEDS:-1}"
export TMX_VLLM_SEED_MODE="${TMX_VLLM_SEED_MODE:-prompt_nonce}"
export VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-4}"
export DYNAMIC_SAMPLING_MAX_ATTEMPTS="${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-2}"

# These side lanes run from a local control plane. Use TPU-internal fanout for
# payload/script distribution, but keep process launches sequential locally.
export TMX_VLLM_BATCHED_SLICE_BOOTSTRAP="${TMX_VLLM_BATCHED_SLICE_BOOTSTRAP:-0}"
export TMX_VLLM_PARALLEL_LAUNCH="${TMX_VLLM_PARALLEL_LAUNCH:-0}"
export TMX_FAST_PACKAGE_FANOUT="${TMX_FAST_PACKAGE_FANOUT:-1}"
export TMX_FAST_WORKER_SCRIPT_FANOUT="${TMX_FAST_WORKER_SCRIPT_FANOUT:-1}"
export TMX_PARALLEL_TRAIN_LAUNCH="${TMX_PARALLEL_TRAIN_LAUNCH:-0}"
export TMX_VLLM_RELOAD_PARALLELISM="${TMX_VLLM_RELOAD_PARALLELISM:-1}"
export TMX_VLLM_HEALTH_WAIT_SECONDS="${TMX_VLLM_HEALTH_WAIT_SECONDS:-1200}"

# Full 16k completions make row-wise GRPO updates prohibitively slow. On the
# v6e-64 slice we need to use the whole 64-row rollout batch as the vectorized
# microbatch; smaller chunks still spend minutes in the first backward pass.
# Override this if the allocation reports HBM pressure.
export TMX_VECTORIZE_GRPO_ROWS="${TMX_VECTORIZE_GRPO_ROWS:-1}"
export TMX_GRPO_ROW_VECTOR_CHUNK="${TMX_GRPO_ROW_VECTOR_CHUNK:-64}"
export LOGPROB_CHUNK_TOKENS="${LOGPROB_CHUNK_TOKENS:-512}"

exec "$SCRIPT_DIR/launch_dapo_drgrpo_tpu.sh" "$@"
