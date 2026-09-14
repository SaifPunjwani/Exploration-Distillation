#!/usr/bin/env bash
# Lambda=0.75 side run: owned cyber v5lite64 train + owned saif v5lite64 rollout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_TPU_PROJECT="${TRAIN_TPU_PROJECT:?set TRAIN_TPU_PROJECT to the GCP project id of the training TPU}"
export TRAIN_ZONE="${TRAIN_ZONE:-us-central1-a}"
export TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
export TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-64}"
export EXPDIS_TPU_HOST_BOUNDS="${EXPDIS_TPU_HOST_BOUNDS:-4,4,1}"
export EXPDIS_EXPECTED_TRAIN_ACCELERATOR="${EXPDIS_EXPECTED_TRAIN_ACCELERATOR:-v5litepod-64}"

export SERVE_TPU_PROJECT="${SERVE_TPU_PROJECT:?set SERVE_TPU_PROJECT to the GCP project id of the serving TPU}"
export SERVE_ZONE="${SERVE_ZONE:-us-central1-a}"
export SERVE_TPU_NAME="${SERVE_TPU_NAME:?set SERVE_TPU_NAME to the serving (vLLM) TPU VM (slice) name}"
export SERVE_WORKERS="${SERVE_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
export EXPDIS_EXPECTED_SERVE_ACCELERATOR="${EXPDIS_EXPECTED_SERVE_ACCELERATOR:-v5litepod-64}"

export LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.75}"
export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-300}"
export GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-4}"
export GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
export RUN_NAME="${RUN_NAME:-dapo16k_drgrpo_novelty075_uc1a_v5lite64_pair_$(date +%Y%m%d_%H%M%S)}"

# Keep rollout fanout stochastic. Duplicate fanout collapses group variance and
# makes dynamic sampling spend entire attempts on uninformative candidates.
export EXPDIS_VLLM_RANDOMIZE_SEEDS="${EXPDIS_VLLM_RANDOMIZE_SEEDS:-1}"
export EXPDIS_VLLM_SEED_MODE="${EXPDIS_VLLM_SEED_MODE:-prompt_nonce}"
export VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-4}"
export DYNAMIC_SAMPLING_MAX_ATTEMPTS="${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-2}"

# This US slice is heterogeneous: workers 0-7 have expdis_jax conda, while
# workers 8-15 only have vllm_tpu_env. The latter has the full train deps and
# libtpu on every worker, so pin it for a consistent 16-host JAX world.
export ENV_DIR="${ENV_DIR:?set ENV_DIR to the Python env directory present on every worker (e.g. /home/<user>/vllm_tpu_env)}"

# These side lanes run from a local control plane. Use TPU-internal fanout for
# payload/script distribution, but keep process launches sequential locally.
export EXPDIS_VLLM_BATCHED_SLICE_BOOTSTRAP="${EXPDIS_VLLM_BATCHED_SLICE_BOOTSTRAP:-0}"
export EXPDIS_VLLM_PARALLEL_LAUNCH="${EXPDIS_VLLM_PARALLEL_LAUNCH:-0}"
export EXPDIS_FAST_PACKAGE_FANOUT="${EXPDIS_FAST_PACKAGE_FANOUT:-1}"
export EXPDIS_FAST_WORKER_SCRIPT_FANOUT="${EXPDIS_FAST_WORKER_SCRIPT_FANOUT:-1}"
export EXPDIS_PARALLEL_TRAIN_LAUNCH="${EXPDIS_PARALLEL_TRAIN_LAUNCH:-0}"
export EXPDIS_VLLM_RELOAD_PARALLELISM="${EXPDIS_VLLM_RELOAD_PARALLELISM:-1}"
export EXPDIS_VLLM_HEALTH_WAIT_SECONDS="${EXPDIS_VLLM_HEALTH_WAIT_SECONDS:-1200}"

exec "$SCRIPT_DIR/../launch_dapo_drgrpo_tpu.sh" "$@"
