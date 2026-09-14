#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

lane="${1:?usage: $0 lambda05_extend|lambda025_trc|deepscaler_lambda05|lambda075_cyber}"
stamp="$(date +%Y%m%d_%H%M%S)"

export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-$HOME/.config/gcloud}"
# Optional: pin the gcloud account; unset = the active gcloud account.
if [ -n "${CLOUDSDK_AUTH_ACCOUNT:-}" ]; then export CLOUDSDK_AUTH_ACCOUNT; fi

common_env() {
  export MODEL_NAME=Qwen/Qwen3-1.7B
  export DATASET_NAME="${DATASET_NAME:-dapo_math_17k}"
  export TMX_FRESH_RUN="${TMX_FRESH_RUN:-1}"
  export INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
  export TMX_PIPELINE_STAGE=explorer_novelty
  export TMX_PIPELINE_STOP_AFTER_EXPLORER=1
  export MAX_PROMPT_LEN=2048
  export MAX_COMPLETION_LEN=16384
  export MAX_TOTAL_LEN=18432
  export SOFT_OVERLONG_EXPECTED_LEN=13107
  export SOFT_OVERLONG_CACHE_LEN=3277
  export GRPO_NUM_GENERATIONS=16
  export GRPO_UPDATES_PER_ROLLOUT=1
  export GRPO_LR=5e-6
  export GRPO_CLIP_EPSILON=0.2
  export GRPO_CLIP_EPSILON_HIGH=0.28
  export GRPO_KL_BETA=0.0
  export GRPO_ADVANTAGE_NORMALIZATION=none
  export DYNAMIC_SAMPLING=true
  export DYNAMIC_SAMPLING_MAX_ATTEMPTS="${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-2}"
  export DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-2}"
  export TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="$DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT"
  export TMX_DYNAMIC_SAMPLING_ACCEPT_REWARD_VARIANCE=1
  export MASK_TRUNCATED_COMPLETIONS=false
  export REWARD_CORRECT=1.0
  export REWARD_INCORRECT=-1.0
  export FORMAT_PENALTY_WEIGHT=0.0
  export NONTERMINATION_PENALTY=0.0
  export INVALID_ANSWER_PENALTY=0.0
  export INCORRECT_NOVELTY_SCALE=1.0
  export VLLM_MAX_MODEL_LEN=18432
  export VLLM_MAX_NUM_BATCHED_TOKENS=18432
  export VLLM_MAX_NUM_SEQS=1
  export VLLM_TP_SIZE=4
  export VLLM_FANOUT_PER_PROMPT=16
  export TMX_VLLM_ENDPOINT_MAX_INFLIGHT=2
  export TMX_VLLM_RANDOMIZE_SEEDS=1
  export TMX_VLLM_SEED_MODE=prompt_nonce
  export TMX_VLLM_SEED_BASE="$RUN_NAME"
  export TMX_VLLM_FANOUT_NONCE="$RUN_NAME"
  export TMX_REQUIRE_SAME_REGION_TPU=1
  export TMX_ALLOW_CROSS_REGION_TPU=0
  # IAP is only for local control-plane SSH. TPU-to-TPU reload traffic stays direct.
  export TMX_GCLOUD_TPU_USE_IAP="${TMX_GCLOUD_TPU_USE_IAP:-1}"
  export TPU_TRANSPORT_FLAG=direct
  export TMX_REPAIR_TRAIN_LIBTPU="${TMX_REPAIR_TRAIN_LIBTPU:-1}"
  export TMX_LIBTPU_VERSION="${TMX_LIBTPU_VERSION:-0.0.21}"
  export TMX_FAST_PACKAGE_FANOUT="${TMX_FAST_PACKAGE_FANOUT:-0}"
  export TMX_FAST_WORKER_SCRIPT_FANOUT="${TMX_FAST_WORKER_SCRIPT_FANOUT:-0}"
  export ALLOW_DEGRADED_SERVE=1
  export TMX_PARALLEL_VLLM_HEALTH=1
  export TMX_TRAIN_VLLM_RELOAD_MODE=direct
  export TMX_TRAIN_VLLM_RELOAD_ON_RESUME="${TMX_TRAIN_VLLM_RELOAD_ON_RESUME:-0}"
  export TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS=20
  export TMX_TRAIN_VLLM_RELOAD_FIRST_STEP="${TMX_TRAIN_VLLM_RELOAD_FIRST_STEP:-20}"
  export TMX_VLLM_RELOAD_ALLOW_PARALLEL=1
  export TMX_VLLM_RELOAD_PARALLELISM="${TMX_VLLM_RELOAD_PARALLELISM:-8}"
  export SAVE_EVERY_STEPS=20
  export TMX_HF_CHECKPOINT_EVERY_STEPS=20
  export TMX_HF_CHECKPOINT_FIRST_STEP="${TMX_HF_CHECKPOINT_FIRST_STEP:-20}"
  export TMX_ARTIFACT_BACKEND=hf
  export TMX_ALLOW_GCS_ARTIFACTS=0
  export TMX_GCS_CHECKPOINT_BASE=
  export TMX_HF_CHECKPOINT_REPO=SaifPunjwani/two-model-exploration-checkpoints
  export TMX_HF_MODEL_REPO=SaifPunjwani/two-model-exploration-models
  export TMX_HF_DATASET_REPO=SaifPunjwani/two-model-exploration-checkpoints
  export TMX_HF_REPO_TYPE=dataset
  export TMX_HF_ARTIFACT_PREFIX="$RUN_NAME"
  export TMX_REQUIRE_HF_ARTIFACT_UPLOAD=1
  export TMX_WANDB_SINGLE_RUN=1
  export TMX_WANDB_COMPACT=1
  export TMX_WANDB_MINIMAL=1
  export WANDB_PROJECT=two-model-explore
  export WANDB_ENTITY="${WANDB_ENTITY:-}"
  export TMX_AIME_PROBE_EVERY_STEPS=20
  export TMX_AIME_PROBE_NUM_PROBLEMS=6
  export TMX_AIME_PROBE_NUM_ROLLOUTS=4
}

case "$lane" in
  lambda05_extend)
    export RUN_NAME="${RUN_NAME:-dapo16k_drgrpo_novelty05_v6etrain_v5serve_resume100_300steps_ckpt20_${stamp}}"
    export SOURCE_RUN="${SOURCE_RUN:-dapo16k_drgrpo_novelty05_hf_resume49_20260429_174651_wandbresume_parallelvllm}"
    export SOURCE_STEP="${SOURCE_STEP:-000100}"
    export INIT_CHECKPOINT="hf://SaifPunjwani/two-model-exploration-checkpoints/${SOURCE_RUN}/explorer/checkpoints/step_${SOURCE_STEP}"
    export TMX_FRESH_RUN=0
    export GRPO_BATCH_SIZE=4
    export GRPO_GRAD_ACCUM=1
    export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-300}"
    export LAMBDA_NOVELTY=0.5
    export TRAIN_TPU_PROJECT="${TRAIN_TPU_PROJECT:?set TRAIN_TPU_PROJECT to the GCP project id of the training TPU}"
    export TRAIN_ZONE=europe-west4-a
    export TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
    export TPU_ACCELERATOR_TYPE=v6e-64
    export TMX_TPU_HOST_BOUNDS=4,4,1
    export TRAIN_WORKERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    export SERVE_TPU_PROJECT="${SERVE_TPU_PROJECT:?set SERVE_TPU_PROJECT to the GCP project id of the serving TPU}"
    export SERVE_ZONE=europe-west4-b
    export SERVE_TPU_NAME="${SERVE_TPU_NAME:?set SERVE_TPU_NAME to the serving (vLLM) TPU VM (slice) name}"
    export SERVE_WORKERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    export VLLM_CONCURRENCY=96
    export MIN_HEALTHY_SERVE_WORKERS=12
    export TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS=12
    export TMX_TRAIN_VLLM_RELOAD_FIRST_STEP=120
    export TMX_HF_CHECKPOINT_FIRST_STEP=120
    common_env
    ;;
  lambda025_trc)
    export RUN_NAME="${RUN_NAME:-dapo16k_drgrpo_novelty025_trc_r1train_r2serve_100steps_ckpt20_${stamp}}"
    export GRPO_BATCH_SIZE=4
    export GRPO_GRAD_ACCUM=1
    export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-100}"
    export LAMBDA_NOVELTY=0.25
    export TRAIN_TPU_PROJECT="${TRAIN_TPU_PROJECT:?set TRAIN_TPU_PROJECT to the GCP project id of the training TPU}"
    export TRAIN_ZONE=europe-west4-b
    export TRAIN_TPU_NAME=trc2-v5lite64-ew4b-r1
    export SERVE_TPU_PROJECT="${SERVE_TPU_PROJECT:?set SERVE_TPU_PROJECT to the GCP project id of the serving TPU}"
    export SERVE_ZONE=europe-west4-b
    export SERVE_TPU_NAME=trc2-v5lite64-ew4b-r2
    export TPU_ACCELERATOR_TYPE=v5litepod-64
    export TMX_TPU_HOST_BOUNDS=4,4,1
    export TRAIN_WORKERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    export SERVE_WORKERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    export VLLM_CONCURRENCY=96
    export MIN_HEALTHY_SERVE_WORKERS=12
    export TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS=12
    common_env
    ;;
  deepscaler_lambda05)
    export RUN_NAME="${RUN_NAME:-deepscaler17k_decontam_drgrpo_novelty05_trc_r3train_r4serve_100steps_ckpt20_${stamp}}"
    export DATASET_NAME=jsonl:deepscaler_17k_decontaminated
    export TMX_TRAIN_DATASET_JSONL="${TMX_TRAIN_DATASET_JSONL:?set TMX_TRAIN_DATASET_JSONL to the training JSONL path on the TPU VM}"
    export MAX_TRAIN_EXAMPLES=17000
    export GRPO_BATCH_SIZE=4
    export GRPO_GRAD_ACCUM=1
    export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-100}"
    export LAMBDA_NOVELTY=0.5
    export TRAIN_TPU_PROJECT="${TRAIN_TPU_PROJECT:?set TRAIN_TPU_PROJECT to the GCP project id of the training TPU}"
    export TRAIN_ZONE=europe-west4-b
    export TRAIN_TPU_NAME=trc2-v5lite64-ew4b-r3
    export SERVE_TPU_PROJECT="${SERVE_TPU_PROJECT:?set SERVE_TPU_PROJECT to the GCP project id of the serving TPU}"
    export SERVE_ZONE=europe-west4-b
    export SERVE_TPU_NAME=trc2-v5lite64-ew4b-r4
    export TPU_ACCELERATOR_TYPE=v5litepod-64
    export TMX_TPU_HOST_BOUNDS=4,4,1
    export TRAIN_WORKERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    export SERVE_WORKERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    export VLLM_CONCURRENCY=96
    export MIN_HEALTHY_SERVE_WORKERS=12
    export TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS=12
    common_env
    ;;
  lambda075_cyber)
    export RUN_NAME="${RUN_NAME:-dapo16k_drgrpo_novelty075_cybertrain_saifserve_100steps_ckpt20_${stamp}}"
    export GRPO_BATCH_SIZE=4
    export GRPO_GRAD_ACCUM=1
    export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-100}"
    export LAMBDA_NOVELTY=0.75
    export TRAIN_TPU_PROJECT="${TRAIN_TPU_PROJECT:?set TRAIN_TPU_PROJECT to the GCP project id of the training TPU}"
    export TRAIN_ZONE=us-central1-a
    export TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
    export SERVE_TPU_PROJECT="${SERVE_TPU_PROJECT:?set SERVE_TPU_PROJECT to the GCP project id of the serving TPU}"
    export SERVE_ZONE=us-central1-a
    export SERVE_TPU_NAME="${SERVE_TPU_NAME:?set SERVE_TPU_NAME to the serving (vLLM) TPU VM (slice) name}"
    export TPU_ACCELERATOR_TYPE=v5litepod-64
    export TMX_TPU_HOST_BOUNDS=4,4,1
    export TRAIN_WORKERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    export SERVE_WORKERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
    export VLLM_CONCURRENCY=96
    export MIN_HEALTHY_SERVE_WORKERS=12
    export TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS=12
    export TMX_VLLM_RELOAD_PARALLELISM=8
    export ENV_DIR="${ENV_DIR:?set ENV_DIR to the Python env directory present on every worker}"
    common_env
    ;;
  *)
    echo "unknown lane: $lane" >&2
    exit 2
    ;;
esac

echo "[active-launch] lane=$lane run=$RUN_NAME"
echo "[active-launch] train=$TRAIN_TPU_PROJECT/$TRAIN_ZONE/$TRAIN_TPU_NAME workers=[$TRAIN_WORKERS]"
echo "[active-launch] serve=$SERVE_TPU_PROJECT/$SERVE_ZONE/$SERVE_TPU_NAME workers=[$SERVE_WORKERS]"
echo "[active-launch] contract lambda=$LAMBDA_NOVELTY G=$GRPO_NUM_GENERATIONS batch=$GRPO_BATCH_SIZE steps=$GRPO_MAX_STEPS ckpt_every=$SAVE_EVERY_STEPS max_completion=$MAX_COMPLETION_LEN dataset=${DATASET_NAME:-dapo_math_17k}"

exec scripts/restart_dapo_drgrpo_twoslice.sh
