#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv_tpu/bin/python" ]; then
    PYTHON_BIN=".venv_tpu/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

DEVICE="${DEVICE:-tpu}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
RUN_TAG="${RUN_TAG:-phase1_matrix}"
SEED="${SEED:-42}"
DATASETS="${DATASETS:-gsm8k}"
CONDITIONS="${CONDITIONS:-base_only_saturation two_model_rnd two_model_output_kl two_model_param_l2 multi_round_rnd}"
OUT_ROOT="${OUT_ROOT:-runs/${RUN_TAG}}"
RUN_AS_ROOT="${RUN_AS_ROOT:-auto}"
PROMOTED_GSM8K_RECIPE="${PROMOTED_GSM8K_RECIPE:-0}"
DATASET_SPLIT_POLICY="${DATASET_SPLIT_POLICY:-auto}"
TOPIC_SPLIT_MODE="${TOPIC_SPLIT_MODE:-heuristic_math_topics}"
TOPIC_PARTITION_POLICY="${TOPIC_PARTITION_POLICY:-largest_holdout}"
ACTUAL_TRAIN_MODE="${ACTUAL_TRAIN_MODE:-sft_grpo}"
ACTUAL_TARGET_CONTRACT="${ACTUAL_TARGET_CONTRACT:-canonical}"
ACTUAL_REWARD_SOURCE="${ACTUAL_REWARD_SOURCE:-correctness}"
ACTUAL_TARGET_CORRECT_FRACTION="${ACTUAL_TARGET_CORRECT_FRACTION:-0.0}"
BASE_EVAL_MODE="${BASE_EVAL_MODE:-grpo}"
DECOUPLING_MODE="${DECOUPLING_MODE:-decoupled}"
NOVELTY_FEATURE_SOURCE="${NOVELTY_FEATURE_SOURCE:-multilayer}"
NOVELTY_LAYERS="${NOVELTY_LAYERS:-}"
NOVELTY_LAYER_POOL="${NOVELTY_LAYER_POOL:-mean}"
NOVELTY_LAYER_AGG="${NOVELTY_LAYER_AGG:-mean}"
NOVELTY_METRIC="${NOVELTY_METRIC:-sqrt_mse}"
NOVELTY_TEXT_CONTRACT="${NOVELTY_TEXT_CONTRACT:-full_raw}"
NOVELTY_FEATURE_MAX_LENGTH="${NOVELTY_FEATURE_MAX_LENGTH:-0}"
FORMAT_PENALTY_WEIGHT="${FORMAT_PENALTY_WEIGHT:-0.2}"
GATE_NOVELTY_BY_QUALITY="${GATE_NOVELTY_BY_QUALITY:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-192}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-96}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-288}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-4}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-2}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-2}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-2}"
MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-256}"
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-2}"
ACTUAL_GRPO_BATCH_SIZE="${ACTUAL_GRPO_BATCH_SIZE:-2}"
ACTUAL_GRPO_GRAD_ACCUM="${ACTUAL_GRPO_GRAD_ACCUM:-1}"
MAX_ACTUAL_EXAMPLES="${MAX_ACTUAL_EXAMPLES:-0}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-4}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-scripts/aggregate_phase1_results.py}"

MAX_TRAIN_GSM8K="${MAX_TRAIN_GSM8K:-2048}"
MAX_EVAL_GSM8K="${MAX_EVAL_GSM8K:-512}"
MAX_TRAIN_DEEPSCALER="${MAX_TRAIN_DEEPSCALER:-4096}"
MAX_EVAL_DEEPSCALER="${MAX_EVAL_DEEPSCALER:-1024}"
MAX_TRAIN_DAPO="${MAX_TRAIN_DAPO:-4096}"
MAX_EVAL_DAPO="${MAX_EVAL_DAPO:-1024}"

BASE_GRPO_STEPS="${BASE_GRPO_STEPS:-120}"
TWO_MODEL_GRPO_STEPS="${TWO_MODEL_GRPO_STEPS:-120}"
MULTI_ROUND_GRPO_STEPS="${MULTI_ROUND_GRPO_STEPS:-60}"
NUM_ROUNDS="${NUM_ROUNDS:-4}"
LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.1}"
KL_ANCHOR_WEIGHT="${KL_ANCHOR_WEIGHT:-0.05}"
PARAM_ANCHOR_WEIGHT="${PARAM_ANCHOR_WEIGHT:-0.05}"

export WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"

if [ "${WANDB_MODE:-}" != "offline" ]; then
  "$PYTHON_BIN" - <<'PY'
import os
import sys
import wandb

api_key = os.environ.get("WANDB_API_KEY") or getattr(getattr(wandb, "api", None), "api_key", None)
if not api_key:
    raise SystemExit(
        "W&B logging is required for the phase-1 matrix. "
        "Run `wandb login` or export WANDB_API_KEY before launching this script."
    )
PY
fi

if [ "$DEVICE" = "tpu" ]; then
  export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
  export USE_TORCH_XLA=1
  export EXPDIS_REQUIRE_TPU="${EXPDIS_REQUIRE_TPU:-1}"
  export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v4-8}"
  export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
fi

mkdir -p "$OUT_ROOT"

if [ "$PROMOTED_GSM8K_RECIPE" != "1" ]; then
  for dataset in $DATASETS; do
    if [ "$dataset" != "gsm8k" ]; then
      echo "All-dataset matrix is gated until a GSM8K recipe is promoted. Set PROMOTED_GSM8K_RECIPE=1 to enable non-GSM8K datasets." >&2
      exit 1
    fi
  done
fi

RUNNER=("$PYTHON_BIN")
if [ "$DEVICE" = "tpu" ]; then
  NEED_SUDO=0
  if [ "$RUN_AS_ROOT" = "always" ]; then
    NEED_SUDO=1
  elif [ "$RUN_AS_ROOT" = "auto" ]; then
    if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v4-8")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
os.environ.setdefault("TPU_WORKER_ID", "0")
os.environ.setdefault("TPU_WORKER_HOSTNAMES", "localhost")
import torch_xla.core.xla_model as xm
xm.xla_device()
PY
    then
      NEED_SUDO=1
    fi
  fi
  if [ "$NEED_SUDO" = "1" ]; then
    echo "[tpu] Current user cannot acquire XLA device; using sudo wrapper for TPU runs."
    RUNNER=(sudo -E -n env "PATH=$PATH" "HOME=$HOME" "$PYTHON_BIN")
  fi
fi

run_case() {
  local dataset="$1"
  local condition="$2"
  local max_train="$3"
  local max_eval="$4"
  shift 4

  local dataset_config_args=()
  if [ "$dataset" = "dapo_math_17k" ]; then
    dataset_config_args+=(--dataset-config-name "${DATASET_CONFIG_DAPO:-all}")
  elif [ "$dataset" = "deepscaler_preview" ]; then
    dataset_config_args+=(--dataset-config-name "${DATASET_CONFIG_DEEPSCALER:-default}")
  fi

  local constraint_args=()
  if [ "$dataset" = "gsm8k" ]; then
    constraint_args+=(--answer-constraint-mode integer)
  else
    constraint_args+=(--answer-constraint-mode expression)
  fi

  local run_dir="${OUT_ROOT}/${dataset}/${condition}"
  local run_name="${RUN_TAG}_${dataset}_${condition}_seed${SEED}"
  mkdir -p "$run_dir"

  "${RUNNER[@]}" run_expdis_torch.py \
    --device "$DEVICE" \
    --use-wandb \
    --model-name "$MODEL_NAME" \
    --dataset-name "$dataset" \
    "${dataset_config_args[@]}" \
    --dataset-split-policy "$DATASET_SPLIT_POLICY" \
    --dataset-split-seed "$SEED" \
    --dataset-manifest-path "${run_dir}/dataset_manifest.json" \
    --topic-split-mode "$TOPIC_SPLIT_MODE" \
    --topic-partition-policy "$TOPIC_PARTITION_POLICY" \
    --decoupling-mode "$DECOUPLING_MODE" \
    --actual-train-mode "$ACTUAL_TRAIN_MODE" \
    --actual-target-contract "$ACTUAL_TARGET_CONTRACT" \
    --actual-reward-source "$ACTUAL_REWARD_SOURCE" \
    --actual-target-correct-fraction "$ACTUAL_TARGET_CORRECT_FRACTION" \
    --base-eval-mode "$BASE_EVAL_MODE" \
    --novelty-feature-source "$NOVELTY_FEATURE_SOURCE" \
    --novelty-layer-pool "$NOVELTY_LAYER_POOL" \
    --novelty-layer-agg "$NOVELTY_LAYER_AGG" \
    --novelty-metric "$NOVELTY_METRIC" \
    --novelty-text-contract "$NOVELTY_TEXT_CONTRACT" \
    --lambda-novelty "$LAMBDA_NOVELTY" \
    --format-penalty-weight "$FORMAT_PENALTY_WEIGHT" \
    --max-train-examples "$max_train" \
    --max-eval-examples "$max_eval" \
    --max-prompt-len "$MAX_PROMPT_LEN" \
    --max-completion-len "$MAX_COMPLETION_LEN" \
    --max-total-len "$MAX_TOTAL_LEN" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --trajectory-batch-size "$TRAJECTORY_BATCH_SIZE" \
    --num-samples-per-prompt "$NUM_SAMPLES_PER_PROMPT" \
    --distill-epochs "$DISTILL_EPOCHS" \
    --distill-batch-size "$DISTILL_BATCH_SIZE" \
    --max-distill-examples "$MAX_DISTILL_EXAMPLES" \
    --actual-grpo-epochs "$ACTUAL_GRPO_EPOCHS" \
    --actual-grpo-batch-size "$ACTUAL_GRPO_BATCH_SIZE" \
    --actual-grpo-grad-accum "$ACTUAL_GRPO_GRAD_ACCUM" \
    --grpo-num-generations "$GRPO_NUM_GENERATIONS" \
    --grpo-generation-batch-size "$GRPO_GENERATION_BATCH_SIZE" \
    --seed "$SEED" \
    --wandb-run-name "$run_name" \
    --plots-dir "${run_dir}/plots" \
    --explorer-traj-path "${run_dir}/explorer_trajectories.jsonl" \
    "${constraint_args[@]}" \
    $(maybe_add_novelty_feature_max_length) \
    $(maybe_add_novelty_layers) \
    $(maybe_disable_quality_gate) \
    "$@"
}

maybe_add_max_actual_examples() {
  if [ "${MAX_ACTUAL_EXAMPLES}" -gt 0 ]; then
    printf '%s\n' --max-actual-examples "${MAX_ACTUAL_EXAMPLES}"
  fi
}

maybe_add_novelty_layers() {
  if [ -n "$NOVELTY_LAYERS" ]; then
    printf '%s\n' --novelty-layers "${NOVELTY_LAYERS}"
  fi
}

maybe_add_novelty_feature_max_length() {
  if [ "${NOVELTY_FEATURE_MAX_LENGTH}" -gt 0 ]; then
    printf '%s\n' --novelty-feature-max-length "${NOVELTY_FEATURE_MAX_LENGTH}"
  fi
}

maybe_disable_quality_gate() {
  if [ "${GATE_NOVELTY_BY_QUALITY}" = "0" ]; then
    printf '%s\n' --no-gate-novelty-by-quality
  fi
}

for dataset in $DATASETS; do
  case "$dataset" in
    gsm8k)
      max_train="$MAX_TRAIN_GSM8K"
      max_eval="$MAX_EVAL_GSM8K"
      ;;
    deepscaler_preview)
      max_train="$MAX_TRAIN_DEEPSCALER"
      max_eval="$MAX_EVAL_DEEPSCALER"
      ;;
    dapo_math_17k)
      max_train="$MAX_TRAIN_DAPO"
      max_eval="$MAX_EVAL_DAPO"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac

  for condition in $CONDITIONS; do
    case "$condition" in
      base_only_saturation)
        run_case "$dataset" "$condition" "$max_train" "$max_eval" \
          --pipeline-mode base_only \
          --grpo-max-steps "$BASE_GRPO_STEPS" \
          --anchor-metric none
        ;;
      two_model_rnd)
        run_case "$dataset" "$condition" "$max_train" "$max_eval" \
          --pipeline-mode two_model \
          --grpo-max-steps "$TWO_MODEL_GRPO_STEPS" \
          --anchor-metric none \
          $(maybe_add_max_actual_examples)
        ;;
      two_model_output_kl)
        run_case "$dataset" "$condition" "$max_train" "$max_eval" \
          --pipeline-mode two_model \
          --grpo-max-steps "$TWO_MODEL_GRPO_STEPS" \
          --anchor-metric output_kl \
          --anchor-weight "$KL_ANCHOR_WEIGHT" \
          $(maybe_add_max_actual_examples)
        ;;
      two_model_param_l2)
        run_case "$dataset" "$condition" "$max_train" "$max_eval" \
          --pipeline-mode two_model \
          --grpo-max-steps "$TWO_MODEL_GRPO_STEPS" \
          --anchor-metric param_l2 \
          --anchor-weight "$PARAM_ANCHOR_WEIGHT" \
          $(maybe_add_max_actual_examples)
        ;;
      multi_round_rnd)
        run_case "$dataset" "$condition" "$max_train" "$max_eval" \
          --pipeline-mode multi_round \
          --grpo-max-steps "$MULTI_ROUND_GRPO_STEPS" \
          --num-rounds "$NUM_ROUNDS" \
          --round-shard-policy topic_balanced \
          --anchor-metric none \
          $(maybe_add_max_actual_examples)
        ;;
      *)
        echo "Unsupported condition: $condition" >&2
        exit 1
        ;;
    esac
  done
done

if [ -f "$SUMMARY_SCRIPT" ]; then
  "$PYTHON_BIN" "$SUMMARY_SCRIPT" --root "$OUT_ROOT" --out-dir "${OUT_ROOT}/aggregate"
fi

echo "Phase 1 matrix complete. Outputs: $OUT_ROOT"
