#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -n "${HF_ACCESS_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HF_ACCESS_TOKEN"
fi
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv_tpu/bin/python" ]; then
    PYTHON_BIN=".venv_tpu/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [ "${INSTALL_EXTRA_DEPS:-1}" = "1" ]; then
  "$PYTHON_BIN" -m pip install --quiet --disable-pip-version-check scikit-learn umap-learn
fi

DEVICE="${DEVICE:-tpu}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
SEEDS="${SEEDS:-13 17 23}"
DATASETS="${DATASETS:-deepscaler_preview dapo_math_17k}"
RUN_TAG="${RUN_TAG:-novelty_full}"
MAX_TRAIN="${MAX_TRAIN:-4096}"
MAX_EVAL="${MAX_EVAL:-1024}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-120}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:-0.0}"
ID_OOD_HOLDOUT_TOPICS="${ID_OOD_HOLDOUT_TOPICS:-geometry,number_theory}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-8}"
EMBEDDING_MAX_TEXTS="${EMBEDDING_MAX_TEXTS:-768}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-256}"
EMBEDDING_CLUSTER_K="${EMBEDDING_CLUSTER_K:-8}"
EMBEDDING_PERM_ITERS="${EMBEDDING_PERM_ITERS:-1000}"
DATASET_CONFIG_DEEPSCALER="${DATASET_CONFIG_DEEPSCALER:-default}"
DATASET_CONFIG_DAPO="${DATASET_CONFIG_DAPO:-all}"
MODEL_TAG="$(echo "$MODEL_NAME" | sed 's#[^A-Za-z0-9]#-#g')"
TOPIC_TAG="$(echo "$ID_OOD_HOLDOUT_TOPICS" | tr ',' '+')"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-EXP-FULL-DEC-SFTGRPO-BASEGRPO}"

export WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
export EXPDIS_REQUIRE_TPU="${EXPDIS_REQUIRE_TPU:-1}"
if [ "$DEVICE" = "tpu" ]; then
  export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
  export USE_TORCH_XLA=1
  export TOKENIZERS_PARALLELISM=false
  export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
  export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
  export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
  export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
  if [ -z "${PJRT_LOCAL_PROCESS_COUNT:-}" ]; then
    unset PJRT_LOCAL_PROCESS_COUNT || true
  else
    export PJRT_LOCAL_PROCESS_COUNT
  fi
  if [ -z "${PJRT_LOCAL_PROCESS_RANK:-}" ]; then
    unset PJRT_LOCAL_PROCESS_RANK || true
  else
    export PJRT_LOCAL_PROCESS_RANK
  fi
  if [ -n "${TPU_VISIBLE_DEVICES:-}" ]; then
    export TPU_VISIBLE_DEVICES
  fi
  if [ -z "${TPU_LIBRARY_PATH:-}" ]; then
    TPU_LIBRARY_PATH="$("$PYTHON_BIN" - <<'PY'
import os
try:
    import libtpu
    base = os.path.dirname(libtpu.__file__)
    print(os.path.join(base, "libtpu.so"))
except Exception:
    print("")
PY
)"
    if [ -n "$TPU_LIBRARY_PATH" ]; then
      export TPU_LIBRARY_PATH
    fi
  fi
fi

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    DATASET_CONFIG_NAME=""
    if [ "$dataset" = "dapo_math_17k" ]; then
      DATASET_CONFIG_NAME="$DATASET_CONFIG_DAPO"
    elif [ "$dataset" = "deepscaler_preview" ]; then
      DATASET_CONFIG_NAME="$DATASET_CONFIG_DEEPSCALER"
    fi

    OUT_DIR="runs/${RUN_TAG}_${dataset}_seed${seed}"
    RUN_NAME="${RUN_NAME_PREFIX}_${dataset}_${MODEL_TAG}_nov-last+multilayer_layers-4-8-12-16_idood-${TOPIC_TAG}_emb-k${EMBEDDING_CLUSTER_K}_seed${seed}"
    mkdir -p "$OUT_DIR"

    EXTRA_ARGS=()
    if [ -n "$DATASET_CONFIG_NAME" ]; then
      EXTRA_ARGS+=(--dataset-config-name "$DATASET_CONFIG_NAME")
    fi

    "$PYTHON_BIN" run_expdis_torch.py \
      --device "$DEVICE" \
      --use-wandb \
      --model-name "$MODEL_NAME" \
      --dataset-name "$dataset" \
      "${EXTRA_ARGS[@]}" \
      --decoupling-mode decoupled \
      --actual-train-mode sft_grpo \
      --base-eval-mode grpo \
      --novelty-study-mode full \
      --novelty-feature-source-sweep "last,multilayer" \
      --reward-shape linear \
      --novelty-layers "4,8,12,16" \
      --novelty-layer-pool mean \
      --novelty-layer-agg mean \
      --novelty-zscore batch \
      --correctness-zscore off \
      --lambda-novelty 0.1 \
      --novelty-metric sqrt_mse \
      --embedding-novelty-eval \
      --embedding-model-name "$EMBEDDING_MODEL_NAME" \
      --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
      --embedding-max-texts "$EMBEDDING_MAX_TEXTS" \
      --embedding-max-length "$EMBEDDING_MAX_LENGTH" \
      --embedding-cluster-k "$EMBEDDING_CLUSTER_K" \
      --embedding-permutation-iters "$EMBEDDING_PERM_ITERS" \
      --topic-split-mode heuristic_math_topics \
      --id-ood-holdout-topics "$ID_OOD_HOLDOUT_TOPICS" \
      --max-train-examples "$MAX_TRAIN" \
      --max-eval-examples "$MAX_EVAL" \
      --grpo-max-steps "$GRPO_MAX_STEPS" \
      --grpo-num-generations 2 \
      --grpo-generation-batch-size 4 \
      --num-samples-per-prompt 2 \
      --reward-correct "$REWARD_CORRECT" \
      --reward-incorrect "$REWARD_INCORRECT" \
      --seed "$seed" \
      --wandb-run-name "${RUN_NAME}" \
      --plots-dir "$OUT_DIR/plots" \
      --explorer-traj-path "$OUT_DIR/explorer_trajectories.jsonl" \
      --novelty-study-tag "$RUN_TAG"
  done
done

echo "Full novelty study complete. Run aggregation with:"
echo "  python scripts/plot_novelty_study.py --root runs --tag ${RUN_TAG}"
