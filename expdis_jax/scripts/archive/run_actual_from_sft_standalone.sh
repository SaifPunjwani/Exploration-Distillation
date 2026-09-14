#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:?set RUN_NAME to the run whose actual_sft_final should be trained}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/expdis/runs}"
ROOT="$RUNS_ROOT/$RUN_NAME"
ACTUAL_DIR="$ROOT/actual"
SFT_CKPT="${SFT_CKPT:-$ACTUAL_DIR/actual_sft_final}"
OUTPUT_DIR="${OUTPUT_DIR:-$ACTUAL_DIR/grpo}"
LOG_FILE="${LOG_FILE:-$ACTUAL_DIR/actual_grpo_standalone.log}"

VLLM_URLS="${VLLM_URLS:?set VLLM_URLS to the comma-separated vLLM server URLs}"

mkdir -p "$ACTUAL_DIR"
cd "$HOME/expdis"
source "$HOME/miniforge3/bin/activate" expdis_jax

export RUN_NAME RUNS_ROOT
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PJRT_DEVICE=TPU TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost
export EXPDIS_ARTIFACT_BACKEND="${EXPDIS_ARTIFACT_BACKEND:-hf}"
export EXPDIS_ALLOW_GCS_ARTIFACTS="${EXPDIS_ALLOW_GCS_ARTIFACTS:-0}"
export EXPDIS_HF_CHECKPOINT_REPO="${EXPDIS_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}"
export EXPDIS_HF_REPO_TYPE="${EXPDIS_HF_REPO_TYPE:-dataset}"
export EXPDIS_GCS_CHECKPOINT_BASE="${EXPDIS_GCS_CHECKPOINT_BASE:-}"
export EXPDIS_WANDB_SINGLE_RUN=1 EXPDIS_WANDB_COMPACT=1
export EXPDIS_WANDB_RUN_ID="$WANDB_RUN_ID"
export EXPDIS_WANDB_ACTUAL_GRPO_RUN_ID="$WANDB_RUN_ID"
export EXPDIS_RESTORE_OPT_STATE=0 EXPDIS_SAVE_OPT_STATE=0
export EXPDIS_VLLM_RETURN_LOGPROBS=0 EXPDIS_TRACK_EXTERNAL_OLD_GAP=0

if [[ "${EXPDIS_ALLOW_STANDALONE_ACTUAL_WITH_PRELOADED_VLLM:-0}" != "1" ]]; then
  echo "[actual] refusing standalone Actual GRPO unless vLLM has already been reloaded to the SFT policy." >&2
  echo "[actual] Use expdis_jax.pipeline for the clean path, or set EXPDIS_ALLOW_STANDALONE_ACTUAL_WITH_PRELOADED_VLLM=1 after verifying vLLM." >&2
  exit 2
fi

if [[ ! -d "$SFT_CKPT" ]]; then
  echo "[actual] missing SFT checkpoint: $SFT_CKPT" >&2
  exit 2
fi

python -m expdis_jax.train \
  --model-name "$MODEL_NAME" \
  --dataset-name dapo_math_17k \
  --max-train-examples 20000 \
  --max-prompt-len 2048 \
  --max-completion-len 16384 \
  --max-total-len 18432 \
  --grpo-batch-size 1 \
  --grpo-grad-accum "${GRPO_GRAD_ACCUM:-2}" \
  --grpo-num-generations 8 \
  --grpo-max-steps "${GRPO_MAX_STEPS:-96}" \
  --grpo-loss-type dr_grpo \
  --optimizer adamw \
  --grpo-lr "${GRPO_LR:-1e-6}" \
  --grpo-clip-epsilon 0.2 \
  --grpo-kl-beta "${ACTUAL_GRPO_KL_BETA:-0.10}" \
  --trajectory-batch-size "${TRAJECTORY_BATCH_SIZE:-2}" \
  --vllm-server-urls "$VLLM_URLS" \
  --vllm-concurrency 16 \
  --vllm-fanout-per-prompt 8 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --enable-thinking auto \
  --reward-correct 1.0 \
  --reward-incorrect 0.0 \
  --format-penalty-weight 0.0 \
  --lambda-novelty 0.0 \
  --novelty-metric sqrt_mse \
  --novelty-clip 0.0 \
  --novelty-feature-source multilayer \
  --novelty-layers 7,14,21 \
  --novelty-layer-pool mean \
  --novelty-layer-agg mean \
  --novelty-text-contract full_raw \
  --novelty-feature-max-length 10240 \
  --novelty-zscore off \
  --correctness-zscore off \
  --rnd-hidden 512 \
  --rnd-lr 1e-4 \
  --incorrect-novelty-scale 0.0 \
  --gate-novelty-by-quality false \
  --benchmark-num-rollouts 32 \
  --benchmark-rollout-max-completion-len 32768 \
  --rescore-chunk-size 1 \
  --logprob-chunk-tokens "${LOGPROB_CHUNK_TOKENS:-16}" \
  --init-checkpoint "$SFT_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --wandb-run-name "$RUN_NAME" \
  --save-every-steps "${SAVE_EVERY_STEPS:-50}" \
  --seed 0

FINAL_CKPT="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'step_*' | sort | tail -1)"
if [[ -z "$FINAL_CKPT" ]]; then
  echo "[actual-export] no final checkpoint found in $OUTPUT_DIR" >&2
  exit 3
fi
export FINAL_CKPT ACTUAL_HF_OUT="$ACTUAL_DIR/actual_hf_final"

python - <<'PY'
import os
import jax
import numpy as np
import orbax.checkpoint as ocp
from huggingface_hub import HfApi
from expdis_jax.weights import export_flax_params_to_hf_dir, load_hf_config

run = os.environ["RUN_NAME"]
ckpt = os.environ["FINAL_CKPT"]
out = os.environ["ACTUAL_HF_OUT"]
model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")
print(f"[actual-export] restoring {ckpt}", flush=True)
payload = ocp.PyTreeCheckpointer().restore(os.path.abspath(ckpt))
params = jax.tree_util.tree_map(lambda x: np.asarray(x), payload["params"])
hf_cfg = load_hf_config(model_name)
export_flax_params_to_hf_dir(
    params,
    model_name,
    int(hf_cfg["num_hidden_layers"]),
    bool(hf_cfg.get("tie_word_embeddings", False)),
    out,
)
print(f"[actual-export] exported {out}", flush=True)
repo_id = os.environ.get("EXPDIS_HF_CHECKPOINT_REPO", "SaifPunjwani/two-model-exploration-checkpoints")
repo_type = os.environ.get("EXPDIS_HF_REPO_TYPE", "dataset")
path_in_repo = f"{run}/actual_hf_final"
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=out,
    path_in_repo=path_in_repo,
    commit_message=f"actual export {path_in_repo}",
)
print(f"[actual-export] uploaded hf://{repo_id}/{path_in_repo}", flush=True)
PY
