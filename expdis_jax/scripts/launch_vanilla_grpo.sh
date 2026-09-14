#!/usr/bin/env bash
# Vanilla GRPO baseline (Shao et al. 2024).
# Mirrors `launch_jax.sh` topology (single-host, 4 TPU chips on worker 0)
# but flips the algorithmic knobs back to canonical GRPO:
#   - advantage = (r - group_mean) / group_std       (std-division ON)
#   - KL penalty β > 0 against reference policy
#   - Symmetric PPO clip ε = 0.2
#   - No novelty bonus (λ = 0)
#   - No dynamic sampling
#   - No SFT, no Actual stage
#
# Geometry matches the contract (group=8: batch=1, grad_accum=2, num_gen=8) so
# any AIME24 delta vs our method is method-attributable, not compute-attributable.
#
# Required env:
#   EXPDIS_VLLM_SERVER_URLS=http://ip1:8000/v1,http://ip2:8000/v1,...
#
# Optional env (defaults shown):
#   RUN_NAME=vanilla_grpo_${DATASET}_$(date +%Y%m%d_%H%M%S)
#   DATASET=deepscaler            (or dapo_math_17k)
#   MODEL_NAME=Qwen/Qwen3-1.7B
#   GRPO_MAX_STEPS=50
#   GRPO_LR=1e-6
#   GRPO_KL_BETA=0.04
#   OUTPUT_ROOT=runs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_DIR="${ENV_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-expdis_jax}"
source "$ENV_DIR/bin/activate" "$ENV_NAME"

VLLM_URLS="${EXPDIS_VLLM_SERVER_URLS:-${VLLM_SERVER_URLS:-}}"
if [ -z "$VLLM_URLS" ]; then
  echo "ERROR: set EXPDIS_VLLM_SERVER_URLS=http://ip:8000/v1,..." >&2
  exit 1
fi

DATASET="${DATASET:-deepscaler}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:-vanilla_grpo_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
OUT="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "$OUT"

# 16k completion to match training/eval context regime
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-16384}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-18432}"
export EXPDIS_ALLOW_LONG_COMPLETION_TRAIN="${EXPDIS_ALLOW_LONG_COMPLETION_TRAIN:-1}"

GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-100}"
GRPO_LR="${GRPO_LR:-1e-6}"
# Vanilla GRPO contract: KL=0 per codex setup (dynamic sampling off; clip-high off)
GRPO_KL_BETA="${GRPO_KL_BETA:-0.0}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:--1.0}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-16}"
VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-8}"

# TPU env (single-host, 4 chips on worker 0 — same as launch_jax.sh)
export WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
if [ -n "${WANDB_ENTITY:-}" ]; then export WANDB_ENTITY; fi  # unset = the W&B default entity
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0,1,2,3}"
export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"

# HuggingFace mirroring — match codex's serious-run setup so artifacts survive
# even if the TPU dies. Per-run namespace via PREFIX.
export EXPDIS_HF_CHECKPOINT_REPO="${EXPDIS_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}"
export EXPDIS_HF_ARTIFACT_PREFIX="${EXPDIS_HF_ARTIFACT_PREFIX:-$RUN_NAME}"
export EXPDIS_HF_MIRROR_CHECKPOINTS="${EXPDIS_HF_MIRROR_CHECKPOINTS:-1}"
export EXPDIS_HF_MIRROR_TRAJECTORIES="${EXPDIS_HF_MIRROR_TRAJECTORIES:-1}"
export EXPDIS_HF_MIRROR_METRICS="${EXPDIS_HF_MIRROR_METRICS:-1}"
export EXPDIS_HF_MIRROR_VLLM_EXPORTS="${EXPDIS_HF_MIRROR_VLLM_EXPORTS:-1}"
export EXPDIS_HF_MIRROR_EVAL_PROBES="${EXPDIS_HF_MIRROR_EVAL_PROBES:-1}"
export EXPDIS_HF_MIRROR_RUN_SUMMARY="${EXPDIS_HF_MIRROR_RUN_SUMMARY:-1}"
export EXPDIS_HF_MIRROR_FINAL_EXPORTS="${EXPDIS_HF_MIRROR_FINAL_EXPORTS:-1}"
export EXPDIS_HF_MIRROR_FINAL_EVAL="${EXPDIS_HF_MIRROR_FINAL_EVAL:-1}"
export EXPDIS_HF_MIRROR_ASYNC="${EXPDIS_HF_MIRROR_ASYNC:-1}"
export EXPDIS_ALLOW_GCS_ARTIFACTS="${EXPDIS_ALLOW_GCS_ARTIFACTS:-0}"
export EXPDIS_GCS_CHECKPOINT_BASE="${EXPDIS_GCS_CHECKPOINT_BASE:-}"
export EXPDIS_REQUIRE_GCS_CHECKPOINT_UPLOAD="${EXPDIS_REQUIRE_GCS_CHECKPOINT_UPLOAD:-0}"

echo "[vanilla-grpo] run=${RUN_NAME} dataset=${DATASET} model=${MODEL_NAME}"
echo "[vanilla-grpo] steps=${GRPO_MAX_STEPS} lr=${GRPO_LR} kl_beta=${GRPO_KL_BETA}"
echo "[vanilla-grpo] vllm_servers=$(echo $VLLM_URLS | tr ',' ' ' | wc -w) URLs"
echo "[vanilla-grpo] output_dir=${OUT}"

# Manifest for reproducibility (single source of truth for this run's algo config).
cat > "$OUT/run_manifest.json" <<EOF
{
  "method": "vanilla_grpo",
  "run_name": "${RUN_NAME}",
  "dataset": "${DATASET}",
  "model": "${MODEL_NAME}",
  "max_steps": ${GRPO_MAX_STEPS},
  "grpo": {
    "advantage_normalization": "std",
    "loss_aggregation": "sequence-mean-token-mean",
    "kl_beta": ${GRPO_KL_BETA},
    "clip_epsilon_low": 0.2,
    "clip_epsilon_high": 0.2,
    "lambda_novelty": 0.0,
    "dynamic_sampling": false,
    "lr": ${GRPO_LR},
    "num_generations": 8,
    "batch_size": 1,
    "grad_accum": 2,
    "reward_correct": ${REWARD_CORRECT},
    "reward_incorrect": ${REWARD_INCORRECT},
    "max_prompt_len": ${MAX_PROMPT_LEN},
    "max_completion_len": ${MAX_COMPLETION_LEN}
  },
  "eval": {
    "benchmark": "AIME24",
    "num_rollouts": 32,
    "max_completion_len": 32768
  }
}
EOF

# Vanilla GRPO contract — geometry matches launch_jax.sh, knobs flipped to canonical.
python -m expdis_jax.train \
  --model-name "$MODEL_NAME" \
  --dataset-name "$DATASET" \
  --max-train-examples "$MAX_TRAIN_EXAMPLES" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --max-total-len "$MAX_TOTAL_LEN" \
  --grpo-batch-size 1 \
  --grpo-grad-accum 2 \
  --grpo-num-generations 8 \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --grpo-loss-type dr_grpo \
  --grpo-loss-aggregation sequence-mean-token-mean \
  --optimizer adamw \
  --grpo-lr "$GRPO_LR" \
  --grpo-clip-epsilon 0.2 \
  --grpo-clip-epsilon-high 0.2 \
  --grpo-kl-beta "$GRPO_KL_BETA" \
  --grpo-advantage-normalization std \
  --trajectory-batch-size 2 \
  --vllm-server-urls "$VLLM_URLS" \
  --vllm-concurrency "$VLLM_CONCURRENCY" \
  --vllm-fanout-per-prompt "$VLLM_FANOUT_PER_PROMPT" \
  --dynamic-sampling false \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --enable-thinking auto \
  --reward-correct "$REWARD_CORRECT" \
  --reward-incorrect "$REWARD_INCORRECT" \
  --format-penalty-weight 0.0 \
  --lambda-novelty 0.0 \
  --novelty-zscore off \
  --correctness-zscore off \
  --benchmark-num-rollouts 32 \
  --benchmark-rollout-max-completion-len 32768 \
  --output-dir "$OUT" \
  --wandb-run-name "$RUN_NAME" \
  --save-every-steps "$SAVE_EVERY_STEPS"

echo "[vanilla-grpo] training complete. To eval AIME24:"
echo "  python -m expdis_jax.eval --model-name <hf_path_or_local_dir> --server-urls $VLLM_URLS"
