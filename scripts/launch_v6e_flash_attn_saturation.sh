#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/two-model-exploration"
source .venv_tpu_slice/bin/activate
source .env 2>/dev/null || true

# === W&B ===
export WANDB_PROJECT=two-model-explore
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# === TPU single-chip training with flash attention ===
export TPU_VISIBLE_CHIPS=0
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export PJRT_DEVICE=TPU

# === Flash attention: eliminates O(N²) memory, enables 8k training on 1 chip ===
export TMX_XLA_FLASH_ATTENTION=1
export TMX_XLA_FORCE_EAGER_ATTN=0
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_OPTIM=adamw_torch

# === Decoupled training: disable token chunking (flash attn handles full seq) ===
export TMX_DECOUPLED_TRAIN_TOKEN_CHUNK=0
export TMX_DECOUPLED_GRAD_CKPT=0
export TMX_DECOUPLED_TRAIN_DEBUG_PROGRESS=1

# === vLLM for generation (ALL 16 v5litepod servers) ===
export TMX_GRPO_USE_VLLM=1
export TMX_TRAJ_USE_VLLM=1
export TMX_VLLM_SERVER_URLS="${TMX_VLLM_SERVER_URLS:?set TMX_VLLM_SERVER_URLS to the comma-separated vLLM server URLs}"
export TMX_VLLM_MODEL_NAME=qwen3_serious_policy
export TMX_VLLM_CONCURRENCY=64
export TMX_VLLM_MAX_RETRIES=5
export TMX_TRAJ_MAX_COMPLETION_LEN=8192

# === Resume from checkpoint (set to 0/"" to start fresh) ===
export TMX_RESUME_STEP=600
export TMX_RESUME_CHECKPOINT=runs/v6e_flash_saturation_8k_20260410_194435/explorer_ckpts/checkpoint-600
export TMX_RESUME_RUN_NAME=v6e_flash_saturation_8k_20260410_194435
export TMX_RESUME_WANDB_ID=89xtbfju

# === GCS checkpoint backup ===
export TMX_GCS_CHECKPOINT_ROOT=gs://two-model-exploration-checkpoints/v6e_flash_saturation_8k

# === Training controls ===
export TMX_SKIP_INTERMEDIATE_EVALS=1
export TMX_SKIP_FINAL_BENCHMARK_EVALS=1
export TMX_STOP_AFTER_EXPLORER=1
export TMX_ABORT_ON_DEGENERATE_EXPLORER=0
export TMX_ABORT_ON_BAD_TRAJ_WINDOW=0
export TMX_GRPO_LOGGING_STRATEGY=steps
export TMX_GRPO_LOGGING_STEPS=1
export TMX_GRPO_SAVE_STRATEGY=steps
export TMX_GRPO_SAVE_STEPS=50
export TMX_GRPO_SAVE_TOTAL_LIMIT=3
export TMX_WANDB_MINIMAL=1
export TMX_NOVELTY_ON_TPU=1
export TMX_NOVELTY_BATCH_SIZE=16

# === Threading (moderate to avoid contention with XLA background threads) ===
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONPATH="$HOME/two-model-exploration"

# Resume into the same run directory and W&B run if TMX_RESUME_STEP is set.
if [ -n "${TMX_RESUME_STEP}" ] && [ "${TMX_RESUME_STEP}" -gt 0 ] 2>/dev/null; then
  RUN_NAME="${TMX_RESUME_RUN_NAME:-v6e_flash_saturation_8k_20260409_115212}"
  export WANDB_RUN_ID="${TMX_RESUME_WANDB_ID:-yq15epui}"
  export WANDB_RESUME=allow
else
  RUN_NAME="v6e_flash_saturation_8k_$(date +%Y%m%d_%H%M%S)"
fi
RUN_DIR="runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

echo "=== v6e flash attention: 8k tokens, 16 vLLM servers, TPU training ===" | tee -a "${RUN_DIR}/console.log"
echo "Run: ${RUN_NAME} (resume_step=${TMX_RESUME_STEP:-0})" | tee -a "${RUN_DIR}/console.log"
date | tee -a "${RUN_DIR}/console.log"

python main.py \
  --device tpu \
  --use-wandb \
  --seed 0 \
  --model-name Qwen/Qwen3-1.7B \
  --dataset-name dapo_math_17k \
  --train-selection-policy random \
  --train-probe-pool-size 256 \
  --train-probe-samples-per-prompt 4 \
  --train-probe-temperature 0.8 \
  --train-probe-batch-size 4 \
  --dataset-split-policy auto \
  --topic-split-mode heuristic_math_topics \
  --topic-partition-policy explicit_or_heuristic \
  --decoupling-mode decoupled \
  --base-eval-mode raw \
  --actual-target-contract full_trace \
  --max-train-examples 20000 \
  --max-eval-examples 32 \
  --grpo-max-steps 1000 \
  --grpo-batch-size 1 \
  --grpo-grad-accum 2 \
  --grpo-loss-type dr_grpo \
  --grpo-num-generations 8 \
  --grpo-generation-batch-size 8 \
  --num-samples-per-prompt 8 \
  --temperature-schedule "1.2,1.0,0.8" \
  --answer-constraint-mode none \
  --exploration-response-style reasoning_final \
  --eval-response-style reasoning_final \
  --exploration-answer-constraint-mode none \
  --eval-answer-constraint-mode none \
  --exploration-thinking-mode think \
  --eval-thinking-mode think \
  --actual-thinking-mode think \
  --distill-epochs 2 \
  --distill-batch-size 1 \
  --max-distill-examples 4096 \
  --actual-grpo-epochs 1 \
  --actual-grpo-batch-size 1 \
  --actual-grpo-grad-accum 2 \
  --actual-min-accepted-trajectories 4 \
  --actual-dataset-policy accepted_only \
  --actual-max-incorrect-ratio 0.25 \
  --actual-target-correct-fraction 0.75 \
  --actual-grpo-kl-coef 0.10 \
  --actual-grpo-lr 1e-6 \
  --distill-lr 5e-6 \
  --max-actual-examples 4096 \
  --novelty-feature-source multilayer \
  --novelty-feature-max-length 512 \
  --novelty-metric sqrt_mse \
  --novelty-text-contract full_raw \
  --rnd-hidden 512 \
  --reward-correct 1.0 \
  --reward-incorrect 0.0 \
  --lambda-novelty 0.10 \
  --reward-shape linear \
  --anchor-metric none \
  --trajectory-batch-size 16 \
  --eval-batch-size 16 \
  --round-budget-mode constant_total \
  --max-prompt-len 2048 \
  --max-completion-len 8192 \
  --exploration-max-completion-len 8192 \
  --eval-max-completion-len 8192 \
  --max-total-len 10240 \
  --format-penalty-weight 0.1 \
  --pipeline-mode two_model \
  --actual-init-source base \
  --actual-train-mode sft_online_grpo \
  --actual-reward-source correctness \
  --wandb-run-name "${RUN_NAME}" \
  --plots-dir "${RUN_DIR}/plots" \
  --explorer-traj-path "${RUN_DIR}/explorer_trajectories.jsonl" \
  --explorer-output-dir "${RUN_DIR}/explorer_ckpts" \
  --saved-model-root "${RUN_DIR}/saved_models" \
  --incorrect-novelty-scale 0.0 \
  --novelty-zscore off \
  >> "${RUN_DIR}/console.log" 2>&1
