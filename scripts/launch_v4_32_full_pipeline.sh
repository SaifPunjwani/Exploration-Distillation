#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/two-model-exploration"
source .venv_tpu/bin/activate
source .env 2>/dev/null || true

# === Full Pipeline on v4-32: Explorer ckpt-1000 → Trajectories → SFT → Online GRPO → Eval ===
# Fallback from v6e-64 (spot preempted). v4-32 in us-central2-b reaches
# vLLM servers in europe-west4-b over VPC internal IPs.

# === W&B ===
export WANDB_PROJECT=two-model-explore
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_XET_HIGH_PERFORMANCE_DOWNLOAD=0

# === TPU single-chip training ===
export TPU_VISIBLE_CHIPS=0
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export PJRT_DEVICE=TPU

# === v4: use eager attention (flash attention Pallas kernel not tested on v4) ===
export TMX_XLA_FLASH_ATTENTION=0
export TMX_XLA_FORCE_EAGER_ATTN=1
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_OPTIM=adamw_torch

# === Decoupled training ===
export TMX_DECOUPLED_TRAIN_TOKEN_CHUNK=0
export TMX_DECOUPLED_GRAD_CKPT=0
export TMX_DECOUPLED_TRAIN_DEBUG_PROGRESS=1

# === vLLM for generation (ALL 16 v5litepod servers in europe-west4-b) ===
export TMX_GRPO_USE_VLLM=1
export TMX_TRAJ_USE_VLLM=1
export TMX_VLLM_SERVER_URLS="${TMX_VLLM_SERVER_URLS:?set TMX_VLLM_SERVER_URLS to the comma-separated vLLM server URLs}"
export TMX_VLLM_MODEL_NAME=qwen3_serious_policy
export TMX_VLLM_CONCURRENCY=64
export TMX_VLLM_MAX_RETRIES=5
export TMX_TRAJ_MAX_COMPLETION_LEN=8192

# === Resume from Explorer checkpoint-1000 (skip Explorer GRPO entirely) ===
export TMX_RESUME_STEP=1000
export TMX_RESUME_CHECKPOINT=runs/explorer_ckpt_1000_from_gcs
export TMX_RESUME_RUN_NAME=""
export TMX_RESUME_WANDB_ID=""

# === GCS checkpoint backup ===
export TMX_GCS_CHECKPOINT_ROOT=gs://two-model-exploration-checkpoints/v4_32_full_pipeline

# === Trajectories: reuse from previous run (2088 accepted, ~1.5hr saved) ===
export TMX_PRECOMPUTED_TRAJ_PATH=runs/v4_32_full_pipeline_20260411_175504/explorer_trajectories.jsonl

# === FULL PIPELINE ===
export TMX_STOP_AFTER_EXPLORER=0
export TMX_STOP_AFTER_TRAJECTORIES=0
export TMX_SKIP_INTERMEDIATE_EVALS=1
export TMX_SKIP_FINAL_BENCHMARK_EVALS=0
export TMX_ABORT_ON_DEGENERATE_EXPLORER=0
export TMX_ABORT_ON_BAD_TRAJ_WINDOW=0
export TMX_GRPO_LOGGING_STRATEGY=steps
export TMX_GRPO_LOGGING_STEPS=1
export TMX_GRPO_SAVE_STRATEGY=steps
export TMX_GRPO_SAVE_STEPS=25
export TMX_GRPO_SAVE_TOTAL_LIMIT=5
export TMX_WANDB_MINIMAL=1
export TMX_NOVELTY_ON_TPU=1
export TMX_NOVELTY_BATCH_SIZE=16

# === Trajectory early stopping ===
export TMX_TRAJ_TARGET_ACCEPTED=2048

# === Actual GRPO save settings ===
export TMX_ACTUAL_GRPO_SAVE_STEPS=25
export TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT=4

# === SFT max sequence length (v4 has 31.75G HBM, 4096 OOMs by 333MB; 3072 fits) ===
export TMX_DISTILL_MAX_TOTAL_LEN=3072

# === Gradient checkpointing for eager attention (required for memory) ===
export TMX_DISTILL_GRAD_CKPT=1

# === Threading ===
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONPATH="$HOME/two-model-exploration"

RUN_NAME="v4_32_full_pipeline_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

echo "=== Full Pipeline (v4-32): Explorer ckpt-1000 → Trajectories → SFT → Online GRPO → Eval ===" | tee -a "${RUN_DIR}/console.log"
echo "Run: ${RUN_NAME}" | tee -a "${RUN_DIR}/console.log"
echo "Explorer source: GCS checkpoint-1000" | tee -a "${RUN_DIR}/console.log"
date | tee -a "${RUN_DIR}/console.log"

python main.py \
  --device tpu \
  --use-wandb \
  --seed 0 \
  --model-name "$HOME/.models/Qwen3-1.7B" \
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
  --novelty-zscore batch \
  --actual-online-grpo-max-steps 250 \
  --benchmark-dataset-name aime_2024 \
  --benchmark-eval-mode sampled \
  --benchmark-num-rollouts 32 \
  --max-benchmark-examples 30 \
  >> "${RUN_DIR}/console.log" 2>&1
