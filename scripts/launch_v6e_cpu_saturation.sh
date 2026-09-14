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

# === CPU Training (708GB RAM handles 8k attention easily) ===
# No TPU chip isolation needed — vLLM owns all chips, training runs on CPU.
export TMX_DECOUPLED_TRAIN_CPU=1
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_DECOUPLED_TRAIN_DEBUG_PROGRESS=1
export TMX_OPTIM=adamw_torch

# === Decoupled training: no token chunking ===
export TMX_DECOUPLED_TRAIN_TOKEN_CHUNK=0
export TMX_DECOUPLED_GRAD_CKPT=0

# === vLLM for generation (ALL 16 v5litepod servers) ===
export TMX_GRPO_USE_VLLM=1
export TMX_TRAJ_USE_VLLM=1
export TMX_VLLM_SERVER_URLS="${TMX_VLLM_SERVER_URLS:?set TMX_VLLM_SERVER_URLS to the comma-separated vLLM server URLs}"
export TMX_VLLM_MODEL_NAME=qwen3_serious_policy
export TMX_VLLM_CONCURRENCY=64
export TMX_VLLM_MAX_RETRIES=3
export TMX_TRAJ_MAX_COMPLETION_LEN=8192

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
export TMX_GRPO_SAVE_TOTAL_LIMIT=10
export TMX_WANDB_MINIMAL=1
export TMX_NOVELTY_ON_TPU=0
export TMX_NOVELTY_BATCH_SIZE=16

# === CPU threading (use moderate threads to avoid contention) ===
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export PYTHONPATH="$HOME/two-model-exploration"

RUN_NAME="v6e_cpu_saturation_8k_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

echo "=== v6e CPU saturation: 8k tokens, 16 vLLM servers, CPU training ===" | tee "${RUN_DIR}/console.log"
echo "Run: ${RUN_NAME}" | tee -a "${RUN_DIR}/console.log"
date | tee -a "${RUN_DIR}/console.log"

python main.py \
  --device cpu \
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
  --lambda-novelty 0.03 \
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
  >> "${RUN_DIR}/console.log" 2>&1
