#!/usr/bin/env bash
set -eo pipefail
cd "$HOME/expdis"
source ~/.venv_tpu_slice/bin/activate 2>/dev/null || source .venv_tpu_slice/bin/activate 2>/dev/null || source ~/venv/bin/activate 2>/dev/null || true

export PYTHONUNBUFFERED=1
export WANDB_PROJECT=expdis
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false
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

# === Flash attention for GRPO; SFT uses eager (Pallas NaN on v6e post-compile) ===
export EXPDIS_XLA_FLASH_ATTENTION=1
export EXPDIS_XLA_FORCE_EAGER_ATTN=0
export EXPDIS_TPU_SLICE_LAUNCHED=1

# === SFT: Eager attention + Adafactor + 4k sequence padding ===
# v6e Pallas flash backward produces NaN on all tokens after XLA graph caching.
# v4 chips don't have this issue (flash at 8k works perfectly).
# Eager attention at 10k = 88.93G (24 layers × 3.2G attention matrices).
# At 4096 tokens: ~14G → fits in 32G v6e chips with model+optimizer.
# GRPO/generation/eval still use flash attention (forward-only, no NaN issue).
export EXPDIS_DISTILL_OPTIM=adafactor
export EXPDIS_DISTILL_GRAD_CKPT=1
export EXPDIS_DISTILL_CE_CHUNK=256
export EXPDIS_DISTILL_MAX_TOTAL_LEN=4096
export EXPDIS_DISTILL_EAGER_ATTN=1

# === Explorer GRPO: AdamW optimizer (better convergence than Adafactor) ===
export EXPDIS_OPTIM=adamw_torch

# === Decoupled training config ===
export EXPDIS_DECOUPLED_TRAIN_TOKEN_CHUNK=0
export EXPDIS_DECOUPLED_GRAD_CKPT=0
export EXPDIS_DECOUPLED_TRAIN_DEBUG_PROGRESS=1

# === Training controls ===
export EXPDIS_STOP_AFTER_EXPLORER=0
export EXPDIS_STOP_AFTER_TRAJECTORIES=0
export EXPDIS_SKIP_INTERMEDIATE_EVALS=1
export EXPDIS_SKIP_FINAL_BENCHMARK_EVALS=0
export EXPDIS_ABORT_ON_DEGENERATE_EXPLORER=0
export EXPDIS_ABORT_ON_BAD_TRAJ_WINDOW=0
export EXPDIS_GRPO_LOGGING_STRATEGY=steps
export EXPDIS_GRPO_LOGGING_STEPS=1
export EXPDIS_GRPO_SAVE_STRATEGY=steps
export EXPDIS_GRPO_SAVE_STEPS=50
export EXPDIS_GRPO_SAVE_TOTAL_LIMIT=3
export EXPDIS_WANDB_MINIMAL=0
export EXPDIS_NOVELTY_ON_TPU=1
export EXPDIS_NOVELTY_BATCH_SIZE=16
export EXPDIS_TRAJ_TARGET_ACCEPTED=2048
export EXPDIS_ACTUAL_GRPO_SAVE_STEPS=25
export EXPDIS_ACTUAL_GRPO_SAVE_TOTAL_LIMIT=4

# === No vLLM (self-contained on single chip) ===
export EXPDIS_GRPO_USE_VLLM=0
export EXPDIS_TRAJ_USE_VLLM=0
# Decoupled GRPO with on-device generation (no vLLM needed)
export EXPDIS_EXPLORER_DECOUPLED_GRPO=1
export EXPDIS_DECOUPLED_ONDEVICE_GEN_BATCH=2

# === Threading ===
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONPATH="$HOME/expdis"

# === W&B key ===
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_key 2>/dev/null || echo '')}"

# Detect local model path
if [ -d "$HOME/checkpoints/Qwen3-1.7B" ]; then
    MODEL_NAME="$HOME/checkpoints/Qwen3-1.7B"
elif [ -d "$HOME/.models/Qwen3-1.7B" ]; then
    MODEL_NAME="$HOME/.models/Qwen3-1.7B"
else
    MODEL_NAME=Qwen/Qwen3-1.7B
fi
echo "[orchestrator] Using model: $MODEL_NAME"

COMMON_ARGS=(
    --device tpu
    --use-wandb
    --seed 0
    --model-name "$MODEL_NAME"
    --dataset-name dapo_math_17k
    --benchmark-dataset-name aime_2024
    --benchmark-num-rollouts 32
    --benchmark-rollout-max-completion-len 32768
    --benchmark-eval-mode sampled
    --max-benchmark-examples 30
    --train-selection-policy random
    --train-probe-pool-size 256
    --train-probe-samples-per-prompt 4
    --train-probe-temperature 0.8
    --train-probe-batch-size 4
    --dataset-split-policy auto
    --topic-split-mode heuristic_math_topics
    --topic-partition-policy explicit_or_heuristic
    --decoupling-mode decoupled
    --base-eval-mode raw
    --actual-target-contract full_trace
    --max-train-examples 20000
    --max-eval-examples 32
    --grpo-max-steps 500
    --grpo-batch-size 1
    --grpo-grad-accum 2
    --grpo-loss-type dr_grpo
    --grpo-num-generations 8
    --grpo-generation-batch-size 8
    --num-samples-per-prompt 8
    --temperature-schedule "1.2,1.0,0.8"
    --answer-constraint-mode none
    --exploration-response-style reasoning_final
    --eval-response-style reasoning_final
    --exploration-answer-constraint-mode none
    --eval-answer-constraint-mode none
    --exploration-thinking-mode think
    --eval-thinking-mode think
    --actual-thinking-mode think
    --distill-epochs 2
    --distill-batch-size 1
    --max-distill-examples 4096
    --actual-train-mode sft_online_grpo
    --actual-online-grpo-max-steps 250
    --actual-reward-source correctness
    --actual-grpo-epochs 1
    --actual-grpo-batch-size 1
    --actual-grpo-grad-accum 2
    --actual-min-accepted-trajectories 4
    --actual-dataset-policy accepted_only
    --actual-max-incorrect-ratio 0.25
    --actual-target-correct-fraction 0.75
    --actual-grpo-kl-coef 0.10
    --actual-grpo-lr 1e-6
    --distill-lr 5e-6
    --max-actual-examples 4096
    --actual-init-source base
    --novelty-feature-source multilayer
    --novelty-feature-max-length 512
    --novelty-metric sqrt_mse
    --novelty-text-contract full_raw
    --rnd-hidden 512
    --reward-correct 1.0
    --reward-incorrect 0.0
    --reward-shape linear
    --anchor-metric none
    --trajectory-batch-size 16
    --eval-batch-size 16
    --benchmark-eval-batch-size 1
    --round-budget-mode constant_total
    --max-prompt-len 2048
    --max-completion-len 8192
    --exploration-max-completion-len 8192
    --eval-max-completion-len 8192
    --max-total-len 10240
    --format-penalty-weight 0.1
    --pipeline-mode two_model
    --incorrect-novelty-scale 0.0
)

run_pipeline() {
    local RUN_NAME="$1"
    local LAMBDA="$2"
    local EXTRA_ARGS=("${@:3}")

    local RUN_DIR="runs/${RUN_NAME}"
    mkdir -p "${RUN_DIR}"

    echo "[orchestrator] Starting ${RUN_NAME} (lambda=${LAMBDA}) at $(date -u)"

    python run_expdis_torch.py "${COMMON_ARGS[@]}" \
        --lambda-novelty "${LAMBDA}" \
        --novelty-zscore off \
        --wandb-run-name "${RUN_NAME}" \
        --plots-dir "${RUN_DIR}/plots" \
        --explorer-traj-path "${RUN_DIR}/explorer_trajectories.jsonl" \
        --explorer-output-dir "${RUN_DIR}/explorer_ckpts" \
        --saved-model-root "${RUN_DIR}/saved_models" \
        "${EXTRA_ARGS[@]}" \
        >> "${RUN_DIR}/console.log" 2>&1

    local EXIT_CODE=$?
    echo "[orchestrator] ${RUN_NAME} finished with exit code ${EXIT_CODE} at $(date -u)"

    # Save to GCS
    gsutil -m cp -r "${RUN_DIR}" "gs://two-model-exploration-checkpoints/${RUN_NAME}/" 2>&1 | tail -3

    return ${EXIT_CODE}
}

LOG="runs/orchestrator_v5.log"
mkdir -p runs
echo "[orchestrator] Starting run sequence v5 at $(date -u)" | tee -a "$LOG"
echo "[orchestrator] Fixes: DR-GRPO norm, PPO-clip Actual GRPO, logit shift, AdamW, raw_completions rewards" | tee -a "$LOG"

# ============================================================
# Run 1: Full pipeline lambda=0 (correctness only)
# Explorer GRPO 500 → Trajectories → SFT → Online GRPO 250 → AIME24 eval
# ============================================================
RUN1_NAME="v5_full_lambda0_$(date -u +%Y%m%d_%H%M%S)"
echo "[orchestrator] Run 1: ${RUN1_NAME} (full pipeline, lambda=0, AdamW)" | tee -a "$LOG"

if run_pipeline "${RUN1_NAME}" 0.0; then
    echo "[orchestrator] Run 1 SUCCESS" | tee -a "$LOG"
else
    echo "[orchestrator] Run 1 FAILED (exit=$?)" | tee -a "$LOG"
fi

sleep 10

# ============================================================
# Run 2: Full pipeline lambda=0.5 (with novelty)
# Explorer GRPO 500 → Trajectories → SFT → Online GRPO 250 → AIME24 eval
# ============================================================
RUN2_NAME="v5_full_lambda05_$(date -u +%Y%m%d_%H%M%S)"
echo "[orchestrator] Run 2: ${RUN2_NAME} (full pipeline, lambda=0.5, AdamW)" | tee -a "$LOG"

if run_pipeline "${RUN2_NAME}" 0.5; then
    echo "[orchestrator] Run 2 SUCCESS" | tee -a "$LOG"
else
    echo "[orchestrator] Run 2 FAILED (exit=$?)" | tee -a "$LOG"
fi

echo "[orchestrator] All runs complete at $(date -u)" | tee -a "$LOG"
