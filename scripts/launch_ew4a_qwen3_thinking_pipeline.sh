#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# Full thinking-mode two-model pipeline on ew4a (v6e-64 = 64 chips, 16 hosts)
#
# Architecture (optimized — NO CPU training):
#   Worker 0:    TPU training (4 chips, Explorer GRPO + Actual SFT+GRPO)
#   Workers 1-15: vLLM servers (60 chips, TP=4 each → 15 parallel servers)
#                 Used for trajectory collection + eval via HTTP
#
# Thinking mode:
#   Explorer generates with <think>...</think> (exploration_thinking_mode=think)
#   Actual trains on full_trace with thinking (actual_thinking_mode=think)
#   Eval uses thinking (eval_thinking_mode=think)
#
# Usage:
#   bash scripts/launch_ew4a_qwen3_thinking_pipeline.sh
#
# Env overrides:
#   GRPO_MAX_STEPS=32       Explorer GRPO steps (default: 24)
#   MAX_TRAIN_EXAMPLES=512  Training prompts (default: 256)
#   SKIP_VLLM_LAUNCH=1      Skip vLLM server launch (if already running)
#   SKIP_SYNC=1              Skip repo sync
#   VLLM_WORKERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"  Which workers run vLLM
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-a}"
TPU_TRANSPORT_FLAG="$(tmx_tpu_transport_flag)"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}"
VENV_DIR="${VENV_DIR:-.venv_tpu_slice}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATASET_NAME="${DATASET_NAME:-dapo_math_17k}"
QUEUE_TAG="${QUEUE_TAG:-ew4a_qwen3_thinking_$(date +%Y%m%d_%H%M%S)}"
SESSION_PREFIX="${SESSION_PREFIX:-ew4a_think}"
RUN_ROOT="${RUN_ROOT:-runs/${QUEUE_TAG}}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:?set SAVED_MODEL_ROOT to the saved-model directory on the TPU VM}"
RUN_ARTIFACT_GCS_PREFIX="${RUN_ARTIFACT_GCS_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/${QUEUE_TAG}}"

if [ "${TMX_ALLOW_LEGACY_QWEN3_THINKING_PIPELINE:-0}" != "1" ]; then
  cat >&2 <<'EOF'
[ew4a-think] deprecated launcher: this is not the canonical serious-run path.
[ew4a-think] use one of:
[ew4a-think]   bash scripts/launch_qwen3_explorer_novelty_saturation.sh
[ew4a-think]   bash scripts/launch_qwen3_one_round_full_dapo.sh
[ew4a-think] set TMX_ALLOW_LEGACY_QWEN3_THINKING_PIPELINE=1 only if you intentionally want the old path.
EOF
  exit 2
fi

# ── Worker allocation ──
# Worker 0 = training (TPU), Workers 1-15 = vLLM inference
TRAIN_WORKER=0
VLLM_WORKERS=(${VLLM_WORKERS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15})

# ── Pipeline parameters (thinking-optimized) ──
# With TPU training, Explorer GRPO is fast enough for more steps.
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-1000}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-64}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-8}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-8}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-8}"
# More samples per prompt = more chances of getting correct trajectories on hard
# DAPO problems. 8 samples means even a 12.5% solve rate yields ~1 correct.
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-8}"
GRPO_LOSS_TYPE="${GRPO_LOSS_TYPE:-dr_grpo}"
TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

# ── Completion lengths ──
# With TPU training, Explorer GRPO can afford longer completions.
# Trajectory collection via vLLM still gets even longer for full thinking traces.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-8192}"
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-2048}"
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-8192}"
# Trajectory collection override: full thinking traces for Actual training data.
TRAJ_MAX_COMPLETION_LEN="${TRAJ_MAX_COMPLETION_LEN:-8192}"
# Distillation max_total_len must accommodate prompt + full thinking completion.
# Pad generously beyond TRAJ_MAX_COMPLETION_LEN + MAX_PROMPT_LEN to avoid
# truncating \boxed{} at the end of long thinking traces.
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-9216}"

# ── Thinking mode controls ──
EXPLORATION_THINKING_MODE="think"
EVAL_THINKING_MODE="think"
# Actual trains on full thinking traces (full_trace contract) so it learns
# to produce <think>...</think> reasoning, which is critical for AIME.
ACTUAL_THINKING_MODE="think"

# ── Temperature ──
# Higher temperature (1.0-1.2) during trajectory collection produces more
# diverse reasoning paths → richer training signal for Actual.
TEMPERATURE_SCHEDULE="${TEMPERATURE_SCHEDULE:-1.2,1.0,0.8}"

# ── Actual model training ──
DISTILL_EPOCHS="${DISTILL_EPOCHS:-3}"
MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-1024}"
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-1}"
ACTUAL_GRPO_BATCH_SIZE="${ACTUAL_GRPO_BATCH_SIZE:-1}"
ACTUAL_GRPO_GRAD_ACCUM="${ACTUAL_GRPO_GRAD_ACCUM:-8}"
ACTUAL_MIN_ACCEPTED_TRAJECTORIES="${ACTUAL_MIN_ACCEPTED_TRAJECTORIES:-8}"
MAX_ACTUAL_EXAMPLES="${MAX_ACTUAL_EXAMPLES:-2048}"
ACTUAL_TRAIN_MODE="${ACTUAL_TRAIN_MODE:-sft_online_grpo}"

# ── Probing (solvable subset selection) ──
TRAIN_PROBE_POOL_SIZE="${TRAIN_PROBE_POOL_SIZE:-512}"
TRAIN_PROBE_SAMPLES_PER_PROMPT="${TRAIN_PROBE_SAMPLES_PER_PROMPT:-2}"
TRAIN_PROBE_BATCH_SIZE="${TRAIN_PROBE_BATCH_SIZE:-4}"

# ── vLLM server settings (per worker) ──
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

# ── Flags ──
SKIP_VLLM_LAUNCH="${SKIP_VLLM_LAUNCH:-0}"
SKIP_SYNC="${SKIP_SYNC:-0}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"

log() { printf '[ew4a-think] %s\n' "$*"; }

gcloud_ssh() {
  local worker="$1"; shift
  tmx_gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker="$worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="$*"
}

gcloud_ssh_batch() {
  local workers="$1"; shift
  tmx_gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker="$workers" \
    --batch-size=all \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="$*"
}

# ── Step 1: Sync repo ──
if [ "$SKIP_SYNC" != "1" ]; then
  log "syncing repo to all workers"
  TPU_NAME="$TPU_NAME" ZONE="$ZONE" SYNC_WORKERS=all ALLOW_DIRTY=1 \
    bash scripts/sync_tpu_slice_repo.sh
fi

# ── Step 2: Bootstrap vLLM env on vLLM workers ──
if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  VLLM_WORKER_LIST=$(IFS=,; echo "${VLLM_WORKERS[*]}")
  log "bootstrapping vllm env on workers: $VLLM_WORKER_LIST"
  gcloud_ssh_batch "$VLLM_WORKER_LIST" \
    "cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' bash scripts/bootstrap_vllm_tpu_env.sh"
fi

# ── Step 3: Launch vLLM on workers 1-15 (NOT worker 0 — reserved for training) ──
if [ "$SKIP_VLLM_LAUNCH" != "1" ]; then
  log "launching vLLM servers on workers: ${VLLM_WORKERS[*]}"
  for worker in "${VLLM_WORKERS[@]}"; do
    gcloud_ssh "$worker" "
      cd '$REMOTE_ROOT'
      pkill -f 'vllm serve' 2>/dev/null || true
      sleep 1
      source '$VLLM_VENV_DIR/bin/activate'
      MODEL_NAME='$MODEL_NAME' \
      SERVED_MODEL_NAME='$MODEL_NAME' \
      PORT='$PORT' \
      TP_SIZE='$TP_SIZE' \
      MAX_MODEL_LEN='$MAX_MODEL_LEN' \
      MAX_NUM_SEQS='$MAX_NUM_SEQS' \
      MAX_NUM_BATCHED_TOKENS=65536 \
      nohup bash scripts/run_vllm_server.sh > /tmp/vllm_thinking_pipeline.log 2>&1 &
      echo 'vLLM launched on worker $worker'
    " &
  done
  wait
  log "all vLLM servers launching, waiting for health..."

  # Wait for at least half the servers to be healthy
  MIN_HEALTHY=$(( ${#VLLM_WORKERS[@]} / 2 ))
  for attempt in $(seq 1 60); do
    healthy=0
    for worker in "${VLLM_WORKERS[@]}"; do
      if gcloud_ssh "$worker" "curl -sf http://localhost:$PORT/health >/dev/null 2>&1" 2>/dev/null; then
        healthy=$((healthy + 1))
      fi
    done
    if [ "$healthy" -ge "$MIN_HEALTHY" ]; then
      log "$healthy/${#VLLM_WORKERS[@]} vLLM servers healthy after ${attempt}0s"
      break
    fi
    if [ "$attempt" = "60" ]; then
      log "ERROR: only $healthy servers healthy after 600s (need $MIN_HEALTHY)"
      exit 1
    fi
    log "  $healthy/${#VLLM_WORKERS[@]} healthy (need $MIN_HEALTHY)..."
    sleep 10
  done
fi

# ── Step 4: Collect internal IPs of vLLM workers for URL list ──
VLLM_URLS=""
for worker in "${VLLM_WORKERS[@]}"; do
  ip=$(gcloud_ssh "$worker" "hostname -I | awk '{print \$1}'" 2>/dev/null | tr -d '\r\n')
  if [ -n "$ip" ]; then
    url="http://${ip}:${PORT}/v1"
    VLLM_URLS="${VLLM_URLS:+$VLLM_URLS,}${url}"
  fi
done
log "vLLM URLs: $VLLM_URLS"

# ── Step 5: Kill any vLLM on worker 0 (free TPU for training) ──
log "ensuring worker 0 TPU is free for training"
gcloud_ssh "$TRAIN_WORKER" "
  pkill -f 'vllm serve' 2>/dev/null || true
  ray stop -f 2>/dev/null || true
  sleep 2
  echo 'worker 0 TPU freed'
"

# ── Step 6: Launch training pipeline on worker 0 (TPU!) ──
SESSION_NAME="${SESSION_PREFIX}_pipeline"
log "launching full thinking pipeline on worker 0 TPU in tmux session: $SESSION_NAME"

gcloud_ssh "$TRAIN_WORKER" "
  cd '$REMOTE_ROOT'
  mkdir -p '$RUN_ROOT'
  tmux kill-session -t '$SESSION_NAME' 2>/dev/null || true
  tmux new-session -d -s '$SESSION_NAME' \"
    cd '$REMOTE_ROOT' && \\
    source '$VENV_DIR/bin/activate' && \\
    export PJRT_DEVICE=TPU && \\
    export TMX_GRPO_GRAD_CKPT=1 && \\
    export TMX_XLA_SAFE_GENERATE=1 && \\
    export TMX_DISABLE_BF16=0 && \\
    export TMX_XLA_SKIP_LOADER_MARK_STEP=1 && \\
    export TMX_SKIP_INTERMEDIATE_EVALS=1 && \\
    export TMX_SKIP_FINAL_BENCHMARK_EVALS=0 && \\
    export TMX_GRPO_SAVE_STRATEGY=steps && \\
    export TMX_GRPO_SAVE_STEPS=8 && \\
    export TMX_GRPO_SAVE_TOTAL_LIMIT=3 && \\
    export TMX_ACTUAL_GRPO_SAVE_STEPS='${TMX_ACTUAL_GRPO_SAVE_STEPS:-25}' && \\
    export TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT='${TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT:-4}' && \\
    export TMX_GRPO_LOGGING_STRATEGY=steps && \\
    export TMX_GRPO_LOGGING_STEPS=1 && \\
    export TMX_ABORT_ON_DEGENERATE_EXPLORER=1 && \\
    export TMX_TRAJ_HEALTH_WINDOW_PROMPTS=32 && \\
    export TMX_TRAJ_HEALTH_MIN_CORRECT=1 && \\
    export TMX_TRAJ_USE_VLLM=1 && \\
    export TMX_TRAJ_MAX_COMPLETION_LEN='$TRAJ_MAX_COMPLETION_LEN' && \\
    export TMX_VLLM_SERVER_URLS='$VLLM_URLS' && \\
    export TMX_VLLM_MODEL_NAME='$MODEL_NAME' && \\
    export TMX_VLLM_CONCURRENCY=64 && \\
    export RUN_ARTIFACT_GCS_PREFIX='$RUN_ARTIFACT_GCS_PREFIX' && \\
    export RUN_SYNC_INTERVAL_SECONDS=90 && \\
    python main.py \\
      --model-name '$MODEL_NAME' \\
      --dataset-name '$DATASET_NAME' \\
      --device tpu \\
      --seed 42 \\
      --pipeline-mode two_model \\
      --decoupling-mode decoupled \\
      --base-eval-mode raw \\
      --train-selection-policy random \\
      --train-probe-pool-size $TRAIN_PROBE_POOL_SIZE \\
      --train-probe-samples-per-prompt $TRAIN_PROBE_SAMPLES_PER_PROMPT \\
      --train-probe-batch-size $TRAIN_PROBE_BATCH_SIZE \\
      --max-train-examples $MAX_TRAIN_EXAMPLES \\
      --max-eval-examples $MAX_EVAL_EXAMPLES \\
      --grpo-max-steps $GRPO_MAX_STEPS \\
      --grpo-batch-size $GRPO_BATCH_SIZE \\
      --grpo-grad-accum $GRPO_GRAD_ACCUM \\
      --grpo-loss-type $GRPO_LOSS_TYPE \\
      --grpo-num-generations $GRPO_NUM_GENERATIONS \\
      --grpo-generation-batch-size $GRPO_GENERATION_BATCH_SIZE \\
      --num-samples-per-prompt $NUM_SAMPLES_PER_PROMPT \\
      --max-prompt-len $MAX_PROMPT_LEN \\
      --max-completion-len $MAX_COMPLETION_LEN \\
      --exploration-max-completion-len $EXPLORATION_MAX_COMPLETION_LEN \\
      --eval-max-completion-len $EVAL_MAX_COMPLETION_LEN \\
      --max-total-len $MAX_TOTAL_LEN \\
      --temperature-schedule '$TEMPERATURE_SCHEDULE' \\
      --exploration-thinking-mode $EXPLORATION_THINKING_MODE \\
      --eval-thinking-mode $EVAL_THINKING_MODE \\
      --actual-thinking-mode $ACTUAL_THINKING_MODE \\
      --exploration-response-style reasoning_final \\
      --eval-response-style reasoning_final \\
      --exploration-answer-constraint-mode none \\
      --eval-answer-constraint-mode none \\
      --answer-constraint-mode none \\
      --actual-target-contract full_trace \\
      --actual-init-source base \\
      --actual-train-mode $ACTUAL_TRAIN_MODE \\
      --actual-reward-source correctness \\
      --actual-dataset-policy accepted_only \\
      --actual-max-incorrect-ratio 0.0 \\
      --actual-target-correct-fraction 1.00 \\
      --actual-grpo-kl-coef 0.10 \\
      --actual-grpo-lr 1e-6 \\
      --actual-grpo-batch-size $ACTUAL_GRPO_BATCH_SIZE \\
      --actual-grpo-grad-accum $ACTUAL_GRPO_GRAD_ACCUM \\
      --actual-grpo-epochs $ACTUAL_GRPO_EPOCHS \\
      --actual-min-accepted-trajectories $ACTUAL_MIN_ACCEPTED_TRAJECTORIES \\
      --max-actual-examples $MAX_ACTUAL_EXAMPLES \\
      --distill-epochs $DISTILL_EPOCHS \\
      --distill-lr 5e-6 \\
      --distill-batch-size 1 \\
      --max-distill-examples $MAX_DISTILL_EXAMPLES \\
      --lambda-novelty 0.03 \\
      --novelty-feature-source multilayer \\
      --novelty-metric sqrt_mse \\
      --novelty-text-contract full_raw \\
      --rnd-hidden 512 \\
      --reward-correct 1.0 \\
      --reward-incorrect 0.0 \\
      --reward-shape linear \\
      --incorrect-novelty-scale 0.0 \\
      --trajectory-batch-size $TRAJECTORY_BATCH_SIZE \\
      --eval-batch-size $EVAL_BATCH_SIZE \\
      --round-budget-mode constant_total \\
      --benchmark-dataset-name aime_2024 \\
      --benchmark-eval-mode sampled \\
      --benchmark-num-rollouts 32 \\
      --benchmark-rollout-temperature 1.0 \\
      --benchmark-rollout-max-completion-len 16000 \\
      --use-wandb \\
      --wandb-run-name '${QUEUE_TAG}' \\
      2>&1 | tee '$RUN_ROOT/pipeline.log' && \\
    echo '=== Pipeline complete, starting AIME24 avg@32 eval ===' && \\
    MODEL_PATH=/tmp/tmx_model_save_actual \\
    EVAL_LABEL='${QUEUE_TAG}_actual' \\
    VLLM_VENV_DIR='$VLLM_VENV_DIR' \\
    REPO_ROOT='$REMOTE_ROOT' \\
    PORT='$PORT' \\
    NUM_ROLLOUTS=32 \\
    TEMPERATURE=1.0 \\
    MAX_TOKENS=16000 \\
    MAX_MODEL_LEN=16384 \\
    SERVED_MODEL_NAME='actual_model' \\
    CONCURRENCY=16 \\
    RESULTS_DIR='$REMOTE_ROOT/$RUN_ROOT' \\
    bash scripts/run_aime24_avg32_eval.sh \\
    2>&1 | tee '$RUN_ROOT/actual_eval.log'
  \"
"

log "=========================================="
log "Pipeline launched!"
log "  Training: worker 0 (TPU, 4 chips)"
log "  Inference: workers ${VLLM_WORKERS[*]} (${#VLLM_WORKERS[@]}×4 = $((${#VLLM_WORKERS[@]} * 4)) chips)"
log "=========================================="
log "Monitor:  ssh $TPU_NAME 'tmux capture-pane -t $SESSION_NAME -p -S -50'"
log "vLLM log: ssh $TPU_NAME 'tail -f /tmp/vllm_thinking_pipeline.log'"
log "GCS:      $RUN_ARTIFACT_GCS_PREFIX"
