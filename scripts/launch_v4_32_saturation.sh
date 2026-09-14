#!/usr/bin/env bash
set -euo pipefail
# V4-32 Explorer Saturation Run
# Architecture: Worker 0 = FSDP training (4 chips), Workers 1-3 = vLLM (3 x TP4)
# TPU: a v4-32 slice (set TPU_NAME; zone us-central2-b), 4 workers x 4 v4 chips (32GB HBM each)
# Contract: 2048 prompt / 8192 completion / 10240 total / group_size=8
# Full DAPO, multilayer novelty, reasoning_final/think, full_trace targets.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh

# ── TPU target ──────────────────────────────────────────────────────────────
TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="us-central2-b"
TPU_TRANSPORT_FLAG="$(tmx_tpu_transport_flag)"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}"

# ── Run identity ────────────────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-v4_saturation_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="two-model-explore"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
RUN_DIR="runs/$RUN_NAME"

# ── Model / dataset ────────────────────────────────────────────────────────
MODEL_NAME="Qwen/Qwen3-1.7B"
SERVED_MODEL_NAME="qwen3_policy"
DATASET_NAME="dapo_math_17k"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-1000}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-32}"
TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

# ── vLLM config ─────────────────────────────────────────────────────────────
PORT=8000
TP_SIZE=4
MAX_MODEL_LEN=10240
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=65536

VLLM_HF_HOME="${REMOTE_ROOT}/.hf"
VLLM_HF_HUB_CACHE="${VLLM_HF_HOME}/hub"
VLLM_HF_DATASETS_CACHE="${VLLM_HF_HOME}/datasets"
VLLM_TRANSFORMERS_CACHE="${VLLM_HF_HUB_CACHE}"
VLLM_LOCAL_MODEL_ROOT="${VLLM_LOCAL_MODEL_ROOT:?set VLLM_LOCAL_MODEL_ROOT to the local model directory on the serving TPU VM}"
MODEL_TAG="Qwen3-1.7B"
VLLM_LOCAL_MODEL_DIR="${VLLM_LOCAL_MODEL_ROOT}/${MODEL_TAG}"

# ── Step controls (skip phases that already ran) ────────────────────────────
SYNC_CODE="${SYNC_CODE:-1}"
BOOTSTRAP_TRAINING="${BOOTSTRAP_TRAINING:-1}"
BOOTSTRAP_VLLM="${BOOTSTRAP_VLLM:-1}"
PREFETCH_MODEL="${PREFETCH_MODEL:-1}"
LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
LAUNCH_TRAINING="${LAUNCH_TRAINING:-1}"

log() { printf '[v4-32] %s\n' "$*"; }

describe_worker_ips() {
  local desc
  desc="$(tmx_gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format=json)"
  python3 - "$desc" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
eps = payload.get("networkEndpoints") or payload.get("workerNetworkEndpoints") or []
print(",".join(ep.get("ipAddress", "") for ep in eps if ep.get("ipAddress")))
PY
}

gcloud_ssh() {
  local worker="$1"; shift
  tmx_gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker="$worker" --zone="$ZONE" "$TPU_TRANSPORT_FLAG" \
    --command="$1"
}

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Sync code to all workers
# ═══════════════════════════════════════════════════════════════════════════
if [ "$SYNC_CODE" = "1" ]; then
  log "syncing code to all 4 workers"
  TPU_NAME="$TPU_NAME" ZONE="$ZONE" SYNC_WORKERS="all" ALLOW_DIRTY=1 \
    bash scripts/sync_tpu_slice_repo.sh
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Bootstrap training env on worker 0
# ═══════════════════════════════════════════════════════════════════════════
if [ "$BOOTSTRAP_TRAINING" = "1" ]; then
  log "bootstrapping training env on worker 0"
  gcloud_ssh 0 "cd '$REMOTE_ROOT' && VENV_DIR=.venv_tpu_slice USE_TPU=1 bash scripts/bootstrap_tpu_env.sh"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Bootstrap vLLM on workers 1-3 (parallel)
# ═══════════════════════════════════════════════════════════════════════════
if [ "$BOOTSTRAP_VLLM" = "1" ]; then
  log "bootstrapping vLLM on workers 1-3"
  for w in 1 2 3; do
    (
      gcloud_ssh "$w" "cd '$REMOTE_ROOT' && VLLM_VENV_DIR='$VLLM_VENV_DIR' bash scripts/bootstrap_vllm_tpu_env.sh"
    ) &
  done
  wait
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 3b: Patch JAX mesh_utils.py for v4 TPU (all vLLM workers)
# v4 has 2 TensorCores/chip but JAX 0.8.1 doesn't handle this in create_device_mesh.
# ═══════════════════════════════════════════════════════════════════════════
if [ "$BOOTSTRAP_VLLM" = "1" ]; then
  log "patching JAX mesh_utils.py for v4 on all workers"
  for w in 0 1 2 3; do
    gcloud_ssh "$w" "
      MESH_UTILS='$VLLM_VENV_DIR/lib/python3.11/site-packages/jax/_src/mesh_utils.py'
      if [ -f \"\$MESH_UTILS\" ]; then
        if grep -q '_TPU_V5P, _TPU_V4' \"\$MESH_UTILS\"; then
          echo 'worker $w: already patched'
        else
          sed -i 's/(device_kind in (_TPU_V5P,) and cores_per_chip == 2)/(device_kind in (_TPU_V5P, _TPU_V4) and cores_per_chip == 2)/' \"\$MESH_UTILS\"
          echo 'worker $w: patched'
        fi
      fi
    " &
  done
  wait
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Prefetch model on workers 1-3 (parallel)
# ═══════════════════════════════════════════════════════════════════════════
if [ "$PREFETCH_MODEL" = "1" ]; then
  log "prefetching model on workers 1-3"
  for w in 1 2 3; do
    (
      gcloud_ssh "$w" "cd '$REMOTE_ROOT' && \
        VLLM_VENV_DIR='$VLLM_VENV_DIR' \
        MODEL_NAME='$MODEL_NAME' \
        LOCAL_MODEL_ROOT='$VLLM_LOCAL_MODEL_ROOT' \
        VLLM_LOCAL_MODEL_DIR='$VLLM_LOCAL_MODEL_DIR' \
        VLLM_PREFETCH_RETRIES=3 \
        HF_HOME='$VLLM_HF_HOME' \
        HF_HUB_CACHE='$VLLM_HF_HUB_CACHE' \
        HF_DATASETS_CACHE='$VLLM_HF_DATASETS_CACHE' \
        TRANSFORMERS_CACHE='$VLLM_TRANSFORMERS_CACHE' \
        bash scripts/prepare_hf_model_snapshot.sh"
    ) &
  done
  wait
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Launch vLLM on workers 1-3
# ═══════════════════════════════════════════════════════════════════════════
if [ "$LAUNCH_VLLM" = "1" ]; then
  log "launching vLLM on workers 1-3"
  for w in 1 2 3; do
    session="vllm_${RUN_NAME}_w${w}"
    gcloud_ssh "$w" "
set -euo pipefail
cd '$REMOTE_ROOT'
tmux kill-server 2>/dev/null || true
sleep 2
cat > /tmp/${session}.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
export VLLM_VENV_DIR='$VLLM_VENV_DIR'
export MODEL_NAME='$VLLM_LOCAL_MODEL_DIR'
export SERVED_MODEL_NAME='$SERVED_MODEL_NAME'
export PORT='$PORT'
export TP_SIZE='$TP_SIZE'
export MAX_MODEL_LEN='$MAX_MODEL_LEN'
export MAX_NUM_SEQS='$MAX_NUM_SEQS'
export MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS'
export HF_HOME='$VLLM_HF_HOME'
export HF_HUB_CACHE='$VLLM_HF_HUB_CACHE'
export HF_DATASETS_CACHE='$VLLM_HF_DATASETS_CACHE'
export TRANSFORMERS_CACHE='$VLLM_TRANSFORMERS_CACHE'
export LOCAL_MODEL_ROOT='$VLLM_LOCAL_MODEL_ROOT'
export VLLM_LOCAL_MODEL_DIR='$VLLM_LOCAL_MODEL_DIR'
export VLLM_PREFETCH_MODEL=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export KV_CACHE_DTYPE=auto
bash scripts/run_vllm_server.sh > /tmp/vllm_${RUN_NAME}.log 2>&1
INNER
chmod +x /tmp/${session}.sh
tmux new-session -d -s '$session' 'bash /tmp/${session}.sh'
" &
  done
  wait
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Wait for vLLM health (workers 1-3 only)
# ═══════════════════════════════════════════════════════════════════════════
log "discovering worker IPs"
ALL_IPS="$(describe_worker_ips)"
# Workers 1-3 IPs (skip worker 0)
INFER_IPS="$(echo "$ALL_IPS" | tr ',' '\n' | tail -n +2 | tr '\n' ',' | sed 's/,$//')"
log "inference IPs: $INFER_IPS"

log "waiting for vLLM health on workers 1-3"
health_output="$(gcloud_ssh 0 "
IPS='$INFER_IPS'
PORT='$PORT'
for attempt in \$(seq 1 90); do
  ready=0
  urls=''
  IFS=',' read -r -a ip_array <<< \"\$IPS\"
  for ip in \"\${ip_array[@]}\"; do
    if curl -s --connect-timeout 3 \"http://\${ip}:\${PORT}/health\" >/dev/null 2>&1; then
      ready=\$((ready + 1))
      urls=\"\${urls:+\$urls,}http://\${ip}:\${PORT}/v1\"
    fi
  done
  echo \"[health] attempt \$attempt: \$ready/3 servers healthy\"
  if [ \"\$ready\" -ge 3 ]; then
    echo \"URLS=\$urls\"
    exit 0
  fi
  sleep 10
done
echo '[health] ERROR: vLLM not healthy after 90 attempts'
exit 1
")"

echo "$health_output" | grep -v '^URLS='
VLLM_SERVER_URLS="$(echo "$health_output" | grep '^URLS=' | head -1 | sed 's/^URLS=//')"
if [ -z "$VLLM_SERVER_URLS" ]; then
  log "ERROR: no healthy vLLM servers found"
  exit 1
fi
log "vLLM ready: $VLLM_SERVER_URLS"

# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Launch FSDP training on worker 0
# ═══════════════════════════════════════════════════════════════════════════
if [ "$LAUNCH_TRAINING" != "1" ]; then
  log "skipping training launch (LAUNCH_TRAINING=0)"
  exit 0
fi

TRAIN_ARGS=(
  main.py
  --device tpu
  --use-wandb
  --seed 0
  --model-name "$MODEL_NAME"
  --dataset-name "$DATASET_NAME"
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
  --max-train-examples "$MAX_TRAIN_EXAMPLES"
  --max-eval-examples "$MAX_EVAL_EXAMPLES"
  --grpo-max-steps "$GRPO_MAX_STEPS"
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
  --novelty-feature-source multilayer
  --novelty-feature-max-length 512
  --novelty-metric sqrt_mse
  --novelty-text-contract full_raw
  --rnd-hidden 512
  --reward-correct 1.0
  --reward-incorrect 0.0
  --lambda-novelty 0.03
  --reward-shape linear
  --anchor-metric none
  --trajectory-batch-size "$TRAJECTORY_BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --round-budget-mode constant_total
  --max-prompt-len 2048
  --max-completion-len 8192
  --exploration-max-completion-len 8192
  --eval-max-completion-len 8192
  --max-total-len 10240
  --format-penalty-weight 0.1
  --pipeline-mode two_model
  --actual-init-source base
  --actual-train-mode sft_online_grpo
  --actual-reward-source correctness
  --wandb-run-name "$RUN_NAME"
  --plots-dir "$RUN_DIR/plots"
  --explorer-traj-path "$RUN_DIR/explorer_trajectories.jsonl"
  --explorer-output-dir "$RUN_DIR/explorer_ckpts"
  --saved-model-root "$REMOTE_ROOT/$RUN_DIR/saved_models"
  --incorrect-novelty-scale 0.0
)

printf -v quoted_train_args '%q ' "${TRAIN_ARGS[@]}"

XLA_CACHE_DIR="/tmp/tmx_xla_cache_${RUN_NAME}"

log "launching FSDP training on worker 0 (4 chips)"
gcloud_ssh 0 "
set -euo pipefail
cd '$REMOTE_ROOT'
tmux kill-session -t 'training_$RUN_NAME' 2>/dev/null || true
pkill -9 -f 'from multiprocessing.spawn import spawn_mai[n]' 2>/dev/null || true
pkill -9 -f 'from multiprocessing.resource_tracker import mai[n]' 2>/dev/null || true
sleep 2
mkdir -p '$REMOTE_ROOT/$RUN_DIR'
cat > /tmp/training_${RUN_NAME}.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
source .venv_tpu_slice/bin/activate
export WANDB_PROJECT='$WANDB_PROJECT'
export WANDB_ENTITY='$WANDB_ENTITY'
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# FSDP env
export PJRT_DEVICE=TPU
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_REQUIRE_TPU=1
export TMX_FSDP=1
export TMX_FSDP_WRAP_CLS=Qwen3DecoderLayer
export TMX_USE_XLA_WORLD_SIZE=1
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE=v4-32
export FSDP_VISIBLE_CHIPS=0,1,2,3
export FSDP_NUM_PROCESSES=4
export TMX_XLA_PERSISTENT_CACHE_DIR='$XLA_CACHE_DIR'
export TMX_XLA_SAFE_GENERATE=1
export TMX_XLA_INCREMENTAL_GENERATE=1
export TMX_XLA_SKIP_LOADER_MARK_STEP=1
# Training env
export TMX_GRPO_GRAD_CKPT=1
export TMX_STRICT_GRPO_GEOMETRY=1
export TMX_DECOUPLED_GRAD_CKPT=1
export TMX_DECOUPLED_TRAIN_CPU=0
export TMX_TPU_HOST_TORCH_THREADS=16
export TMX_TPU_HOST_TORCH_INTEROP_THREADS=2
export TMX_GRPO_LOGPROB_CHUNK_SIZE=2
export TMX_NOVELTY_BATCH_SIZE=32
export TMX_NOVELTY_ON_TPU=0
export TMX_EXPLORER_DECOUPLED_BUFFER_GROUPS=32
export TMX_EXPLORER_DECOUPLED_BATCH_GROUPS=1
export TMX_EXPLORER_DECOUPLED_GRAD_ACCUM=2
export TMX_EXPLORER_DECOUPLED_ROW_MICROBATCH=1
export TMX_EXPLORER_DECOUPLED_TRAIN_FSDP=1
export TMX_EXPLORER_DECOUPLED_TRAIN_FSDP_GRAD_CKPT=1
export TMX_DECOUPLED_ROW_MARK_STEP=1
export TMX_DECOUPLED_TRAIN_FIXED_SHAPES=1
export TMX_DECOUPLED_TRAIN_FIXED_PROMPT_WIDTH=512
export TMX_DECOUPLED_TRAIN_FIXED_COMPLETION_WIDTH=8192
export TMX_WANDB_MINIMAL=1
export TMX_DISABLE_BF16=0
export TMX_TRAJ_USE_VLLM=1
export TMX_TRAJ_MAX_COMPLETION_LEN=8192
export TMX_GRPO_LOGGING_STRATEGY=steps
export TMX_GRPO_LOGGING_STEPS=1
export TMX_GRPO_SAVE_STRATEGY=steps
export TMX_GRPO_SAVE_STEPS=25
export TMX_GRPO_SAVE_TOTAL_LIMIT=10
export TMX_ACTUAL_GRPO_SAVE_STEPS=25
export TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT=4
export TMX_SKIP_INTERMEDIATE_EVALS=1
export TMX_SKIP_FINAL_BENCHMARK_EVALS=1
export TMX_STOP_AFTER_EXPLORER=1
export TMX_ABORT_ON_DEGENERATE_EXPLORER=1
export TMX_ABORT_ON_BAD_TRAJ_WINDOW=1
export TMX_VLLM_SERVER_URLS='$VLLM_SERVER_URLS'
export TMX_VLLM_MODEL_NAME='$SERVED_MODEL_NAME'
export TMX_VLLM_CONCURRENCY=16
export TMX_VLLM_MAX_RETRIES=3
export TMX_GCS_CHECKPOINT_BASE=gs://llmvae-checkpoints/two-model-exploration/checkpoints
export TMX_EXPLORER_VLLM_RELOAD_TPU_NAME='$TPU_NAME'
export TMX_EXPLORER_VLLM_RELOAD_ZONE='$ZONE'
export TMX_EXPLORER_VLLM_RELOAD_WORKERS=1,2,3
export TMX_EXPLORER_VLLM_MODEL_NAME=explorer_model
export TMX_VLLM_RELOAD_TPU_NAME='$TPU_NAME'
export TMX_VLLM_RELOAD_ZONE='$ZONE'
export TMX_VLLM_RELOAD_WORKER=1
export TMX_ACTUAL_VLLM_MODEL_NAME=actual_model
export PORT='$PORT'
export TP_SIZE='$TP_SIZE'
export MAX_MODEL_LEN='$MAX_MODEL_LEN'
export MAX_NUM_SEQS='$MAX_NUM_SEQS'
export MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS'
export VLLM_VENV_DIR='$VLLM_VENV_DIR'
mkdir -p '$REMOTE_ROOT/$RUN_DIR'
bash scripts/run_tpu_fsdp_local.sh $quoted_train_args 2>&1 | tee -a '$REMOTE_ROOT/$RUN_DIR/console.log'
INNER
chmod +x /tmp/training_${RUN_NAME}.sh
tmux new-session -d -s 'training_$RUN_NAME' 'bash /tmp/training_${RUN_NAME}.sh'
tmux ls | grep 'training_$RUN_NAME' || true
"

log "launched: run=$RUN_NAME"
log "monitor: gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=0 --zone=$ZONE $TPU_TRANSPORT_FLAG --command=\"tmux capture-pane -t training_$RUN_NAME -p -S -80\""
log "console: gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=0 --zone=$ZONE $TPU_TRANSPORT_FLAG --command=\"tail -100 $REMOTE_ROOT/$RUN_DIR/console.log\""
