#!/usr/bin/env bash
set -euo pipefail

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-b}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3}"
SERVE_WORKERS="${SERVE_WORKERS:-4 5 6 7 8 9 10 11 12 13 14 15}"
RUN_NAME="${RUN_NAME:-jax_lambda0_multihost_$(date +%Y%m%d_%H%M%S)}"
COORD_PORT="${COORD_PORT:-12345}"
VENV_NAME="${VENV_NAME:-expdis_jax}"
REMOTE_REPO="${REMOTE_REPO:?set REMOTE_REPO to the absolute repo path on every TPU worker (e.g. /home/<user>/expdis)}"
SYNC_LOCAL_PACKAGE="${SYNC_LOCAL_PACKAGE:-1}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-500}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.0}"
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKER_IPS_RAW="$(
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --zone "$ZONE" \
    --format='get(networkEndpoints[].ipAddress)'
)"
IFS=';' read -r -a WORKER_IPS <<< "$WORKER_IPS_RAW"

if [ "${#WORKER_IPS[@]}" -lt 16 ]; then
  echo "ERROR: expected 16 TPU worker IPs, got ${#WORKER_IPS[@]} from: $WORKER_IPS_RAW" >&2
  exit 1
fi

train_workers=($TRAIN_WORKERS)
serve_workers=($SERVE_WORKERS)

coord_worker="${train_workers[0]}"
coord_ip="${WORKER_IPS[$coord_worker]}"
train_worker_hosts=""
for w in "${train_workers[@]}"; do
  if [ -n "$train_worker_hosts" ]; then
    train_worker_hosts+=","
  fi
  train_worker_hosts+="${WORKER_IPS[$w]}"
done
vllm_urls=""
for w in "${serve_workers[@]}"; do
  ip="${WORKER_IPS[$w]}"
  if gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$w" --zone="$ZONE" --tunnel-through-iap \
    --command='curl -sf --max-time 2 http://localhost:8000/health >/dev/null && echo OK || echo DOWN' 2>/dev/null | grep -q OK; then
    if [ -n "$vllm_urls" ]; then
      vllm_urls+=","
    fi
    vllm_urls+="http://${ip}:8000/v1"
  fi
done

if [ -z "$vllm_urls" ]; then
  echo "ERROR: no healthy vLLM workers found in SERVE_WORKERS=$SERVE_WORKERS" >&2
  exit 1
fi

LOCAL_TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$LOCAL_TMP_DIR"
}
trap cleanup EXIT

for idx in "${!train_workers[@]}"; do
  w="${train_workers[$idx]}"
  worker_script="$LOCAL_TMP_DIR/${RUN_NAME}_w${w}.sh"
  cat > "$worker_script" <<INNER
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_REPO'
source ~/miniforge3/bin/activate '$VENV_NAME'
export WANDB_PROJECT='expdis'
$(if [ -n "${WANDB_ENTITY:-}" ]; then printf "export WANDB_ENTITY='%s'\n" "$WANDB_ENTITY"; fi)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PJRT_DEVICE=TPU
export TPU_ACCELERATOR_TYPE='${TPU_ACCELERATOR_TYPE}'
export TPU_SKIP_MDS_QUERY=1
export TPU_VISIBLE_CHIPS='0,1,2,3'
export TPU_WORKER_ID='${idx}'
export TPU_WORKER_HOSTNAMES='${train_worker_hosts}'
export TPU_HOST_BOUNDS='2,2,1'
export TPU_PROCESS_BOUNDS='2,2,1'
export TPU_CHIPS_PER_HOST_BOUNDS='2,2,1'
export TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1'
export JAX_COORDINATOR_ADDRESS='${coord_ip}:${COORD_PORT}'
export JAX_NUM_PROCESSES='${#train_workers[@]}'
export JAX_PROCESS_ID='${idx}'
export JAX_LOCAL_DEVICE_IDS='0,1,2,3'
$(if [ "$idx" = "0" ]; then printf "export JAX_COORDINATOR_BIND_ADDRESS='0.0.0.0:%s'\n" "$COORD_PORT"; fi)
python -m expdis_jax.train \
  --model-name Qwen/Qwen3-1.7B \
  --dataset-name dapo_math_17k \
  --max-train-examples '$MAX_TRAIN_EXAMPLES' \
  --max-prompt-len 2048 \
  --max-completion-len 16384 \
  --max-total-len 18432 \
  --grpo-batch-size 1 \
  --grpo-grad-accum 2 \
  --grpo-num-generations 8 \
  --grpo-max-steps '$GRPO_MAX_STEPS' \
  --grpo-loss-type dr_grpo \
  --optimizer adamw \
  --grpo-lr 5e-6 \
  --grpo-clip-epsilon 0.2 \
  --grpo-kl-beta 0.0 \
  --trajectory-batch-size 2 \
  --vllm-server-urls '$vllm_urls' \
  --vllm-concurrency 16 \
  --vllm-fanout-per-prompt 8 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --enable-thinking auto \
  --reward-correct 1.0 \
  --reward-incorrect 0.0 \
  --format-penalty-weight 0.1 \
  --lambda-novelty '$LAMBDA_NOVELTY' \
  --rescore-chunk-size 1 \
  --logprob-chunk-tokens 64 \
  --output-dir ~/runs/$RUN_NAME \
  --wandb-run-name '$RUN_NAME' \
  --save-every-steps '$SAVE_EVERY_STEPS' \
  --seed 0
INNER
  chmod +x "$worker_script"
  if [ "$SYNC_LOCAL_PACKAGE" = "1" ]; then
    gcloud alpha compute tpus tpu-vm scp --recurse "$LOCAL_REPO_ROOT/expdis_jax" "$TPU_NAME:${REMOTE_REPO}/" \
      --worker="$w" --zone="$ZONE" --tunnel-through-iap >/dev/null
  fi
  gcloud alpha compute tpus tpu-vm scp "$worker_script" "$TPU_NAME:/tmp/${RUN_NAME}_w${w}.sh" \
    --worker="$w" --zone="$ZONE" --tunnel-through-iap >/dev/null
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$w" --zone="$ZONE" --tunnel-through-iap --command="
    set -euo pipefail
    pkill -9 -f '[v]llm serve' 2>/dev/null || true
    pkill -9 -f '[p]ython -m expdis_jax.train' 2>/dev/null || true
    tmux kill-session -t vllm 2>/dev/null || true
    tmux kill-session -t vllm_server 2>/dev/null || true
    tmux kill-session -t '$RUN_NAME' 2>/dev/null || true
    mkdir -p ~/runs/$RUN_NAME
    chmod +x /tmp/${RUN_NAME}_w${w}.sh
    tmux new-session -d -s '$RUN_NAME' /tmp/${RUN_NAME}_w${w}.sh
    tmux ls | grep '$RUN_NAME'
  "
done

echo "RUN_NAME=$RUN_NAME"
echo "COORDINATOR=${coord_ip}:${COORD_PORT}"
echo "VLLM_URLS=$vllm_urls"
