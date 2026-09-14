#!/usr/bin/env bash
# Vanilla GRPO baseline, multi-host JAX FSDP across train workers.
#
# Topology on a v5lite-64 (16 hosts × 4 chips):
#   TRAIN_WORKERS = 0,1,2,3   (4 hosts × 4 chips = 16 chips for FSDP train)
#   SERVE_WORKERS = 4..15     (12 hosts running vLLM at 8000)
#
# Mirrors `launch_jax_multihost_v5lite.sh` (codex's path) but flips the algorithmic
# knobs back to canonical vanilla GRPO (Shao 2024 / tunix reference):
#   - advantage = (r - group_mean) / (group_std + 1e-4)  ddof=1
#   - sequence-mean-token-mean loss aggregation
#   - symmetric PPO clip ε=0.2
#   - KL=0, no novelty, no dynamic sampling
#
# Usage:
#   PROJECT=<gcp-project> \
#   TPU_NAME=<tpu-vm-name> \
#   ZONE=<tpu-zone> \
#   REMOTE_REPO=/home/<user>/expdis \
#   bash expdis_jax/scripts/launch_vanilla_grpo_multihost.sh

set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to the GCP project of the TPU}"
TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:?set ZONE to the TPU zone}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3}"
SERVE_WORKERS="${SERVE_WORKERS:-4 5 6 7 8 9 10 11 12 13 14 15}"
RUN_NAME="${RUN_NAME:-vanilla_grpo_mh_$(date +%Y%m%d_%H%M%S)}"
COORD_PORT="${COORD_PORT:-12345}"
VENV_NAME="${VENV_NAME:-expdis_jax}"
REMOTE_REPO="${REMOTE_REPO:?set REMOTE_REPO to the absolute repo path on every TPU worker (e.g. /home/<user>/expdis)}"
SYNC_LOCAL_PACKAGE="${SYNC_LOCAL_PACKAGE:-1}"

DATASET="${DATASET:-deepscaler}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-100}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10}"
GRPO_LR="${GRPO_LR:-1e-6}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-16384}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-18432}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKER_IPS_RAW="$(
  gcloud --project="$PROJECT" alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --zone "$ZONE" \
    --format='get(networkEndpoints[].ipAddress)'
)"
IFS=';' read -r -a WORKER_IPS <<< "$WORKER_IPS_RAW"
if [ "${#WORKER_IPS[@]}" -lt 16 ]; then
  echo "ERROR: expected 16 TPU worker IPs, got ${#WORKER_IPS[@]}: $WORKER_IPS_RAW" >&2
  exit 1
fi

train_workers=($TRAIN_WORKERS)
serve_workers=($SERVE_WORKERS)

coord_worker="${train_workers[0]}"
coord_ip="${WORKER_IPS[$coord_worker]}"
train_worker_hosts=""
for w in "${train_workers[@]}"; do
  if [ -n "$train_worker_hosts" ]; then train_worker_hosts+=","; fi
  train_worker_hosts+="${WORKER_IPS[$w]}"
done

# Probe vLLM /health on each serve worker; only include healthy URLs.
echo "[mh-vanilla] probing vLLM serve workers..." >&2
vllm_urls=""
for w in "${serve_workers[@]}"; do
  ip="${WORKER_IPS[$w]}"
  if gcloud --project="$PROJECT" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
       --worker="$w" --zone="$ZONE" --tunnel-through-iap \
       --ssh-flag="-o ConnectTimeout=10" \
       --command='curl -fsS --max-time 3 http://localhost:8000/health >/dev/null && echo OK || echo DOWN' 2>/dev/null \
       | grep -q OK; then
    if [ -n "$vllm_urls" ]; then vllm_urls+=","; fi
    vllm_urls+="http://${ip}:8000/v1"
  fi
done
if [ -z "$vllm_urls" ]; then
  echo "ERROR: no healthy vLLM workers found in SERVE_WORKERS=$SERVE_WORKERS" >&2
  exit 1
fi

echo "[mh-vanilla] run=$RUN_NAME"
echo "[mh-vanilla] train_workers=[${train_workers[*]}] coord=${coord_ip}:${COORD_PORT}"
echo "[mh-vanilla] vllm_urls=$vllm_urls"

LOCAL_TMP_DIR="$(mktemp -d)"
trap "rm -rf '$LOCAL_TMP_DIR'" EXIT

for idx in "${!train_workers[@]}"; do
  w="${train_workers[$idx]}"
  worker_script="$LOCAL_TMP_DIR/${RUN_NAME}_w${w}.sh"
  cat > "$worker_script" <<INNER
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_REPO'
source ~/miniforge3/bin/activate '$VENV_NAME'

# Multi-host JAX env (matches launch_jax_multihost_v5lite.sh)
export WANDB_PROJECT='expdis'
$(if [ -n "${WANDB_ENTITY:-}" ]; then printf "export WANDB_ENTITY='%s'\n" "$WANDB_ENTITY"; fi)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PJRT_DEVICE=TPU
export TPU_ACCELERATOR_TYPE='v5litepod-64'
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
export EXPDIS_JAX_CLUSTER_DETECTION_METHOD='deactivate'

# Long-completion training contract
export EXPDIS_ALLOW_LONG_COMPLETION_TRAIN=1

# HuggingFace mirroring (matches codex setup so artifacts survive even if TPU dies)
export EXPDIS_HF_CHECKPOINT_REPO='SaifPunjwani/two-model-exploration-checkpoints'
export EXPDIS_HF_ARTIFACT_PREFIX='$RUN_NAME'
export EXPDIS_HF_MIRROR_CHECKPOINTS=1
export EXPDIS_HF_MIRROR_TRAJECTORIES=1
export EXPDIS_HF_MIRROR_METRICS=1
export EXPDIS_HF_MIRROR_VLLM_EXPORTS=1
export EXPDIS_HF_MIRROR_EVAL_PROBES=1
export EXPDIS_HF_MIRROR_RUN_SUMMARY=1
export EXPDIS_HF_MIRROR_FINAL_EXPORTS=1
export EXPDIS_HF_MIRROR_FINAL_EVAL=1
export EXPDIS_HF_MIRROR_ASYNC=1
export EXPDIS_ALLOW_GCS_ARTIFACTS=0
export EXPDIS_GCS_CHECKPOINT_BASE=''
export EXPDIS_REQUIRE_GCS_CHECKPOINT_UPLOAD=0
export EXPDIS_SAVE_BASE_POLICY_TRAJECTORIES=0

# Optional intermediate AIME probes (every 25 steps; cheap n=4)
export EXPDIS_AIME_PROBE_EVERY_STEPS=25
export EXPDIS_AIME_PROBE_NUM_PROBLEMS=6
export EXPDIS_AIME_PROBE_NUM_ROLLOUTS=4

python -m expdis_jax.train \
  --model-name '$MODEL_NAME' \
  --dataset-name '$DATASET' \
  --max-train-examples '$MAX_TRAIN_EXAMPLES' \
  --max-prompt-len '$MAX_PROMPT_LEN' \
  --max-completion-len '$MAX_COMPLETION_LEN' \
  --max-total-len '$MAX_TOTAL_LEN' \
  --grpo-batch-size 1 \
  --grpo-grad-accum 2 \
  --grpo-num-generations 8 \
  --grpo-max-steps '$GRPO_MAX_STEPS' \
  --grpo-loss-type dr_grpo \
  --grpo-loss-aggregation sequence-mean-token-mean \
  --optimizer adamw \
  --grpo-lr '$GRPO_LR' \
  --grpo-clip-epsilon 0.2 \
  --grpo-clip-epsilon-high 0.2 \
  --grpo-kl-beta 0.0 \
  --grpo-advantage-normalization std \
  --trajectory-batch-size 2 \
  --vllm-server-urls '$vllm_urls' \
  --vllm-concurrency 16 \
  --vllm-fanout-per-prompt 8 \
  --dynamic-sampling false \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --enable-thinking auto \
  --reward-correct 1.0 \
  --reward-incorrect -1.0 \
  --format-penalty-weight 0.0 \
  --soft-overlong-expected-len 0 \
  --soft-overlong-cache-len 0 \
  --lambda-novelty 0.0 \
  --novelty-zscore off \
  --correctness-zscore off \
  --rescore-chunk-size 1 \
  --logprob-chunk-tokens 64 \
  --benchmark-num-rollouts 32 \
  --benchmark-rollout-max-completion-len 32768 \
  --output-dir ~/runs/$RUN_NAME \
  --wandb-run-name '$RUN_NAME' \
  --save-every-steps '$SAVE_EVERY_STEPS' \
  --seed 0
INNER
  chmod +x "$worker_script"
  if [ "$SYNC_LOCAL_PACKAGE" = "1" ]; then
    gcloud --project="$PROJECT" alpha compute tpus tpu-vm scp --recurse \
      "$LOCAL_REPO_ROOT/expdis_jax" "$TPU_NAME:${REMOTE_REPO}/" \
      --worker="$w" --zone="$ZONE" --tunnel-through-iap >/dev/null
  fi
  gcloud --project="$PROJECT" alpha compute tpus tpu-vm scp \
    "$worker_script" "$TPU_NAME:/tmp/${RUN_NAME}_w${w}.sh" \
    --worker="$w" --zone="$ZONE" --tunnel-through-iap >/dev/null
  gcloud --project="$PROJECT" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker="$w" --zone="$ZONE" --tunnel-through-iap --command="
    set -euo pipefail
    pkill -9 -f '[v]llm serve' 2>/dev/null || true
    pkill -9 -f '[V]LLM::EngineCore' 2>/dev/null || true
    pkill -9 -f '[p]ython -m expdis_jax.train' 2>/dev/null || true
    tmux kill-session -t vllm 2>/dev/null || true
    tmux kill-session -t vllm_server 2>/dev/null || true
    fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
    rm -f /tmp/libtpu_lockfile 2>/dev/null || true
    sleep 3
    mkdir -p \$HOME/runs_logs
    mkdir -p ~/runs/$RUN_NAME
    chmod +x /tmp/${RUN_NAME}_w${w}.sh
    nohup /tmp/${RUN_NAME}_w${w}.sh \
      > \$HOME/runs_logs/${RUN_NAME}_w${w}.log 2>&1 < /dev/null &
    echo \"NOHUP_PID=\$!\"
    sleep 3
    head -3 \$HOME/runs_logs/${RUN_NAME}_w${w}.log 2>/dev/null
  "
done

echo "RUN_NAME=$RUN_NAME"
echo "COORDINATOR=${coord_ip}:${COORD_PORT}"
echo "VLLM_URLS=$vllm_urls"
echo "DATASET=$DATASET CONTEXT=${MAX_COMPLETION_LEN}/${MAX_TOTAL_LEN}"
echo "Run logs: tmux a -t $RUN_NAME on each train worker, or tail /tmp/<run>.sh.log"
