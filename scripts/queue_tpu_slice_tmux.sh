#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

. scripts/tpu_target_lib.sh
resolve_tpu_target_defaults
resolve_tpu_worker_env

TPU_NAME="${TPU_NAME}"
ZONE="${ZONE}"
TPU_TRANSPORT_FLAG="$(expdis_tpu_transport_flag)"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VENV_DIR="${VENV_DIR:-.venv_tpu_slice}"
SESSION_PREFIX="${SESSION_PREFIX:-tpu_slice}"
QUEUE_TAG="${QUEUE_TAG:-tpu_slice_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/${QUEUE_TAG}}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
LOCAL_CHIPS="${LOCAL_CHIPS:-4}"
WORKERS=(${WORKERS})
RETRY_ON_FAILURE="${RETRY_ON_FAILURE:-1}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-15}"

export WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"

ensure_gcloud_ssh_key() {
  local key_path="${HOME}/.ssh/google_compute_engine"
  if [ ! -f "$key_path" ]; then
    return
  fi
  if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l >/dev/null 2>&1; then
    eval "$(ssh-agent -s)" >/dev/null
  fi
  ssh-add "$key_path" >/dev/null 2>&1 || true
}

ensure_gcloud_ssh_key

kill_existing_sessions() {
  pkill -f 'python -u run_expdis_torch.py --device [t]pu' || true
  pkill -f 'python run_expdis_torch.py --device [t]pu' || true
  pkill -f 'python -c import torch_xla.core.xla_model as x[m]' || true
  tmux ls 2>/dev/null | sed 's/:.*//' | grep '^'"${SESSION_PREFIX}"'_' | xargs -r -n1 tmux kill-session -t || true
  local worker
  for worker in "${WORKERS[@]}"; do
    expdis_gcloud alpha compute tpus tpu-vm ssh \
      "$TPU_NAME" \
      --worker="$worker" \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG" \
      --command="pkill -f 'python -u run_expdis_torch.py --device [t]pu' || true; pkill -f 'python run_expdis_torch.py --device [t]pu' || true; pkill -f 'python -c import torch_xla.core.xla_model as x[m]' || true; tmux ls 2>/dev/null | sed 's/:.*//' | grep '^${SESSION_PREFIX}_' | xargs -r -n1 tmux kill-session -t || true"
  done
}

if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  kill_existing_sessions
  bash scripts/bootstrap_tpu_slice.sh
fi

mkdir -p "$RUN_ROOT"

COMMON_ARGS=(
  --device tpu
  --use-wandb
  --dataset-split-policy auto
  --topic-split-mode heuristic_math_topics
  --topic-partition-policy explicit_or_heuristic
  --decoupling-mode decoupled
  --actual-train-mode sft_grpo
  --actual-target-contract canonical
  --actual-reward-source correctness
  --base-eval-mode grpo
  --max-train-examples 128
  --max-eval-examples 64
  --grpo-max-steps 8
  --grpo-batch-size 1
  --grpo-num-generations 2
  --grpo-generation-batch-size 2
  --num-samples-per-prompt 2
  --distill-epochs 1
  --distill-batch-size 1
  --max-distill-examples 128
  --actual-grpo-epochs 1
  --actual-grpo-batch-size 1
  --actual-grpo-grad-accum 1
  --max-actual-examples 128
  --lambda-novelty 0.1
  --novelty-feature-source multilayer
  --novelty-metric sqrt_mse
  --novelty-text-contract full_raw
  --format-penalty-weight 0.25
)

RUN_NAMES=(
  "${QUEUE_TAG}_gsm8k_qwen15_base_p192_c16"
  "${QUEUE_TAG}_gsm8k_qwen15_two_l005_p192_c16"
  "${QUEUE_TAG}_gsm8k_qwen15_two_l010_p192_c16"
  "${QUEUE_TAG}_gsm8k_qwen15_two_l015_p192_c16"
  "${QUEUE_TAG}_gsm8k_qwen15_two_l010_p192_c24"
  "${QUEUE_TAG}_gsm8k_qwen15_two_outkl_p192_c16"
  "${QUEUE_TAG}_gsm8k_qwen15_two_paraml2_p192_c16"
  "${QUEUE_TAG}_gsm8k_qwen15_multiround_p192_c16"
  "${QUEUE_TAG}_deepscaler_qwen15_base_p224_c24"
  "${QUEUE_TAG}_deepscaler_qwen15_two_l010_p224_c24"
  "${QUEUE_TAG}_deepscaler_qwen15_two_l010_p224_c32"
  "${QUEUE_TAG}_deepscaler_qwen15_two_l015_p224_c24"
  "${QUEUE_TAG}_deepscaler_qwen15_two_outkl_p224_c24"
  "${QUEUE_TAG}_deepscaler_qwen15_two_paraml2_p224_c24"
  "${QUEUE_TAG}_deepscaler_qwen15_multiround_p224_c24"
  "${QUEUE_TAG}_dapo_qwen15_base_p288_c24"
  "${QUEUE_TAG}_dapo_qwen15_two_l010_p288_c24"
  "${QUEUE_TAG}_dapo_qwen15_two_l010_p320_c32"
  "${QUEUE_TAG}_dapo_qwen15_two_l015_p288_c24"
  "${QUEUE_TAG}_dapo_qwen15_two_outkl_p288_c24"
  "${QUEUE_TAG}_dapo_qwen15_two_paraml2_p288_c24"
  "${QUEUE_TAG}_dapo_qwen15_multiround_p288_c24"
  "${QUEUE_TAG}_gsm8k_qwen15_two_l010_p160_c16"
  "${QUEUE_TAG}_gsm8k_qwen15_two_l010_p224_c24"
)

RUN_ARGS=(
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode base_only --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 192 --max-completion-len 16 --max-total-len 208 --anchor-metric none"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 192 --max-completion-len 16 --max-total-len 208 --lambda-novelty 0.05"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 192 --max-completion-len 16 --max-total-len 208 --lambda-novelty 0.10"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 192 --max-completion-len 16 --max-total-len 208 --lambda-novelty 0.15"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 192 --max-completion-len 24 --max-total-len 216 --lambda-novelty 0.10"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 192 --max-completion-len 16 --max-total-len 208 --anchor-metric output_kl --anchor-weight 0.05"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 192 --max-completion-len 16 --max-total-len 208 --anchor-metric param_l2 --anchor-weight 0.05"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode multi_round --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 192 --max-completion-len 16 --max-total-len 208 --num-rounds 4"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name deepscaler_preview --dataset-config-name default --answer-constraint-mode expression --pipeline-mode base_only --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 224 --max-completion-len 24 --max-total-len 248 --anchor-metric none"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name deepscaler_preview --dataset-config-name default --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 224 --max-completion-len 24 --max-total-len 248 --lambda-novelty 0.10"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name deepscaler_preview --dataset-config-name default --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 224 --max-completion-len 32 --max-total-len 256 --lambda-novelty 0.10"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name deepscaler_preview --dataset-config-name default --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 224 --max-completion-len 24 --max-total-len 248 --lambda-novelty 0.15"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name deepscaler_preview --dataset-config-name default --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 224 --max-completion-len 24 --max-total-len 248 --anchor-metric output_kl --anchor-weight 0.05"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name deepscaler_preview --dataset-config-name default --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 224 --max-completion-len 24 --max-total-len 248 --anchor-metric param_l2 --anchor-weight 0.05"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name deepscaler_preview --dataset-config-name default --answer-constraint-mode expression --pipeline-mode multi_round --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 224 --max-completion-len 24 --max-total-len 248 --num-rounds 4"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name dapo_math_17k --dataset-config-name all --answer-constraint-mode expression --pipeline-mode base_only --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 288 --max-completion-len 24 --max-total-len 312 --anchor-metric none"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name dapo_math_17k --dataset-config-name all --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 288 --max-completion-len 24 --max-total-len 312 --lambda-novelty 0.10"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name dapo_math_17k --dataset-config-name all --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 320 --max-completion-len 32 --max-total-len 352 --lambda-novelty 0.10"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name dapo_math_17k --dataset-config-name all --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 288 --max-completion-len 24 --max-total-len 312 --lambda-novelty 0.15"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name dapo_math_17k --dataset-config-name all --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 288 --max-completion-len 24 --max-total-len 312 --anchor-metric output_kl --anchor-weight 0.05"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name dapo_math_17k --dataset-config-name all --answer-constraint-mode expression --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 288 --max-completion-len 24 --max-total-len 312 --anchor-metric param_l2 --anchor-weight 0.05"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name dapo_math_17k --dataset-config-name all --answer-constraint-mode expression --pipeline-mode multi_round --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 288 --max-completion-len 24 --max-total-len 312 --num-rounds 4"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 160 --max-completion-len 16 --max-total-len 176 --lambda-novelty 0.10"
  "--model-name Qwen/Qwen2.5-Math-1.5B-Instruct --dataset-name gsm8k --answer-constraint-mode integer --pipeline-mode two_model --trajectory-batch-size 1 --eval-batch-size 1 --max-prompt-len 224 --max-completion-len 24 --max-total-len 248 --lambda-novelty 0.10"
)

slot_count=$(( ${#WORKERS[@]} * LOCAL_CHIPS ))

declare -a SLOT_LOCAL_SCRIPTS=()
declare -a SLOT_REMOTE_SCRIPTS=()
declare -a SLOT_SESSIONS=()
declare -a SLOT_WORKERS=()
declare -a SLOT_CREATED=()

build_slot_script() {
  local worker="$1"
  local chip="$2"
  local session="$3"
  local local_script="$4"

  rm -f "$local_script"
  cat >"$local_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi
. "$VENV_DIR/bin/activate"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_ENTITY="$WANDB_ENTITY"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY="$WANDB_API_KEY"
fi
if [ -n "${WANDB_MODE:-}" ]; then
  export WANDB_MODE="$WANDB_MODE"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HF_TOKEN"
fi
if [ -n "${HF_ACCESS_TOKEN:-}" ]; then
  export HF_ACCESS_TOKEN="$HF_ACCESS_TOKEN"
fi
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export EXPDIS_TPU_SLICE_LAUNCHED=1
export EXPDIS_REQUIRE_TPU=1
export TPU_SKIP_MDS_QUERY=1
export EXPDIS_WANDB_MASTER_ONLY=0
export TPU_VISIBLE_CHIPS="$chip"
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_WORKER_ID="$worker"
export TPU_WORKER_HOSTNAMES="\$(python - <<'PY'
import urllib.request
req = urllib.request.Request(
    'http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints',
    headers={'Metadata-Flavor': 'Google'},
)
try:
    with urllib.request.urlopen(req, timeout=1.0) as resp:
        raw = resp.read().decode('utf-8').strip()
except Exception:
    raw = ''
hosts = []
for item in raw.split(','):
    parts = [part.strip() for part in item.split(':') if part.strip()]
    if parts:
        hosts.append(parts[-1])
print(','.join(hosts))
PY
)"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/_slot_logs"
slot_log="$RUN_ROOT/_slot_logs/${session}.log"
echo "[slot] worker=$worker chip=$chip session=$session start=\$(date -Iseconds)" | tee -a "\$slot_log"
EOF
}

append_run() {
  local local_script="$1"
  local run_name="$2"
  local run_args="$3"
  local run_dir="$RUN_ROOT/$run_name"
  local common=""
  local arg
  for arg in "${COMMON_ARGS[@]}"; do
    common+=" $(printf '%q' "$arg")"
  done
  cat >>"$local_script" <<EOF
run_dir="$run_dir"
run_log="\$run_dir/console.log"
mkdir -p "\$run_dir"
echo "[run] start name=$run_name time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
if python -u run_expdis_torch.py$common $run_args \\
  --wandb-run-name "$run_name" \\
  --plots-dir "\$run_dir/plots" \\
  --explorer-traj-path "\$run_dir/explorer_trajectories.jsonl" \\
  2>&1 | tee -a "\$run_log"
then
  echo "[run] success name=$run_name time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
else
  status=\${PIPESTATUS[0]}
  echo "[run] failure name=$run_name code=\$status time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
  if [ "$RETRY_ON_FAILURE" = "1" ]; then
    echo "[run] retry name=$run_name after=${RETRY_SLEEP_SECONDS}s" | tee -a "\$slot_log" "\$run_log"
    sleep "$RETRY_SLEEP_SECONDS"
    if python -u run_expdis_torch.py$common $run_args \\
      --wandb-run-name "${run_name}_retry1" \\
      --plots-dir "\$run_dir/plots_retry1" \\
      --explorer-traj-path "\$run_dir/explorer_trajectories_retry1.jsonl" \\
      2>&1 | tee -a "\$run_log"
    then
      echo "[run] retry-success name=$run_name time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
    else
      retry_status=\${PIPESTATUS[0]}
      echo "[run] retry-failure name=$run_name code=\$retry_status time=\$(date -Iseconds)" | tee -a "\$slot_log" "\$run_log"
    fi
  fi
fi
EOF
}

for idx in "${!RUN_NAMES[@]}"; do
  slot=$(( idx % slot_count ))
  worker="${WORKERS[$((slot / LOCAL_CHIPS))]}"
  chip="$((slot % LOCAL_CHIPS))"
  session="${SESSION_PREFIX}_w${worker}_c${chip}"
  local_script="/tmp/${session}.sh"
  remote_script="/tmp/${session}.sh"

  if [ -z "${SLOT_CREATED[$slot]:-}" ]; then
    build_slot_script "$worker" "$chip" "$session" "$local_script"
    SLOT_LOCAL_SCRIPTS+=("$local_script")
    SLOT_REMOTE_SCRIPTS+=("$remote_script")
    SLOT_SESSIONS+=("$session")
    SLOT_WORKERS+=("$worker")
    SLOT_CREATED[$slot]=1
  fi

  append_run "$local_script" "${RUN_NAMES[$idx]}" "${RUN_ARGS[$idx]}"
done

for local_script in "${SLOT_LOCAL_SCRIPTS[@]}"; do
  echo 'echo "[slot] finished at $(date -Iseconds)" | tee -a "$slot_log"' >>"$local_script"
  chmod +x "$local_script"
done

for idx in "${!SLOT_LOCAL_SCRIPTS[@]}"; do
  local_script="${SLOT_LOCAL_SCRIPTS[$idx]}"
  remote_script="${SLOT_REMOTE_SCRIPTS[$idx]}"
  session="${SLOT_SESSIONS[$idx]}"
  worker="${SLOT_WORKERS[$idx]}"

  expdis_gcloud alpha compute tpus tpu-vm scp \
    "$local_script" \
    "${TPU_NAME}:${remote_script}" \
    --worker="$worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG"

  expdis_gcloud alpha compute tpus tpu-vm ssh \
    "$TPU_NAME" \
    --worker="$worker" \
    --zone="$ZONE" \
    "$TPU_TRANSPORT_FLAG" \
    --command="chmod +x '$remote_script' && (tmux has-session -t '$session' 2>/dev/null && tmux kill-session -t '$session' || true) && tmux new-session -d -s '$session' 'bash $remote_script'"
done

echo "Queued ${#RUN_NAMES[@]} TPU runs across ${#SLOT_SESSIONS[@]} tmux sessions."
echo "Queue tag: $QUEUE_TAG"
echo "Inspect workers with:"
echo "  gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all --batch-size=all --zone=$ZONE $TPU_TRANSPORT_FLAG --command='tmux ls || true'"
