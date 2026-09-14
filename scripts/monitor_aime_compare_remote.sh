#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh

PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"
ZONE="${ZONE:-europe-west4-b}"
TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
WORKER="${WORKER:-0}"
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-$(tmx_resolve_tpu_accelerator_type 2>/dev/null || true)}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
V6E_ZONE="${V6E_ZONE:-europe-west4-a}"
V6E_QUEUED_RESOURCE="${V6E_QUEUED_RESOURCE:?set V6E_QUEUED_RESOURCE to the queued-resource name of the v6e slice}"

BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_PASS_K="${BENCHMARK_ROLLOUT_PASS_K:-1,4,8,16,32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-16384}"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-256}"
USE_WANDB="${USE_WANDB:-auto}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_AUTO_RELAUNCHES="${MAX_AUTO_RELAUNCHES:-3}"

BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
BASE_RUN_NAME="${BASE_RUN_NAME:-ew4b_qwen25_base_raw_aime24_avg32}"
ACTUAL_ARTIFACT="${ACTUAL_ARTIFACT:?set ACTUAL_ARTIFACT to the W&B artifact reference (<entity>/<project>/<artifact>:<alias>) of the promoted Actual model}"
ACTUAL_MODEL_DIR="${ACTUAL_MODEL_DIR:-artifacts/main_recipe_20260324_20_two_ws_distill_dl5e6-actual-model}"
ACTUAL_RUN_NAME="${ACTUAL_RUN_NAME:-ew4b_qwen25_actual_promoted_aime24_avg32}"

LOCAL_MONITOR_ROOT="${LOCAL_MONITOR_ROOT:-runs/aime_compare_monitor_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOCAL_MONITOR_ROOT"
LOG_PATH="${LOCAL_MONITOR_ROOT}/monitor.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_PATH"
}

ssh_remote() {
  CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --worker "$WORKER" \
    --tunnel-through-iap \
    --command "$1"
}

scp_remote_file() {
  local remote_path="$1"
  local local_path="$2"
  mkdir -p "$(dirname "$local_path")"
  CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
  gcloud alpha compute tpus tpu-vm scp \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --worker "$WORKER" \
    --tunnel-through-iap \
    "${TPU_NAME}:${remote_path}" \
    "$local_path" >/dev/null 2>&1 || true
}

remote_file_exists() {
  local remote_path="$1"
  ssh_remote "test -f '$remote_path' && echo YES || echo NO" 2>/dev/null | tail -n 1
}

remote_run_alive() {
  local run_name="$1"
  ssh_remote "if tmux has-session -t '$run_name' 2>/dev/null; then echo YES; elif pgrep -af '$run_name' | grep -v 'pgrep -af' >/dev/null 2>&1; then echo YES; else echo NO; fi" 2>/dev/null | tail -n 1
}

remote_exit_code() {
  local run_name="$1"
  ssh_remote "cat '$REMOTE_ROOT/runs/$run_name/exit_code.txt' 2>/dev/null || true" 2>/dev/null | tail -n 1
}

local_attempt_count() {
  local run_name="$1"
  local path="$LOCAL_MONITOR_ROOT/${run_name}_attempts.txt"
  if [ -f "$path" ]; then
    cat "$path"
  else
    echo 0
  fi
}

increment_local_attempt_count() {
  local run_name="$1"
  local path="$LOCAL_MONITOR_ROOT/${run_name}_attempts.txt"
  local current
  current="$(local_attempt_count "$run_name")"
  current=$((current + 1))
  printf '%s\n' "$current" > "$path"
  echo "$current"
}

reset_local_attempt_count() {
  local run_name="$1"
  printf '0\n' > "$LOCAL_MONITOR_ROOT/${run_name}_attempts.txt"
}

copy_run_outputs() {
  local run_name="$1"
  local local_dir="$LOCAL_MONITOR_ROOT/$run_name"
  scp_remote_file "$REMOTE_ROOT/runs/$run_name/console.log" "$local_dir/console.log"
  scp_remote_file "$REMOTE_ROOT/runs/$run_name/exit_code.txt" "$local_dir/exit_code.txt"
  scp_remote_file "$REMOTE_ROOT/runs/$run_name/analysis/benchmark_summary.json" "$local_dir/benchmark_summary.json"
  scp_remote_file "$REMOTE_ROOT/runs/$run_name/analysis/aime_rollouts.json" "$local_dir/aime_rollouts.json"
  scp_remote_file "$REMOTE_ROOT/runs/$run_name/analysis/benchmark_rollouts_base.json" "$local_dir/benchmark_rollouts_base.json"
}

ensure_actual_artifact() {
  local shard1="$REMOTE_ROOT/$ACTUAL_MODEL_DIR/model-00001-of-00002.safetensors"
  local shard2="$REMOTE_ROOT/$ACTUAL_MODEL_DIR/model-00002-of-00002.safetensors"
  if [ "$(remote_file_exists "$shard1")" = "YES" ] && [ "$(remote_file_exists "$shard2")" = "YES" ]; then
    log "Actual artifact ready at $ACTUAL_MODEL_DIR"
    return 0
  fi

  log "Actual artifact missing; downloading $ACTUAL_ARTIFACT"
  local auth_prefix=""
  if [ -n "${WANDB_API_KEY:-}" ]; then
    auth_prefix="export WANDB_API_KEY='${WANDB_API_KEY}'; "
  fi
  ssh_remote "cd '$REMOTE_ROOT' || exit 1; mkdir -p '$(dirname "$ACTUAL_MODEL_DIR")'; ${auth_prefix}.venv_tpu_slice/bin/python -c 'import wandb; api = wandb.Api(); artifact = api.artifact(\"$ACTUAL_ARTIFACT\"); print(artifact.download(root=\"$ACTUAL_MODEL_DIR\"))'" >/dev/null
  if [ "$(remote_file_exists "$shard1")" != "YES" ] || [ "$(remote_file_exists "$shard2")" != "YES" ]; then
    log "Actual artifact download did not complete cleanly"
    return 1
  fi
  log "Actual artifact download complete"
}

launch_eval_run() {
  local run_name="$1"
  local model_name="$2"
  local remote_script="$REMOTE_ROOT/.tmx_launch_${run_name}.sh"

  local dataset_cfg_line=""
  if [ -n "$BENCHMARK_DATASET_CONFIG_NAME" ]; then
    dataset_cfg_line="export BENCHMARK_DATASET_CONFIG_NAME='$BENCHMARK_DATASET_CONFIG_NAME'"
  fi

  ssh_remote "cat > '$remote_script' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
mkdir -p 'runs/$run_name/analysis' 'runs/$run_name/plots'
: > 'runs/$run_name/console.log'
export PYTHONUNBUFFERED=1
export PYTHON_BIN='.venv_tpu_slice/bin/python'
export DEVICE='tpu'
export PJRT_DEVICE='TPU'
export TPU_ACCELERATOR_TYPE='$TPU_ACCELERATOR_TYPE'
export TPU_VISIBLE_CHIPS='0'
export TPU_PROCESS_BOUNDS='1,1,1'
export TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1'
export TPU_CHIPS_PER_HOST_BOUNDS='1,1,1'
export TPU_HOST_BOUNDS='1,1,1'
export TPU_WORKER_ID='0'
export TPU_WORKER_HOSTNAMES='localhost'
export MODEL_NAME='$model_name'
export RUN_NAME='$run_name'
export BENCHMARK_DATASET_NAME='$BENCHMARK_DATASET_NAME'
$dataset_cfg_line
export MAX_BENCHMARK='$MAX_BENCHMARK'
export BENCHMARK_NUM_ROLLOUTS='$BENCHMARK_NUM_ROLLOUTS'
export BENCHMARK_ROLLOUT_TEMPERATURE='$BENCHMARK_ROLLOUT_TEMPERATURE'
export BENCHMARK_ROLLOUT_TOP_P='$BENCHMARK_ROLLOUT_TOP_P'
export BENCHMARK_ROLLOUT_PASS_K='$BENCHMARK_ROLLOUT_PASS_K'
export BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN='$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN'
export BENCHMARK_EVAL_BATCH_SIZE='$BENCHMARK_EVAL_BATCH_SIZE'
export MAX_PROMPT_LEN='$MAX_PROMPT_LEN'
export MAX_COMPLETION_LEN='$MAX_COMPLETION_LEN'
export USE_WANDB='$USE_WANDB'
bash scripts/run_aime_rollout_eval.sh > 'runs/$run_name/console.log' 2>&1
ec=\$?
echo \$ec > 'runs/$run_name/exit_code.txt'
exit \$ec
EOF
chmod +x '$remote_script'
if tmux has-session -t '$run_name' 2>/dev/null; then
  echo ALREADY_RUNNING
  exit 0
fi
tmux new-session -d -s '$run_name' '$remote_script'
echo LAUNCHED" >/dev/null
  local attempts
  attempts="$(increment_local_attempt_count "$run_name")"
  log "Launched $run_name (attempt $attempts/$MAX_AUTO_RELAUNCHES)"
}

tail_remote_log() {
  local run_name="$1"
  ssh_remote "tail -n 5 '$REMOTE_ROOT/runs/$run_name/console.log' 2>/dev/null || true" 2>/dev/null | tee -a "$LOG_PATH" >/dev/null
}

v6e_state() {
  CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
  gcloud alpha compute tpus queued-resources list \
    --project "$PROJECT" \
    --zone "$V6E_ZONE" \
    --format='value(state.state,name)' 2>/dev/null | awk -v want="$V6E_QUEUED_RESOURCE" '$2==want{print $1}'
}

log "Monitor root: $LOCAL_MONITOR_ROOT"
log "Target TPU: $TPU_NAME ($ZONE), worker=$WORKER"
log "Benchmark: $BENCHMARK_DATASET_NAME rollout_n=$BENCHMARK_NUM_ROLLOUTS batch=$BENCHMARK_EVAL_BATCH_SIZE"

while true; do
  local_v6e_state="$(v6e_state)"
  log "v6e queued resource state: ${local_v6e_state:-unknown}"

  base_summary_remote="$REMOTE_ROOT/runs/$BASE_RUN_NAME/analysis/benchmark_summary.json"
  actual_summary_remote="$REMOTE_ROOT/runs/$ACTUAL_RUN_NAME/analysis/benchmark_summary.json"

  base_done="$(remote_file_exists "$base_summary_remote")"
  actual_done="$(remote_file_exists "$actual_summary_remote")"

  if [ "$base_done" = "YES" ]; then
    log "Base benchmark summary present"
    reset_local_attempt_count "$BASE_RUN_NAME"
    copy_run_outputs "$BASE_RUN_NAME"
  else
    if [ "$(remote_run_alive "$BASE_RUN_NAME")" = "YES" ]; then
      log "Base eval still running"
      tail_remote_log "$BASE_RUN_NAME"
    else
      code="$(remote_exit_code "$BASE_RUN_NAME")"
      attempts="$(local_attempt_count "$BASE_RUN_NAME")"
      if [ "$attempts" -ge "$MAX_AUTO_RELAUNCHES" ]; then
        log "Base eval reached auto-relaunch limit ($attempts/$MAX_AUTO_RELAUNCHES); waiting for manual intervention"
      else
        if [ -n "$code" ]; then
          log "Base eval exited with code $code; relaunching"
        else
          log "Base eval not running; launching"
        fi
        launch_eval_run "$BASE_RUN_NAME" "$BASE_MODEL_NAME"
      fi
    fi
  fi

  if [ "$base_done" = "YES" ]; then
    if ensure_actual_artifact; then
      if [ "$actual_done" = "YES" ]; then
        log "Actual benchmark summary present"
        reset_local_attempt_count "$ACTUAL_RUN_NAME"
        copy_run_outputs "$ACTUAL_RUN_NAME"
      else
        if [ "$(remote_run_alive "$ACTUAL_RUN_NAME")" = "YES" ]; then
          log "Actual eval still running"
          tail_remote_log "$ACTUAL_RUN_NAME"
        else
          code="$(remote_exit_code "$ACTUAL_RUN_NAME")"
          attempts="$(local_attempt_count "$ACTUAL_RUN_NAME")"
          if [ "$attempts" -ge "$MAX_AUTO_RELAUNCHES" ]; then
            log "Actual eval reached auto-relaunch limit ($attempts/$MAX_AUTO_RELAUNCHES); waiting for manual intervention"
          else
            if [ -n "$code" ]; then
              log "Actual eval exited with code $code; relaunching"
            else
              log "Actual eval not running; launching"
            fi
            launch_eval_run "$ACTUAL_RUN_NAME" "$REMOTE_ROOT/$ACTUAL_MODEL_DIR"
          fi
        fi
      fi
    fi
  fi

  if [ "$base_done" = "YES" ] && [ "$actual_done" = "YES" ]; then
    log "Both AIME comparison runs are complete"
    break
  fi

  sleep "$POLL_SECONDS"
done
