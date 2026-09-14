#!/usr/bin/env bash
# Local control-plane helper:
# after the active lambda=0 run saves an Actual HF bundle, reset vLLM to the
# base model and launch a matching lambda=0.5 run on the same TPU slice.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST to the ssh host alias of the training VM}"
REMOTE_REPO="${REMOTE_REPO:?set REMOTE_REPO to the absolute repo path on every TPU worker}"
REMOTE_RUNS_ROOT="${REMOTE_RUNS_ROOT:?set REMOTE_RUNS_ROOT to the absolute runs directory on the training VM}"
TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-b}"
WORKERS="${WORKERS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LOG="${LOG:-runs_logs/launch_lambda05_after_lambda0.log}"
mkdir -p "$(dirname "$LOG")"

VLLM_URLS="${VLLM_URLS:?set VLLM_URLS to the comma-separated vLLM server URLs}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

remote() {
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$REMOTE_HOST" "$@"
}

LAMBDA0_RUN="${LAMBDA0_RUN:-$(remote 'cat ~/active_jax_run_name 2>/dev/null || true' || true)}"
if [ -z "$LAMBDA0_RUN" ]; then
  log "no active lambda=0 run found"
  exit 1
fi

log "watching lambda0_run=$LAMBDA0_RUN for Actual HF bundle"
while true; do
  state="$(
    remote "bash -lc '
      ROOT=\"$REMOTE_RUNS_ROOT/$LAMBDA0_RUN\"
      if [ -d \"\$ROOT/actual/actual_hf_final\" ]; then echo local_actual=yes; else echo local_actual=no; fi
      python3 - <<PY
from huggingface_hub import HfApi
repo = \"${TMX_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}\"
path = \"$LAMBDA0_RUN/actual_hf_final/config.json\"
try:
    print(\"hf_actual=yes\" if path in set(HfApi().list_repo_files(repo_id=repo, repo_type=\"dataset\")) else \"hf_actual=no\")
except Exception:
    print(\"hf_actual=unknown\")
PY
      grep -E \"actual_hf_final|TMX_SKIP_FINAL_EVAL|pipeline/actual_hf_final_saved|Traceback|ERROR\" \"\$ROOT/launcher.log\" 2>/dev/null | tail -8
    '" 2>&1 || true
  )"
  log "$state"
  if printf '%s\n' "$state" | grep -q 'hf_actual=yes'; then
    break
  fi
  sleep "$POLL_SECONDS"
done

log "lambda=0 Actual bundle exists; resetting vLLM workers to base Qwen"
TPU_NAME="$TPU_NAME" \
ZONE="$ZONE" \
WORKERS="$WORKERS" \
MODEL="Qwen/Qwen3-1.7B" \
MAX_MODEL_LEN=10240 \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}" \
bash "$(dirname "${BASH_SOURCE[0]}")/bootstrap_vllm_slice.sh" 2>&1 | tee -a "$LOG"

log "checking vLLM health from worker 0"
remote "bash -lc '
  set -e
  IFS=, read -ra URLS <<< \"$VLLM_URLS\"
  for u in \"\${URLS[@]}\"; do
    h=\"\${u%/v1}/health\"
    for i in \$(seq 1 60); do
      if curl -fsS --connect-timeout 2 \"\$h\" >/dev/null 2>&1; then
        echo \"healthy \$h\"
        break
      fi
      if [ \"\$i\" = 60 ]; then
        echo \"ERROR unhealthy \$h\" >&2
        exit 1
      fi
      sleep 5
    done
  done
'"

RUN_NAME="${LAMBDA05_RUN_NAME:-jax_contract_lambda05_proof_$(date +%Y%m%d_%H%M%S)}"
log "launching lambda=0.5 run=$RUN_NAME"
remote "bash -lc '
  set -euo pipefail
  RUN_NAME=\"$RUN_NAME\"
  ROOT=\"$REMOTE_RUNS_ROOT/\$RUN_NAME\"
  mkdir -p \"\$ROOT\"
  echo \"\$RUN_NAME\" > ~/active_jax_run_name
  cd \"$REMOTE_REPO\"
  source ~/miniforge3/bin/activate tmx_jax
  export RUN_NAME=\"\$RUN_NAME\"
  export RUNS_ROOT=\"$REMOTE_RUNS_ROOT\"
  export TMX_VLLM_SERVER_URLS=\"$VLLM_URLS\"
  export VLLM_SERVER_URLS=\"$VLLM_URLS\"
  export TMX_ARTIFACT_BACKEND=\"\${TMX_ARTIFACT_BACKEND:-hf}\"
  export TMX_ALLOW_GCS_ARTIFACTS=\"\${TMX_ALLOW_GCS_ARTIFACTS:-0}\"
  export TMX_HF_CHECKPOINT_REPO=\"\${TMX_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}\"
  export TMX_HF_REPO_TYPE=\"\${TMX_HF_REPO_TYPE:-dataset}\"
  export TMX_GCS_CHECKPOINT_BASE=\"\${TMX_GCS_CHECKPOINT_BASE:-}\"
  export LAMBDA_NOVELTY=0.5
  export GRPO_MAX_STEPS=\"\${LAMBDA05_GRPO_MAX_STEPS:-200}\"
  export ACTUAL_ONLINE_GRPO_MAX_STEPS=\"\${LAMBDA05_ACTUAL_ONLINE_GRPO_MAX_STEPS:-128}\"
  export MIN_ACCEPTED_TRAJECTORIES=\"\${LAMBDA05_MIN_ACCEPTED_TRAJECTORIES:-512}\"
  export MAX_ACCEPTED_TRAJECTORIES=\"\${LAMBDA05_MAX_ACCEPTED_TRAJECTORIES:-1024}\"
  export DISTILL_EPOCHS=\"\${LAMBDA05_DISTILL_EPOCHS:-1}\"
  export MAX_DISTILL_EXAMPLES=\"\${LAMBDA05_MAX_DISTILL_EXAMPLES:-1024}\"
  export SAVE_EVERY_STEPS=50
  export TMX_SKIP_FINAL_EVAL=\"\${LAMBDA05_SKIP_FINAL_EVAL:-1}\"
  export TMX_WANDB_SINGLE_RUN=1
  export TMX_WANDB_COMPACT=1
	  export TMX_VLLM_RETURN_LOGPROBS=0
	  export TMX_TRACK_EXTERNAL_OLD_GAP=0
	  export TMX_SAVE_OPT_STATE=0
	  export TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS=50
	  export TMX_TRAIN_VLLM_RELOAD_ON_RESUME=1
	  export TMX_TRAIN_VLLM_RELOAD_MODE=external
	  export TMX_EXTERNAL_VLLM_RELOAD_WAIT=1
	  export TMX_EXTERNAL_VLLM_RELOAD_REQUIRED=1
	  export TMX_VLLM_RELOAD_SERVER_URLS=\"$VLLM_URLS\"
	  export TMX_VLLM_RELOAD_TPU_NAME=\"$TPU_NAME\"
	  export TMX_VLLM_RELOAD_ZONE=\"$ZONE\"
	  export TMX_VLLM_RELOAD_WORKERS=\"$WORKERS\"
  tmux kill-session -t \"\$RUN_NAME\" 2>/dev/null || true
  tmux new-session -d -s \"\$RUN_NAME\" \"bash tmx_jax/scripts/launch_jax_pipeline.sh > '\$ROOT'/launcher.log 2>&1\"
  tmux ls | grep \"\$RUN_NAME\"
'"

log "lambda=0.5 launch dispatched"
