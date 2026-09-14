#!/usr/bin/env bash
# Local control-plane helper:
# wait for the active clean run to save step_000050 and acknowledge the vLLM
# reload, then relaunch the same run from that checkpoint with the latest code.
set -u

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST to the ssh host alias of the training VM}"
REMOTE_REPO="${REMOTE_REPO:?set REMOTE_REPO to the absolute repo path on every TPU worker}"
REMOTE_RUNS_ROOT="${REMOTE_RUNS_ROOT:?set REMOTE_RUNS_ROOT to the absolute runs directory on the training VM}"
TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-b}"
WORKERS="${WORKERS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG="${LOG:-runs_logs/resume_fast_after_step50.log}"
mkdir -p "$(dirname "$LOG")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

remote() {
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$REMOTE_HOST" "$@"
}

RUN_NAME="$(remote 'cat ~/active_jax_run_name 2>/dev/null || true' || true)"
if [ -z "$RUN_NAME" ]; then
  log "no active run found; exiting"
  exit 1
fi
ROOT="$REMOTE_RUNS_ROOT/$RUN_NAME"
CKPT="$ROOT/explorer/step_000050"
log "watching run=$RUN_NAME checkpoint=$CKPT"

while true; do
  state="$(
    remote "bash -lc '
      ROOT=\"$ROOT\"
      CKPT=\"$CKPT\"
      if [ -d \"\$CKPT\" ]; then echo ckpt=yes; else echo ckpt=no; fi
      if grep -q \"external reload acknowledged: .*step_000050\" \"\$ROOT/launcher.log\" 2>/dev/null; then echo reload=yes; else echo reload=no; fi
      grep -E \"^\\[train\\] step|^\\[ckpt\\]|external reload acknowledged|ERROR|Traceback\" \"\$ROOT/launcher.log\" 2>/dev/null | tail -5
    '" 2>&1 || true
  )"
  log "$state"
  if printf '%s\n' "$state" | grep -q 'ckpt=yes' && printf '%s\n' "$state" | grep -q 'reload=yes'; then
    break
  fi
  sleep "$POLL_SECONDS"
done

log "step 50 checkpoint and reload are complete; relaunching same run from checkpoint with fast path"
remote "bash -lc '
  set -euo pipefail
  RUN_NAME=\"$RUN_NAME\"
  ROOT=\"$ROOT\"
  CKPT=\"$CKPT\"
  VLLM_URLS=\"${VLLM_URLS:?set VLLM_URLS to the comma-separated vLLM server URLs}\"
  WANDB_ID=\$(grep -oE \"https://wandb.ai/[^ ]+/runs/[^ ]+\" \"\$ROOT/launcher.log\" 2>/dev/null | sed -E \"s#.*/runs/##\" | grep -v \"^jaxpipe-\" | tail -1 || true)
  pkill -9 -f \"[p]ython -m tmx_jax.pipeline\" 2>/dev/null || true
  sleep 5
  cd \"$REMOTE_REPO\"
  source ~/miniforge3/bin/activate tmx_jax
  export RUN_NAME=\"\$RUN_NAME\"
  export RUNS_ROOT=\"$REMOTE_RUNS_ROOT\"
  export INIT_CHECKPOINT=\"\$CKPT\"
  export TMX_RESTORE_OPT_STATE=0
  export TMX_SAVE_OPT_STATE=0
  export TMX_VLLM_SERVER_URLS=\"\$VLLM_URLS\"
  export VLLM_SERVER_URLS=\"\$VLLM_URLS\"
  export TMX_ARTIFACT_BACKEND=\"\${TMX_ARTIFACT_BACKEND:-hf}\"
  export TMX_ALLOW_GCS_ARTIFACTS=\"\${TMX_ALLOW_GCS_ARTIFACTS:-0}\"
  export TMX_HF_CHECKPOINT_REPO=\"\${TMX_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}\"
  export TMX_HF_REPO_TYPE=\"\${TMX_HF_REPO_TYPE:-dataset}\"
  export TMX_GCS_CHECKPOINT_BASE=\"\${TMX_GCS_CHECKPOINT_BASE:-}\"
  export TMX_WANDB_SINGLE_RUN=1
  export TMX_WANDB_COMPACT=1
	  export TMX_VLLM_RETURN_LOGPROBS=0
	  export TMX_TRACK_EXTERNAL_OLD_GAP=0
	  export TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS=50
	  export TMX_TRAIN_VLLM_RELOAD_ON_RESUME=1
	  export TMX_TRAIN_VLLM_RELOAD_MODE=external
	  export TMX_EXTERNAL_VLLM_RELOAD_WAIT=1
	  export TMX_EXTERNAL_VLLM_RELOAD_REQUIRED=1
	  export TMX_VLLM_RELOAD_SERVER_URLS=\"\$VLLM_URLS\"
	  export TMX_VLLM_RELOAD_TPU_NAME=\"$TPU_NAME\"
	  export TMX_VLLM_RELOAD_ZONE=\"$ZONE\"
	  export TMX_VLLM_RELOAD_WORKERS=\"$WORKERS\"
  export GRPO_MAX_STEPS=\"\${PROOF_GRPO_MAX_STEPS:-100}\"
  export ACTUAL_ONLINE_GRPO_MAX_STEPS=\"\${PROOF_ACTUAL_ONLINE_GRPO_MAX_STEPS:-96}\"
  export MIN_ACCEPTED_TRAJECTORIES=\"\${PROOF_MIN_ACCEPTED_TRAJECTORIES:-512}\"
  export MAX_ACCEPTED_TRAJECTORIES=\"\${PROOF_MAX_ACCEPTED_TRAJECTORIES:-1024}\"
  export DISTILL_EPOCHS=\"\${PROOF_DISTILL_EPOCHS:-1}\"
  export MAX_DISTILL_EXAMPLES=\"\${PROOF_MAX_DISTILL_EXAMPLES:-1024}\"
  export BENCHMARK_NUM_ROLLOUTS=\"\${PROOF_BENCHMARK_NUM_ROLLOUTS:-32}\"
  export BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN=\"\${PROOF_BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-32768}\"
  export TMX_SKIP_FINAL_EVAL=\"\${PROOF_SKIP_FINAL_EVAL:-1}\"
  if [ -n \"\$WANDB_ID\" ]; then
    export TMX_WANDB_RUN_ID=\"\$WANDB_ID\"
    export TMX_WANDB_EXPLORER_RUN_ID=\"\$WANDB_ID\"
    export TMX_WANDB_DISTILL_RUN_ID=\"\$WANDB_ID\"
    export TMX_WANDB_ACTUAL_RUN_ID=\"\$WANDB_ID\"
    export TMX_WANDB_ACTUAL_GRPO_RUN_ID=\"\$WANDB_ID\"
  fi
  tmux kill-session -t \"\${RUN_NAME}_resume_fast\" 2>/dev/null || true
  tmux new-session -d -s \"\${RUN_NAME}_resume_fast\" \"bash tmx_jax/scripts/launch_jax_pipeline.sh >> '\$ROOT'/launcher.log 2>&1\"
  tmux ls | grep \"\${RUN_NAME}_resume_fast\"
'"
log "relaunch command dispatched"
