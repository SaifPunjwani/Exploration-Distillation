#!/usr/bin/env bash
# Local read-only overnight status guard for the active JAX TPU campaign.
# This intentionally does not restart training. Recovery/reload actions are
# handled by the dedicated reload/resume/lambda-chain supervisors.
set -u

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST to the ssh alias of the training VM}"
REMOTE_RUNS_ROOT="${REMOTE_RUNS_ROOT:?set REMOTE_RUNS_ROOT to the absolute runs directory on the training VM}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LOG="${LOG:-runs_logs/overnight_status_guard.log}"
mkdir -p "$(dirname "$LOG")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

while true; do
  state="$(
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$REMOTE_HOST" "bash -lc '
      RUN=\$(cat ~/active_jax_run_name 2>/dev/null || true)
      echo active_run=\$RUN
      if [ -z \"\$RUN\" ]; then exit 0; fi
      ROOT=\"$REMOTE_RUNS_ROOT/\$RUN\"
      echo root=\$ROOT
      if pgrep -f \"python -m expdis_jax.pipeline\" >/dev/null; then echo pipeline=alive; else echo pipeline=dead; fi
      echo latest_train:
      grep -E \"^\\[train\\] step|^\\[ckpt\\]|external reload|accepted .*correct trajectories|actual_hf_final|EXPDIS_SKIP_FINAL_EVAL|Traceback|ERROR\" \"\$ROOT/launcher.log\" 2>/dev/null | tail -12 || true
      echo latest_ckpts:
      find \"\$ROOT\" -maxdepth 4 -type d -name \"step_*\" 2>/dev/null | sort | tail -8 || true
      echo trajectory_bytes:
      find \"\$ROOT\" -maxdepth 3 -type f \\( -name \"*trajector*.jsonl\" -o -name \"*rollout*.jsonl\" \\) -printf \"%s %p\\n\" 2>/dev/null | sort -n | tail -5 || true
      echo hf_artifacts:
      python3 - <<PY
from huggingface_hub import HfApi
repo = \"${EXPDIS_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}\"
prefix = \"\$RUN/\"
try:
    files = [f for f in HfApi().list_repo_files(repo_id=repo, repo_type=\"dataset\") if f.startswith(prefix)]
    print(\"hf_count=\" + str(len(files)))
    print(\"\\n\".join(files[-12:]))
except Exception as exc:
    print(\"hf_error=\" + str(exc))
PY
    '" 2>&1 || true
  )"
  log "$state"
  sleep "$POLL_SECONDS"
done
