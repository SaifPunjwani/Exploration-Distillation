#!/usr/bin/env bash
# Keep the local vLLM reload watcher alive for long-running spot TPU jobs.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG="${LOG:-$REPO_ROOT/runs_logs/external_vllm_reload_supervisor.log}"
mkdir -p "$(dirname "$LOG")"

while true; do
  printf '[%s] starting external reload watcher\n' "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
  (
    cd "$REPO_ROOT"
    exec /bin/bash expdis_jax/scripts/external_vllm_reload_watcher.sh
  )
  rc=$?
  printf '[%s] watcher exited rc=%s; restarting in 5s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$rc" | tee -a "$LOG"
  sleep 5
done
