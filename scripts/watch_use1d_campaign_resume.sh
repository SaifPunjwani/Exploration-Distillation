#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SESSION_NAME="${SESSION_NAME:-use1d_full_campaign_resume}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-use1d_six_model_campaign_20260331_211041}"
LOG_PATH="${LOG_PATH:-runs/${CAMPAIGN_TAG}_resume.log}"
POLL_SECONDS="${POLL_SECONDS:-120}"

mkdir -p "$(dirname "$LOG_PATH")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_PATH"
}

has_dispatch_progress() {
  if [ ! -f "$LOG_PATH" ]; then
    return 1
  fi
  grep -qE 'launching raw AIME24 herd eval|launching training queue: worker=[1-8]' "$LOG_PATH"
}

while true; do
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    sleep "$POLL_SECONDS"
    continue
  fi

  if has_dispatch_progress; then
    log "dispatch progress already recorded; watchdog exiting"
    exit 0
  fi

  log "resume supervisor missing; relaunching ${SESSION_NAME}"
  tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT_DIR' && bash scripts/resume_use1d_full_campaign.sh >> '$LOG_PATH' 2>&1"
  sleep "$POLL_SECONDS"
done
