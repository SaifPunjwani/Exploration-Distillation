#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GCS_PREFIX="${GCS_PREFIX:-}"
LOCAL_DIR="${LOCAL_DIR:-}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-120}"

if [ -z "$GCS_PREFIX" ] || [ -z "$LOCAL_DIR" ]; then
  echo "Usage: GCS_PREFIX=gs://bucket/path LOCAL_DIR=local/path [SYNC_INTERVAL_SECONDS=120] $0" >&2
  exit 1
fi

mkdir -p "$LOCAL_DIR"

while true; do
  gcloud storage rsync -r "$GCS_PREFIX" "$LOCAL_DIR" >/dev/null 2>&1 || true
  sleep "$SYNC_INTERVAL_SECONDS"
done
