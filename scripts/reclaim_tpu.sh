#!/usr/bin/env bash
set -euo pipefail

# Reclaim TPU device access by freeing /dev/accel* holders.
# Usage:
#   sudo -E bash scripts/reclaim_tpu.sh
# Optional env vars:
#   KILL_HOLDERS=0   # don't send SIGTERM (default: 1)
#   KILL_FORCE=1     # send SIGKILL if still held (default: 0)
#   RESTART_RUNTIME=1  # restart TPU runtime services (default: 0)

KILL_HOLDERS="${KILL_HOLDERS:-1}"
KILL_FORCE="${KILL_FORCE:-0}"
RESTART_RUNTIME="${RESTART_RUNTIME:-0}"

shopt -s nullglob
ACCEL_DEVS=(/dev/accel*)
if [ ${#ACCEL_DEVS[@]} -eq 0 ]; then
  echo "No /dev/accel* devices found. Are you on a TPU VM?"
  exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "Not running as root. Re-run with: sudo -E bash scripts/reclaim_tpu.sh"
  exit 1
fi

echo "Checking TPU device holders..."
PIDS="$(fuser /dev/accel* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)"

if [ -z "$PIDS" ]; then
  echo "No processes currently hold /dev/accel*."
else
  echo "Processes holding TPU devices:"
  ps -fp $PIDS || true

  if [ "$KILL_HOLDERS" = "1" ]; then
    echo "Sending SIGTERM to holders..."
    kill $PIDS || true
    for _ in $(seq 1 10); do
      sleep 1
      STILL="$(fuser /dev/accel* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)"
      if [ -z "$STILL" ]; then
        break
      fi
    done

    STILL="$(fuser /dev/accel* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)"
    if [ -n "$STILL" ]; then
      echo "Still held by: $STILL"
      if [ "$KILL_FORCE" = "1" ]; then
        echo "Sending SIGKILL..."
        kill -9 $STILL || true
      else
        echo "Set KILL_FORCE=1 to send SIGKILL."
      fi
    fi
  else
    echo "KILL_HOLDERS=0 set; not sending signals."
  fi
fi

if [ "$RESTART_RUNTIME" = "1" ]; then
  echo "Restarting TPU runtime services..."
  systemctl restart tpu-runtime.service || true
  systemctl restart tpu-worker-configuration.service || true
fi

echo "Re-checking /dev/accel* access..."
python3 - <<'PY'
import os
ok = False
for i in range(4):
    p = f"/dev/accel{i}"
    try:
        fd = os.open(p, os.O_RDWR)
        os.close(fd)
        print(f"{p} OK")
        ok = True
    except Exception as e:
        print(f"{p} FAIL {e}")
if not ok:
    raise SystemExit("TPU still not openable. Check for other holders or set RESTART_RUNTIME=1.")
PY
