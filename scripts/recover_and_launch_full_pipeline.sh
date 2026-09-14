#!/usr/bin/env bash
set -euo pipefail

# === Recovery + Full Pipeline Launch ===
# Polls for v6e-64 TPU to come up, bootstraps env, downloads Explorer
# checkpoint from GCS, syncs code, and launches the full pipeline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-a}"
PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GCS_CHECKPOINT="gs://two-model-exploration-checkpoints/v6e_flash_saturation_8k/checkpoint-1000/checkpoint-1000"
LOCAL_CHECKPOINT_DIR="runs/explorer_ckpt_1000_from_gcs"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# --- Ensure queued resource exists ---
ensure_qr() {
  local qr_state node_state
  qr_state="$(gcloud alpha compute tpus queued-resources describe "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" --format='value(state.state)' 2>/dev/null || true)"
  node_state="$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" --format='value(state)' 2>/dev/null || true)"
  if [ -z "$qr_state" ] && [ -z "$node_state" ]; then
    log "Creating queued resource $TPU_NAME..."
    gcloud alpha compute tpus queued-resources create "$TPU_NAME" \
      --project="$PROJECT" --zone="$ZONE" \
      --accelerator-type=v6e-64 --runtime-version=v2-alpha-tpuv6e \
      --node-id="$TPU_NAME" --provisioning-model=SPOT \
      --internal-ips --network=default --subnetwork=default
  fi
}

# --- Poll until READY + HEALTHY ---
log "Waiting for $TPU_NAME to become READY..."
ensure_qr
while true; do
  state="$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" --format='value(state)' 2>/dev/null || true)"
  health="$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" --format='value(health)' 2>/dev/null || true)"
  log "state=${state:-NONE} health=${health:-NONE}"
  if [ "$state" = "READY" ] && [ "$health" = "HEALTHY" ]; then
    log "TPU is READY+HEALTHY!"
    break
  fi
  ensure_qr
  sleep "$POLL_SECONDS"
done

# --- Bootstrap: sync code + set up env on worker 0 ---
log "Syncing code to $TPU_NAME worker 0..."
gcloud alpha compute tpus tpu-vm scp --recurse \
  "$ROOT_DIR" "${TPU_NAME}:${REMOTE_ROOT}" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --tunnel-through-iap \
  --compress 2>&1 | tail -5

log "Setting up environment on worker 0..."
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --tunnel-through-iap \
  --command="bash -c '
    cd $REMOTE_ROOT

    # Create venv if needed
    if [ ! -d .venv_tpu_slice ]; then
      python3 -m venv .venv_tpu_slice
    fi
    source .venv_tpu_slice/bin/activate

    # Install deps
    pip install -q -e . 2>&1 | tail -3

    # Download Explorer checkpoint from GCS
    echo \"Downloading Explorer checkpoint from GCS...\"
    mkdir -p ${LOCAL_CHECKPOINT_DIR}
    gsutil -m cp -r ${GCS_CHECKPOINT}/* ${LOCAL_CHECKPOINT_DIR}/ 2>&1 | tail -5
    ls -la ${LOCAL_CHECKPOINT_DIR}/

    echo \"Bootstrap complete.\"
  '" 2>&1

# --- Launch full pipeline ---
log "Launching full pipeline..."
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --tunnel-through-iap \
  --command="bash -c '
    cd $REMOTE_ROOT
    source .venv_tpu_slice/bin/activate
    nohup bash scripts/launch_v6e_full_pipeline.sh > /tmp/full_pipeline_launch.log 2>&1 &
    echo \"Pipeline launched with PID \$!\"
    sleep 3
    head -20 /tmp/full_pipeline_launch.log 2>/dev/null || true
  '" 2>&1

log "Full pipeline launched on $TPU_NAME. Monitor with:"
log "  gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --tunnel-through-iap --command='tail -50 $REMOTE_ROOT/runs/v6e_full_pipeline_*/console.log'"
