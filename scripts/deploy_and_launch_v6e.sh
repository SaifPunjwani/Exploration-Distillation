#!/usr/bin/env bash
set -euo pipefail

# === Deploy code to v6e TPU and launch full pipeline ===
# Handles: code sync, env bootstrap, hf_xet fix, flash attention test, launch

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- Auto-detect TPU name and zone ---
TPU_NAME="${TPU_NAME:-}"
ZONE="${ZONE:-}"
PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"
# Repo checkout on the TPU VM. The staged tarball unpacks a top-level
# directory with the same basename, so keep them aligned.
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
TMX_VLLM_SERVER_URLS="${TMX_VLLM_SERVER_URLS:?set TMX_VLLM_SERVER_URLS to the comma-separated vLLM server URLs}"

if [ -z "$TPU_NAME" ] || [ -z "$ZONE" ]; then
  echo "Auto-detecting v6e TPU..."
  for z in europe-west4-a us-east1-d; do
    found="$(gcloud alpha compute tpus tpu-vm list --project="$PROJECT" --zone="$z" \
      --filter='acceleratorType~v6e AND state=READY' --format='value(name)' 2>/dev/null | head -1)"
    if [ -n "$found" ]; then
      TPU_NAME="$found"
      ZONE="$z"
      echo "Found: $TPU_NAME in $ZONE"
      break
    fi
  done
  if [ -z "$TPU_NAME" ]; then
    echo "ERROR: No READY v6e TPU found"
    exit 1
  fi
fi

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# --- Upload code to GCS (faster than SCP over IAP) ---
log "Uploading code to GCS..."
STAGING="gs://two-model-exploration-checkpoints/staging/tmx_code.tar.gz"
tar czf /tmp/tmx_code.tar.gz \
  --exclude='.git' --exclude='wandb' --exclude='runs' --exclude='.venv*' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.env' \
  --exclude='CLAUDE.md' --exclude='.claude' \
  -C "$(dirname "$ROOT_DIR")" "$(basename "$ROOT_DIR")"
gsutil -q cp /tmp/tmx_code.tar.gz "$STAGING"
log "Code uploaded to GCS"

# --- Bootstrap on worker 0 ---
log "Bootstrapping $TPU_NAME worker 0 in $ZONE..."
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --tunnel-through-iap \
  --command="bash -c '
    set -euo pipefail
    cd $(dirname "$REMOTE_ROOT")

    # Download code from GCS
    echo \"Downloading code from GCS...\"
    gsutil -q cp gs://two-model-exploration-checkpoints/staging/tmx_code.tar.gz /tmp/
    rm -rf $REMOTE_ROOT
    mkdir -p $REMOTE_ROOT
    tar xzf /tmp/tmx_code.tar.gz -C $(dirname "$REMOTE_ROOT")/ --strip-components=0
    echo \"Code synced.\"

    cd $REMOTE_ROOT

    # Create venv if needed
    if [ ! -d .venv_tpu_slice ]; then
      echo \"Creating venv...\"
      python3 -m venv .venv_tpu_slice
    fi
    source .venv_tpu_slice/bin/activate

    # Install deps
    echo \"Installing dependencies...\"
    pip install -q -e . 2>&1 | tail -5

    # Fix hf_xet deadlock
    pip uninstall -y hf-xet 2>/dev/null || true
    echo \"hf_xet removed (prevents download deadlock on TPU VMs)\"

    # Quick flash attention test
    echo \"Testing flash attention...\"
    TPU_VISIBLE_CHIPS=0 \
    TPU_PROCESS_BOUNDS=1,1,1 \
    TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
    TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 \
    TPU_HOST_BOUNDS=1,1,1 \
    TPU_WORKER_HOSTNAMES=localhost \
    TPU_WORKER_ID=0 \
    PJRT_DEVICE=TPU \
    python3 -c "
import torch, torch_xla.core.xla_model as xm
from torch_xla.experimental.custom_kernel import flash_attention
device = xm.xla_device()
q = torch.randn(1, 8, 1024, 128, dtype=torch.bfloat16, device=device)
k = torch.randn(1, 8, 1024, 128, dtype=torch.bfloat16, device=device)
v = torch.randn(1, 8, 1024, 128, dtype=torch.bfloat16, device=device)
q.requires_grad_(True)
out = flash_attention(q, k, v, causal=True)
xm.mark_step()
out.sum().backward()
xm.mark_step()
print(f\"FLASH ATTENTION OK: output={out.shape}, grad_nan={torch.isnan(q.grad).any().item()}\")
" 2>&1 || echo \"WARN: flash attention test failed\"

    # Test vLLM connectivity (first server)
    echo \"Testing vLLM connectivity...\"
    curl -s --connect-timeout 5 ${TMX_VLLM_SERVER_URLS%%,*}/models | head -1 || echo \"WARN: vLLM server unreachable (may need VPC peering)\"

    echo \"Bootstrap complete.\"
  '" 2>&1

# --- Launch pipeline ---
log "Launching full pipeline on $TPU_NAME..."
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --tunnel-through-iap \
  --command="bash -c '
    cd $REMOTE_ROOT
    source .venv_tpu_slice/bin/activate
    TMX_VLLM_SERVER_URLS=$TMX_VLLM_SERVER_URLS nohup bash scripts/launch_v6e_full_pipeline.sh > /tmp/full_pipeline_launch.log 2>&1 &
    echo \"Pipeline launched with PID \$!\"
    sleep 5
    head -30 /tmp/full_pipeline_launch.log 2>/dev/null || true
  '" 2>&1

log "Full pipeline launched on $TPU_NAME ($ZONE)"
log "Monitor: gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 --tunnel-through-iap --command='tail -50 $REMOTE_ROOT/runs/v6e_full_pipeline_*/console.log'"
