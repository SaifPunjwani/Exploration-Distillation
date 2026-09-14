#!/bin/bash
# Self-contained TPU bootstrap — run as user (not root) on any v6e or v4
set -x
export PATH=$HOME/.local/bin:$PATH

echo "=== Bootstrap started at $(date -u) ==="

# 1. Install pinned packages (transformers 4.46 for tpu_flash support)
pip install --quiet "torch==2.6.0" "torch_xla[tpu]==2.6.0" -f https://storage.googleapis.com/libtpu-releases/index.html
pip install --quiet "transformers==4.46.0" "accelerate==1.2.1" datasets trl peft sentencepiece protobuf wandb

# 2. Fix libtpu NaN on v6e (safe no-op on v4)
pip install --quiet --force-reinstall "libtpu-nightly==0.1.dev20241115+nightly" -f https://storage.googleapis.com/libtpu-releases/index.html 2>/dev/null || true

# 3. Deploy code from GCS
mkdir -p ~/two-model-exploration/tmx ~/two-model-exploration/scripts ~/checkpoints/Qwen3-1.7B
gsutil -m -q cp "gs://two-model-exploration-checkpoints/code/tmx/*.py" ~/two-model-exploration/tmx/
gsutil -q cp gs://two-model-exploration-checkpoints/code/main.py ~/two-model-exploration/main.py
gsutil -q cp gs://two-model-exploration-checkpoints/code/scripts/orchestrate_v4_runs.sh ~/two-model-exploration/scripts/

# 4. Download secrets
gsutil -q cp gs://two-model-exploration-checkpoints/secrets/wandb_key ~/.wandb_key 2>/dev/null || true

# 5. Download model
if [ ! -f ~/checkpoints/Qwen3-1.7B/config.json ]; then
    echo "Downloading model..."
    gsutil -m -q cp -r "gs://two-model-exploration-checkpoints/base_models/Qwen3-1.7B/*" ~/checkpoints/Qwen3-1.7B/
fi

# 6. Download checkpoint-50 and trajectories
mkdir -p ~/two-model-exploration/runs/actual_grpo_checkpoint/checkpoint-50
gsutil -m -q cp "gs://two-model-exploration-checkpoints/v6e_actual_grpo/checkpoint-50/checkpoint-50/*" \
    ~/two-model-exploration/runs/actual_grpo_checkpoint/checkpoint-50/ 2>/dev/null || true
mkdir -p ~/two-model-exploration/runs/precomputed_trajectories
gsutil -q cp "gs://two-model-exploration-checkpoints/precomputed_trajectories/explorer_trajectories.jsonl" \
    ~/two-model-exploration/runs/precomputed_trajectories/ 2>/dev/null || true

# 7. Verify
echo "=== Verify ==="
python3 -c "
import torch, torch_xla.core.xla_model as xm
d = xm.xla_device()
t = torch.randn(2,2).to(d)
xm.mark_step()
print(f'TPU OK: {d}')
import trl, transformers
print(f'trl={trl.__version__} transformers={transformers.__version__}')
" 2>&1
ls ~/checkpoints/Qwen3-1.7B/config.json && echo "Model OK"
ls ~/two-model-exploration/main.py && echo "Code OK"

echo "=== Bootstrap complete at $(date -u) ==="
echo "BOOTSTRAP COMPLETE" > /tmp/bootstrap_done

# 8. Launch orchestrator
cd ~/two-model-exploration
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null)
export PYTHONUNBUFFERED=1
nohup bash scripts/orchestrate_v4_runs.sh > /tmp/orchestrator_v4.log 2>&1 &
echo "Orchestrator PID=$!" | tee /tmp/orchestrator_pid
