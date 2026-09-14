#!/usr/bin/env bash
# Gate 1: container/import smoke.
#
# After bootstrap_tunix_env.sh has installed Tunix+MaxText on the TPU workers,
# verify imports + JAX device visibility + HF read access. Does NOT launch
# any training. For multi-host pod slices, set TPU_WORKER=all; otherwise JAX
# TPU init can hang waiting for the rest of the hosts.
#
# Required env:
#   TPU_PROJECT, TPU_ZONE, TPU_NAME (a TRC v5litepod-256 or a v5litepod-64 in your own project)
# Optional env:
#   TPU_WORKER  default 0
#   ENV_NAME    default tmx_tunix_py312
#   HF_TOKEN    if set, also tests HF read access
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT}"
TPU_ZONE="${TPU_ZONE:?set TPU_ZONE}"
TPU_NAME="${TPU_NAME:?set TPU_NAME}"
TPU_WORKER="${TPU_WORKER:-all}"
ENV_NAME="${ENV_NAME:-tmx_tunix_py312}"
HOME_DIR="${HOME_DIR:?set HOME_DIR to the home directory on the TPU VM}"

echo "[gate1] tpu=${TPU_NAME} worker=${TPU_WORKER}"

echo "[gate1] syncing tmx package to ${TPU_NAME}:${TPU_WORKER}"
gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm scp \
    --recurse "${ROOT_DIR}/tmx" \
    "${TPU_NAME}:${HOME_DIR}/two-model-exploration/" \
    --worker="${TPU_WORKER}" \
    --zone="${TPU_ZONE}"

REMOTE=$(cat <<REMOTE
set -euo pipefail
source "${HOME_DIR}/miniforge3/bin/activate" "${ENV_NAME}"

python - <<'PY'
import jax
print(f"[gate1] jax {jax.__version__}")
print(f"[gate1] local_devices={len(jax.local_devices())}")
print(f"[gate1] global_devices={len(jax.devices())}")
print(f"[gate1] process_index={jax.process_index()} process_count={jax.process_count()}")

import maxtext  # noqa: F401
print("[gate1] maxtext import OK")

import tunix  # noqa: F401
from tunix.rl.grpo.drgrpo_learner import DrGRPOConfig  # noqa: F401
from tunix.rl.grpo.dapo_learner import DAPOConfig  # noqa: F401
from tunix.cli.config import obtain_reward_fn  # noqa: F401
print("[gate1] tunix DrGRPO + DAPO + reward_loader OK")

import sys
sys.path.insert(0, "${HOME_DIR}/two-model-exploration")
from tmx.orchestration.tunix_rewards import correctness, incorrectness_penalty
out_c = correctness(["p"], ["\\\\boxed{42}"], answer=["42"])
out_i = incorrectness_penalty(["p"], ["\\\\boxed{99}"], answer=["42"])
assert out_c == [1.0], out_c
assert out_i == [-1.0], out_i
print("[gate1] tmx.tunix_rewards signature + behavior OK")

print("[gate1] PASS")
PY

if [ -n "\${HF_TOKEN:-}" ]; then
    HF_TOKEN="\${HF_TOKEN}" python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files(
    "SaifPunjwani/two-model-exploration-checkpoints",
    repo_type="dataset",
)
print(f"[gate1] HF read OK: {len(files)} files visible")
ck = "dapo16k_drgrpo_novelty05_hf_resume49_20260429_174651_wandbresume_parallelvllm/explorer/checkpoints/step_000100"
present = any(f.startswith(ck) for f in files)
print(f"[gate1] step_000100 checkpoint visible: {present}")
print("[gate1] HF PASS")
PY
fi
REMOTE
)

gcloud --project="${TPU_PROJECT}" alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --worker="${TPU_WORKER}" \
    --batch-size="${TPU_BATCH_SIZE:-64}" \
    --zone="${TPU_ZONE}" \
    --command="HF_TOKEN='${HF_TOKEN:-}' bash -s" <<< "${REMOTE}"

echo "[gate1] OK"
