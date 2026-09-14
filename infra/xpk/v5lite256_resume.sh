#!/usr/bin/env bash
# One-command resume of the λ=0.5 paper run on TRC2 v5litepod-256.
#
# Pre-conditions (verified by this script):
#   - TRC2 v5litepod-256 exists (the nanny created it; we never create TPUs).
#   - External IPs ON, no NAT.
#   - HF token in env (HF_TOKEN).
#   - WANDB API key in env (WANDB_API_KEY).
#
# Default effect:
#   Run the new Tunix/MaxText/vLLM architecture gates and, after they pass,
#   launch the λ=0.5 step_000100 continuation on the TRC v5litepod-256.
#   vLLM remains the KV-cached rollout engine; Tunix/MaxText owns the JAX
#   training loop; TMX owns only reward/novelty/storage glue.
#
# Tunable env (defaults shown):
#   TPU_PROJECT     <trc-project>
#   TPU_ZONE        europe-west4-b
#   TPU_NAME        trc2-v5lite256-ew4b-r1
#   TPU_WORKER      all
#   TMX_USE_TUNIX_EXPERIMENTAL 1
#   DRY_RUN         0
#   SKIP_BOOTSTRAP  0   (set 1 to skip the install)
#   SKIP_PREFLIGHT  0   (set 1 to skip local checks)
#   GATES           "1 3 4 6"
#   TIERS           "1 2 3"   (gate 4 rollout tiers; tier 3 = 16K)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}"
export TPU_ZONE="${TPU_ZONE:-europe-west4-b}"
export TPU_NAME="${TPU_NAME:-trc2-v5lite256-ew4b-r1}"
export TMX_USE_TUNIX_EXPERIMENTAL="${TMX_USE_TUNIX_EXPERIMENTAL:-1}"
export DRY_RUN="${DRY_RUN:-0}"
export TPU_WORKER="${TPU_WORKER:-all}"
export TPU_BATCH_SIZE="${TPU_BATCH_SIZE:-64}"
export ENV_NAME="${ENV_NAME:-tmx_tunix_py312}"
export GATES="${GATES:-1 3 4 6}"
export TIERS="${TIERS:-1 2 3}"
export SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
export SPEC="${SPEC:-${ROOT_DIR}/configs/tunix/tmx_dapo_drgrpo_resume_step100.json}"

echo "[v5lite256] target=${TPU_NAME}@${TPU_PROJECT}/${TPU_ZONE} worker=${TPU_WORKER}"
echo "[v5lite256] gates=[${GATES}] tier4=[${TIERS}]"
echo "[v5lite256] spec=${SPEC}"

# Refuse the run if HF/WANDB tokens aren't present — it'd just fail later
# at upload time, more loudly and after wasting compute.
if [ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "[v5lite256] ERROR: set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) before launching" >&2
    exit 2
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[v5lite256] WARN: WANDB_API_KEY not set; metrics won't reach W&B"
fi

if [ "${SKIP_PREFLIGHT}" != "1" ]; then
    bash "${ROOT_DIR}/infra/xpk/preflight_local.sh"
fi

if [ "${TMX_USE_TUNIX_EXPERIMENTAL}" = "1" ]; then
    echo "[v5lite256] new architecture path: Tunix + MaxText + vLLM-on-TPU"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "bash ${ROOT_DIR}/infra/xpk/run_paper_pipeline.sh"
        exit 0
    fi
    bash "${ROOT_DIR}/infra/xpk/run_paper_pipeline.sh"
else
    echo "[v5lite256] fallback path: raw TPU + tmx-jax + vLLM"
    export TRC2_PROJECT="${TPU_PROJECT}"
    export TRC2_ZONE="${TPU_ZONE}"
    export TRC2_TPU_NAME="${TPU_NAME}"
    export SOURCE_RUN="${SOURCE_RUN:-dapo16k_drgrpo_novelty05_hf_resume49_20260429_174651_wandbresume_parallelvllm}"
    export SOURCE_STEP="${SOURCE_STEP:-000100}"
    export LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.5}"
    export TMX_FRESH_RUN="${TMX_FRESH_RUN:-0}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "TRC2_PROJECT=${TRC2_PROJECT} TRC2_ZONE=${TRC2_ZONE} TRC2_TPU_NAME=${TRC2_TPU_NAME} SOURCE_RUN=${SOURCE_RUN} SOURCE_STEP=${SOURCE_STEP} LAMBDA_NOVELTY=${LAMBDA_NOVELTY} TMX_FRESH_RUN=${TMX_FRESH_RUN} ${ROOT_DIR}/scripts/launch_trc2_v5lite256_tmx_jax_resume.sh"
        exit 0
    fi
    exec "${ROOT_DIR}/scripts/launch_trc2_v5lite256_tmx_jax_resume.sh"
fi

echo "[v5lite256] DONE"
