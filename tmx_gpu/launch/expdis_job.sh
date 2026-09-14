#!/usr/bin/env bash
# Lepton job entrypoint: ExpDis scout->pool->SFT->central pipeline on one GPU node.
# Mirrors base_eval_job.sh conventions (env-driven, W&B-durable state).
#
# GPU layout: GPU 0 = training (grpo_gpu/sft_gpu), GPUs 1..N-1 = vLLM pool.
# All durable state (metrics, trajectories, accepted.jsonl, run_summary.json,
# final model) is pushed to W&B — log collection is off in the Lepton workspace.
#
# Expects env:
#   WANDB_API_KEY   (secret)                 CODE_ARTIFACT (wandb artifact ref;
#     fresh-c8 requires an immutable version, e.g. entity/project/code:v27)
#   TRAINING_CONTAINER_IMAGE_IDENTITY (strict C9/MR-ME only; must equal the
#     canonical immutable v0.25.0 tag@sha256 identity embedded below)
#   CODE_ARTIFACT_DIGEST, ORCHESTRATOR_COMMAND_SHA256, and the RUN_NAME-unique
#     ORCHESTRATOR_RECEIPT_ARTIFACT=entity/project/<run>-orchestrator-receipt:latest
#     are mandatory for strict C9/MR-ME and must be present in the remote spec.
#   RUN_NAME        (required)               MODEL          (default Qwen/Qwen3-1.7B)
#   ROUNDS=1  SCOUTS_PER_ROUND=1  SCOUT_STEPS=200  CENTRAL_STEPS=100
#     Stabilized integrated MultiScout sets ROUNDS=1, SCOUTS_PER_ROUND>1;
#     disjoint-shard annealed MR-ME sets ROUNDS>1, SCOUTS_PER_ROUND>1.
#     SINGLE_SCOUT_MULTIROUND_PROFILE=c9_selected_batch_single_scout_multiround
#     explicitly selects the distinct signed K=1, R=1..4 method.
#     immutable CODE_ARTIFACT and runs K full-budget Scouts per 1/R shard.
#   LAMBDA=0.5  LAMBDA_SCHEDULE="0.75,0.5,0.35,0.25"  POLICY=coverage_pool_c8
#   MIN_ACCEPTED=500  MAX_ACCEPTED=500 (contract: TPU production cap;
#     2000 is a GPU-MAX change and flips config_tier)
#   RELOAD_EVERY=10  SAVE_EVERY=25  KEEP_CHECKPOINTS=1
#   CKPT_ARTIFACT_EVERY=25   PROBE_EVERY=25  DYNAMIC_MAX_ATTEMPTS=0
#   RL_VALIDITY_MODE=c8 (legacy reproduces c7 stale-policy behavior)
#   EXPDIS_STABILIZATION_MODE=c8_candidate_attempt_update (frozen C8 default) or
#     c9_selected_batch_update; NOVELTY_NORMALIZATION=raw or
#     selected_batch_zscore. C9 requires SCIENTIFIC_VARIANT exactly
#     expdis_c9_selected_batch_rnd and immutable CODE_ARTIFACT :vN evidence.
#   SCOUT_LR="" (implicit C8 5e-6; C9 canary commonly 1e-6)
#   CENTRAL_LR="" (implicit 1e-6)
#   PROMPTS_PER_STEP=4  NUM_GENERATIONS=16  DATASET=dapo_math_17k
#   MAX_TRAIN_EXAMPLES=20000  SEED=0  NUM_GPUS (default: all visible)
#   MAX_MODEL_LEN=36864 (fits 32k P1 probes; training rollouts still use the completion budget)
#   SKIP_SCOUT=0  TRAJECTORIES="" (space list of local jsonl paths)
#     (TRAJ_ARTIFACTS auto-forces SKIP_SCOUT=1 with a warning: pooled
#      trajectories are meaningless without --skip-scout)
#   SFT_TERMINATION_MODE=append_eos (c8; use legacy only to reproduce c7)
#   SFT_GRAD_ACCUM=1 (c8/TPU parity; c7 used the labeled deviation 8)
#   SFT_SAVE_STEPS="" (for exposure-gated selection preregister exactly
#     16,32,64,126; empty preserves canonical final-only behavior)
#   FRESH_C8_SFT_GATE=0 (set 1 only on a fresh canonical SingleScout run;
#     requires STOP_AFTER_SFT=1, SFT_GRAD_ACCUM=1, and
#     SFT_SAVE_STEPS=125,250,500,1000)
#   FRESH_C8_MULTISCOUT_COMPONENT=0 (set 1 only for a Scout-only seed 1/2
#     component; binds the same exact c8 Scout/code/12,800-row evidence without
#     weakening the canonical seed-0 SingleScout SFT profile)
#   C8_SOURCE_BALANCE=0 (set 1 only for pooled MultiScout c8 runs)
#   STOP_AFTER_SFT=0 (set 1 for the mandatory SFT-only evaluation gate)
#   FROZEN_SFT_LIBRARY_ARTIFACT="" (training-only ga8 diagnostic; must be the
#     exact c7-reuse results:v1 artifact, with TRAJ_ARTIFACTS, SKIP_SCOUT=1,
#     STOP_AFTER_SFT=1, SFT_GRAD_ACCUM=8, SFT_SAVE_STEPS=16,32,64,126)
#   SFT_MODEL_ARTIFACT="" SFT_ACCEPTED_ARTIFACT=""
#   SFT_GATE_BUNDLE_ARTIFACTS="16=ref 32=ref 64=ref 126=ref"
#   SFT_SELECTION_MANIFEST_ARTIFACT="" (all four evidence classes are required
#     together for a fail-closed external exposure-gated Central continuation;
#     direct local SFT_MODEL is rejected by this launcher)
#   SFT_EVIDENCE_PROFILE=c7_reuse_ga8 (or fresh_c8_single_scout; controls the
#     exact preregistered gate steps and evidence-index version)
#   RESUME=1 (resume from pipeline_state.json when the job retries on the
#     same /workspace; a fresh workspace starts from scratch — restore a
#     checkpoint artifact + pipeline_state.json first to resume across pods)
#   PUSH_MODEL_ARTIFACT=1  EXTRA_ARGS="" (shell-lexed passthrough for the
#     small set of operational pipeline flags not already controlled here;
#     duplicate/provenance-bound flags are rejected before any downloads)
# GPU-MAX tier (defaults = contract tier; any change is logged as config_tier):
#   COMPLETION_BUDGET=0 (e.g. 24576/32768)  PPO_MODE=0  UPDATES_PER_ROLLOUT=1
#   OPTIMIZER=adamw  SFT_OPTIMIZER=adamw
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# --- durable logging: mirror everything to a file and push to W&B on exit ----
# (base_eval_job.sh convention: log collection is off in the Lepton workspace — streamed
# logs vanish with the pod, so job.log + the pool's vllm logs must reach W&B
# even when the pipeline dies early.)
exec > >(tee -a /workspace/job.log) 2>&1
upload_logs() {
  status=$?
  echo "[bootstrap] exit status=$status — uploading logs to W&B"
  python3 - "$status" <<'PY' || true
import glob, os, re, sys, wandb
run_name = os.environ.get("RUN_NAME", "unknown")
out_dir = os.environ.get("OUT_DIR", f"/workspace/runs/{run_name}")
artifact_metadata = {}
single_scout_multiround = bool(
    os.environ.get("SINGLE_SCOUT_MULTIROUND_PROFILE", ""))
if single_scout_multiround:
    artifact_metadata = {
        "method": "disjoint_shard_annealed_single_scout_multiround",
        "method_version":
            "gpu_disjoint_shard_annealed_single_scout_multiround_v1",
        "method_is_failure_targeted_curriculum": False,
        "mrme": False,
        "multi_explorer": False,
        "single_scout_multiround": True,
        "stabilization_profile": "c9_selected_batch_multi_explorer",
        "rounds": int(os.environ.get("ROUNDS", "1")),
        "scouts_per_round": 1,
        "base_seed": int(os.environ.get("SEED", "0")),
    }
elif int(os.environ.get("SCOUTS_PER_ROUND", "1")) > 1:
    rounds = int(os.environ.get("ROUNDS", "1"))
    artifact_metadata = {
        "method": ("integrated_multiscout" if rounds == 1 else
                   "disjoint_shard_annealed_mrme"),
        "method_is_failure_targeted_curriculum": False,
        "stabilization_profile": os.environ.get(
            "C9_MULTIEXPLORER_PROFILE") or
            "c8_candidate_attempt_multi_explorer",
        "rounds": rounds,
        "scouts_per_round": int(os.environ["SCOUTS_PER_ROUND"]),
        "base_seed": int(os.environ.get("SEED", "0")),
    }
elif os.environ.get("FRESH_C8_MULTISCOUT_COMPONENT", "0") == "1":
    artifact_metadata = {
        "fresh_c8_evidence_profile": "fresh_c8_multiscout_component",
        "component_seed": int(os.environ["SEED"]),
    }
elif os.environ.get("FRESH_C8_SFT_GATE", "0") == "1":
    artifact_metadata = {
        "fresh_c8_evidence_profile": "fresh_c8_single_scout",
        "seed": int(os.environ.get("SEED", "0")),
    }
elif (os.environ.get("C9_EXECUTION_PROFILE") or
      os.environ.get("EXPDIS_STABILIZATION_MODE") == "c9_selected_batch_update"):
    artifact_metadata = {
        "expdis_stabilization_mode": os.environ["EXPDIS_STABILIZATION_MODE"],
        "c9_execution_profile": os.environ.get("C9_EXECUTION_PROFILE", ""),
        "c10_execution_profile": os.environ.get("C10_EXECUTION_PROFILE", ""),
        "stabilization_cell": os.environ.get("STABILIZATION_CELL", ""),
        "scientific_variant": os.environ["SCIENTIFIC_VARIANT"],
        "novelty_normalization": os.environ["NOVELTY_NORMALIZATION"],
        "scout_lr": (float(os.environ["SCOUT_LR"])
                     if os.environ.get("SCOUT_LR") else 5e-6),
        "truncation_baseline_mode": os.environ.get(
            "TRUNCATION_BASELINE_MODE", "all_rows"),
        "mask_truncated": os.environ.get("MASK_TRUNCATED", "0") == "1",
    }
rid = re.sub(r"[^a-zA-Z0-9_-]", "-", f"joblogs-{run_name}")[:120]
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "two-model-explore-gpu"),
                 entity=os.environ.get("WANDB_ENTITY") or None,
                 name=f"joblogs-{run_name}", job_type="job-logs",
                 resume="allow", id=rid)
art = wandb.Artifact(
    re.sub(r"[^a-zA-Z0-9_.-]", "-", f"joblogs-{run_name}"),
    type="logs", metadata=artifact_metadata)
for f in ["/workspace/job.log"] + sorted(glob.glob(os.path.join(out_dir, "vllm_logs", "*.log"))):
    if os.path.exists(f):
        art.add_file(f, name=os.path.basename(f))
run.log_artifact(art)
run.summary["exit_status"] = int(sys.argv[1])
run.finish()
print("[bootstrap] logs uploaded")
PY
}
trap upload_logs EXIT

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODEL_REVISION="${MODEL_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
CANONICAL_TRAINING_CONTAINER_IMAGE="vllm/vllm-openai:v0.25.0@sha256:fc56161ee42a011aeee78b65d0a81b6683c7d04402fd40503d14d4d6c98f07cb"
TRAINING_CONTAINER_IMAGE_IDENTITY="${TRAINING_CONTAINER_IMAGE_IDENTITY:-}"
RUN_NAME="${RUN_NAME:?RUN_NAME required}"
CODE_ARTIFACT_DIGEST="${CODE_ARTIFACT_DIGEST:-}"
ORCHESTRATOR_COMMAND_SHA256="${ORCHESTRATOR_COMMAND_SHA256:-}"
ORCHESTRATOR_RECEIPT_ARTIFACT="${ORCHESTRATOR_RECEIPT_ARTIFACT:-}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"  # fits P1-budget probes (2048+32768); rollouts still cap at the completion budget
ROUNDS="${ROUNDS:-1}"
SCOUTS_PER_ROUND="${SCOUTS_PER_ROUND:-1}"
SCOUT_STEPS="${SCOUT_STEPS:-200}"
CENTRAL_STEPS="${CENTRAL_STEPS:-100}"
# Track which lambda envs the caller ACTUALLY set: pipeline_gpu gives an
# explicit --lambda-novelty precedence over --lambda-schedule (and errors if
# both are passed inconsistently for a multi-round run), so only forward the
# flags the caller chose.
LAMBDA_EXPLICIT="${LAMBDA:+1}"
LAMBDA_SCHEDULE_EXPLICIT="${LAMBDA_SCHEDULE:+1}"
LAMBDA="${LAMBDA:-0.5}"
LAMBDA_SCHEDULE="${LAMBDA_SCHEDULE:-0.75,0.5,0.35,0.25}"
POLICY="${POLICY:-coverage_pool_c8}"
MIN_ACCEPTED="${MIN_ACCEPTED:-500}"
MAX_ACCEPTED="${MAX_ACCEPTED:-500}"
HARVEST_MAX_PASSES="${HARVEST_MAX_PASSES:-0}"
RELOAD_EVERY="${RELOAD_EVERY:-10}"
RL_VALIDITY_MODE="${RL_VALIDITY_MODE:-c8}"
EXPDIS_STABILIZATION_MODE="${EXPDIS_STABILIZATION_MODE:-c8_candidate_attempt_update}"
NOVELTY_NORMALIZATION="${NOVELTY_NORMALIZATION:-raw}"
C9_EXECUTION_PROFILE="${C9_EXECUTION_PROFILE:-}"
C10_EXECUTION_PROFILE="${C10_EXECUTION_PROFILE:-}"
C9_MULTIEXPLORER_PROFILE="${C9_MULTIEXPLORER_PROFILE:-}"
SINGLE_SCOUT_MULTIROUND_PROFILE="${SINGLE_SCOUT_MULTIROUND_PROFILE:-}"
STABILIZATION_CELL="${STABILIZATION_CELL:-}"
SCIENTIFIC_VARIANT="${SCIENTIFIC_VARIANT:-}"
SCOUT_LR="${SCOUT_LR:-}"
CENTRAL_LR="${CENTRAL_LR:-}"
SCOUT_ONLY="${SCOUT_ONLY:-0}"
SKIP_SCOUT="${SKIP_SCOUT:-0}"
TRAJECTORIES="${TRAJECTORIES:-}"
SAVE_EVERY="${SAVE_EVERY:-25}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-1}"
CKPT_ARTIFACT_EVERY="${CKPT_ARTIFACT_EVERY:-25}"
PROBE_EVERY="${PROBE_EVERY:-25}"
DYNAMIC_MAX_ATTEMPTS="${DYNAMIC_MAX_ATTEMPTS:-0}"
RESUME="${RESUME:-1}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
DATASET="${DATASET:-dapo_math_17k}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
SEED="${SEED:-0}"
COMPLETION_BUDGET="${COMPLETION_BUDGET:-0}"
PPO_MODE="${PPO_MODE:-0}"
UPDATES_PER_ROLLOUT="${UPDATES_PER_ROLLOUT:-1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
SFT_OPTIMIZER="${SFT_OPTIMIZER:-adamw}"
SFT_TERMINATION_MODE="${SFT_TERMINATION_MODE:-append_eos}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-1}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-}"
MASK_TRUNCATED="${MASK_TRUNCATED:-0}"
TRUNCATION_BASELINE_MODE="${TRUNCATION_BASELINE_MODE:-all_rows}"
C8_SOURCE_BALANCE="${C8_SOURCE_BALANCE:-0}"
STOP_AFTER_SFT="${STOP_AFTER_SFT:-0}"
FRESH_C8_SFT_GATE="${FRESH_C8_SFT_GATE:-0}"
FRESH_C8_MULTISCOUT_COMPONENT="${FRESH_C8_MULTISCOUT_COMPONENT:-0}"
SFT_EVIDENCE_PROFILE="${SFT_EVIDENCE_PROFILE:-c7_reuse_ga8}"
DIRECT_SFT_MODEL="${SFT_MODEL:-}"
SFT_MODEL=""
SFT_EVIDENCE_INDEX=""
FROZEN_SFT_LIBRARY_ARTIFACT="${FROZEN_SFT_LIBRARY_ARTIFACT:-}"
export FROZEN_SFT_LIBRARY_ARTIFACT
FROZEN_SFT_LIBRARY_EVIDENCE_INDEX=""
TRAJECTORY_SOURCE_MANIFEST=""
CODE_SOURCE_MANIFEST=""
ORCHESTRATOR_RECEIPT_BINDING=""
C10_EXCLUSIVE_LEASE_BINDING=""
export SFT_EVIDENCE_PROFILE CODE_SOURCE_MANIFEST
export CODE_ARTIFACT CODE_ARTIFACT_DIGEST ORCHESTRATOR_COMMAND_SHA256
export ORCHESTRATOR_RECEIPT_ARTIFACT ORCHESTRATOR_RECEIPT_BINDING
export C10_EXCLUSIVE_LEASE_BINDING
export FRESH_C8_SFT_GATE FRESH_C8_MULTISCOUT_COMPONENT SEED
export EXPDIS_STABILIZATION_MODE NOVELTY_NORMALIZATION C9_EXECUTION_PROFILE
export C10_EXECUTION_PROFILE TRUNCATION_BASELINE_MODE MASK_TRUNCATED
export C9_MULTIEXPLORER_PROFILE
export SINGLE_SCOUT_MULTIROUND_PROFILE
export STABILIZATION_CELL SCIENTIFIC_VARIANT SCOUT_ONLY
export SCOUT_LR CENTRAL_LR
export ROUNDS SCOUTS_PER_ROUND
export RESUME
export SCOUT_STEPS CENTRAL_STEPS LAMBDA LAMBDA_SCHEDULE POLICY
export LAMBDA_EXPLICIT LAMBDA_SCHEDULE_EXPLICIT
export RELOAD_EVERY SAVE_EVERY KEEP_CHECKPOINTS CKPT_ARTIFACT_EVERY
export PROBE_EVERY DYNAMIC_MAX_ATTEMPTS PROMPTS_PER_STEP NUM_GENERATIONS
export DATASET MAX_TRAIN_EXAMPLES COMPLETION_BUDGET PPO_MODE
export UPDATES_PER_ROLLOUT OPTIMIZER SFT_OPTIMIZER MAX_MODEL_LEN
export SKIP_SCOUT TRAJECTORIES STOP_AFTER_SFT
WORK=/workspace
OUT_DIR="$WORK/runs/$RUN_NAME"
export RUN_NAME OUT_DIR MODEL MODEL_REVISION
export TRAINING_CONTAINER_IMAGE_IDENTITY
export WANDB_DIR="${WANDB_DIR:-$OUT_DIR/wandb}"
mkdir -p "$WORK" "$OUT_DIR" "$WANDB_DIR" && cd "$WORK"

# Parse EXTRA_ARGS exactly once into an argv array.  Expanding the raw string in
# the Python invocation both lost quoting and made review of duplicate flags
# unreliable.  shlex provides ordinary command-line quoting without eval, so
# shell metacharacters remain inert data.  Values containing NUL are rejected.
EXTRA_ARGV=()
parse_extra_args() {
  local raw="${EXTRA_ARGS:-}" token tmp
  [ -z "$raw" ] && return 0
  tmp="$(mktemp "${TMPDIR:-/tmp}/expdis-extra-args.XXXXXX")" || return 1
  if ! EXTRA_ARGS_RAW="$raw" python3 - "$tmp" <<'PY'
import os
import pathlib
import shlex
import sys

raw = os.environ.get("EXTRA_ARGS_RAW", "")
tokens = shlex.split(raw, posix=True)
if any("\0" in token for token in tokens):
    raise ValueError("EXTRA_ARGS contains NUL")
pathlib.Path(sys.argv[1]).write_bytes(
    b"".join(token.encode("utf-8") + b"\0" for token in tokens))
PY
  then
    rm -f "$tmp"
    echo "[bootstrap] FATAL: EXTRA_ARGS could not be parsed safely"
    return 1
  fi
  while IFS= read -r -d '' token; do
    EXTRA_ARGV+=("$token")
  done < "$tmp"
  rm -f "$tmp"
}

# Every option already supplied by this launcher, every scientific knob, and
# every provenance/evidence input must have a single source of truth.  argparse
# accepts duplicate long options and normally lets the last one win, including
# the --flag=value spelling.  Rejecting the complete protected set prevents the
# actual run from diverging from env-derived W&B metadata or signed manifests.
# Only operational vLLM capacity flags that have no env-derived duplicate here
# remain available through EXTRA_ARGS.
reject_protected_extra_args() {
  local token flag
  for token in "$@"; do
    flag="${token%%=*}"
    case "$flag" in
      --model|--model-revision|--container-image-identity|\
      --orchestrator-receipt-binding|--orchestrator-command-sha256|\
      --orchestrator-receipt-artifact|--orchestrator-run-name|\
      --scientific-variant|--output-dir|--run-name|\
      --rounds|--scouts-per-round|--scout-steps|--central-steps|\
      --explorers-per-round|--explorer-steps|--main-steps|\
      --scout-lr|--central-lr|--lambda-novelty|--lambda-schedule|\
      --explorer-lr|--main-lr|--explorer-only|--skip-explorer|\
      --novelty-layers|--expdis-stabilization-mode|\
      --novelty-normalization|--c9-execution-profile|\
      --c10-execution-profile|--c10-exclusive-lease-binding|\
      --truncation-baseline-mode|\
      --c9-multiexplorer-profile|\
      --single-scout-multiround-profile|\
      --stabilization-cell|--policy|--c8-source-balance|\
      --min-accepted|--max-accepted|--scout-only|--skip-scout|\
      --trajectories|--harvest-max-passes|--harvest-prompts-per-pass|\
      --prompts-per-step|--num-generations|--completion-budget|\
      --ppo-mode|--updates-per-rollout|--mask-truncated|\
      --no-mask-truncated|--optimizer|--sft-optimizer|--reload-every|\
      --rl-validity-mode|--save-every|--keep-checkpoints|\
      --ckpt-artifact-every|--probe-every|--probe-dataset|\
      --max-consecutive-gen-failures|--dynamic-max-attempts|--resume|\
      --sft-epochs|--sft-lr|--sft-grad-accum|\
      --sft-termination-mode|--sft-save-steps|--stop-after-sft|\
      --sft-model|--sft-evidence-index|\
      --frozen-sft-library-evidence-index|--fresh-c8-sft-gate|\
      --fresh-c8-multiscout-component|--code-source-manifest|\
      --sft-training-evidence-manifest|--trajectory-source-manifest|\
      --dataset|--max-train-examples|--sync-dir|--vllm-gpus|\
      --endpoints|--served-model-name|--vllm-max-model-len|\
      --gen-concurrency|\
      --device|--param-dtype|--seed|--wandb|\
      --code-artifact|--code-artifact-ref|--code-artifact-digest|\
      --code-manifest)
        echo "[bootstrap] FATAL: EXTRA_ARGS cannot override protected option $flag; use its dedicated launcher environment variable"
        return 1
        ;;
    esac
  done
}

parse_extra_args
reject_protected_extra_args "${EXTRA_ARGV[@]}"

if [[ ! "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$SCOUTS_PER_ROUND" =~ ^[1-9][0-9]*$ ]]; then
  echo "[bootstrap] FATAL: ROUNDS and SCOUTS_PER_ROUND must be positive integers"
  exit 1
fi
if [ "$SCOUTS_PER_ROUND" -gt 1 ] && [ "$ROUNDS" -le 1 ] && \
   [ "$C9_MULTIEXPLORER_PROFILE" != "c9_selected_batch_multi_explorer" ]; then
  echo "[bootstrap] FATAL: R=1,K>1 requires C9_MULTIEXPLORER_PROFILE=c9_selected_batch_multi_explorer"
  exit 1
fi
if [ -n "$SINGLE_SCOUT_MULTIROUND_PROFILE" ] && \
   { [ "$SINGLE_SCOUT_MULTIROUND_PROFILE" != \
       "c9_selected_batch_single_scout_multiround" ] || \
     [ "$SCOUTS_PER_ROUND" -ne 1 ] || [ "$ROUNDS" -gt 4 ] || \
     [ -n "$C9_MULTIEXPLORER_PROFILE" ] || \
     [ -n "$C9_EXECUTION_PROFILE" ] || [ -n "$STABILIZATION_CELL" ]; }; then
  echo "[bootstrap] FATAL: invalid K=1 single-Scout multi-round profile identity"
  exit 1
fi
if [ -n "$C10_EXECUTION_PROFILE" ] && \
   { [ "$C10_EXECUTION_PROFILE" != "canary" ] || \
     [ -n "$C9_EXECUTION_PROFILE" ] || [ -n "$STABILIZATION_CELL" ] || \
     [ -n "$C9_MULTIEXPLORER_PROFILE" ] || \
     [ -n "$SINGLE_SCOUT_MULTIROUND_PROFILE" ]; }; then
  echo "[bootstrap] FATAL: C10 canary identity cannot mix with C9/higher-order profiles"
  exit 1
fi

echo "[bootstrap] $(date -u) model=$MODEL revision=$MODEL_REVISION gpus=$NUM_GPUS run=$RUN_NAME rounds=$ROUNDS scouts_per_round=$SCOUTS_PER_ROUND"
nvidia-smi -L || true

STRICT_TRAINING_ENVIRONMENT=0
if [ -n "$C9_EXECUTION_PROFILE" ] || \
   [ -n "$C10_EXECUTION_PROFILE" ] || \
   [ -n "$C9_MULTIEXPLORER_PROFILE" ] || \
   [ -n "$SINGLE_SCOUT_MULTIROUND_PROFILE" ] || \
   [ "$EXPDIS_STABILIZATION_MODE" = "c9_selected_batch_update" ]; then
  STRICT_TRAINING_ENVIRONMENT=1
fi
if [ "$STRICT_TRAINING_ENVIRONMENT" = "1" ] && \
   [ "$TRAINING_CONTAINER_IMAGE_IDENTITY" != \
     "$CANONICAL_TRAINING_CONTAINER_IMAGE" ]; then
  echo "[bootstrap] FATAL: strict C9/MR-ME requires TRAINING_CONTAINER_IMAGE_IDENTITY=$CANONICAL_TRAINING_CONTAINER_IMAGE" >&2
  exit 1
fi
if [ "$STRICT_TRAINING_ENVIRONMENT" = "1" ] && \
   { [ -z "${CODE_ARTIFACT:-}" ] || [ -z "$CODE_ARTIFACT_DIGEST" ] || \
     [ -z "$ORCHESTRATOR_COMMAND_SHA256" ] || \
     [ -z "$ORCHESTRATOR_RECEIPT_ARTIFACT" ]; }; then
  echo "[bootstrap] FATAL: strict C9/MR-ME requires code digest, command SHA256, and RUN_NAME-unique orchestrator receipt artifact" >&2
  exit 1
fi
if [ "$STRICT_TRAINING_ENVIRONMENT" = "1" ] && \
   [[ ! "$CODE_ARTIFACT" =~ ^[^/[:space:]]+/[^/[:space:]]+/[^/:[:space:]]+:v[0-9]+$ ]]; then
  echo "[bootstrap] FATAL: strict C9/MR-ME CODE_ARTIFACT must be fully qualified entity/project/name:vN" >&2
  exit 1
fi
if [ "$STRICT_TRAINING_ENVIRONMENT" = "1" ] && [ "$NUM_GPUS" != "8" ]; then
  echo "[bootstrap] FATAL: strict C9/MR-ME requires exactly 8 visible GPUs" >&2
  exit 1
fi

# ---- deps: uv -> pip -> python -m pip fallback chain --------------------------
# The immutable v0.25 image supplies torch/vLLM/transformers/tokenizers.  Pin
# every added support package exactly; grpo_gpu independently checks the full
# resulting stack (including image-supplied packages) before a strict update.
export DEBIAN_FRONTEND=noninteractive
PKGS=("wandb==0.28.0" "datasets==5.0.0" "requests==2.34.2" \
      "huggingface_hub==1.21.0" "hf-transfer==0.1.9" \
      "numpy==2.2.6" "safetensors==0.8.0")
install_deps() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --system --quiet "${PKGS[@]}" 2>&1 | tail -1 && return 0
    echo "[bootstrap] uv install failed; falling back to pip"
  fi
  pip install --quiet --no-cache-dir "${PKGS[@]}" 2>&1 | tail -1 && return 0
  python3 -m pip install --quiet --no-cache-dir "${PKGS[@]}" 2>&1 | tail -1
}
install_deps || { echo "[bootstrap] FATAL: dependency install failed"; exit 1; }
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore-gpu}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

# ---- fetch code from the W&B artifact -----------------------------------------
if [ -n "${CODE_ARTIFACT:-}" ]; then
  python3 - <<'PY'
import hashlib, json, os, pathlib, re, shutil, wandb
api = wandb.Api()
art = api.artifact(os.environ["CODE_ARTIFACT"])
resolved_ref = str(getattr(art, "qualified_name", "") or "")
if not re.fullmatch(r"[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+", resolved_ref):
    raise RuntimeError(f"resolved code artifact ref is not immutable: {resolved_ref!r}")
expected_digest = os.environ.get("CODE_ARTIFACT_DIGEST", "")
if expected_digest and art.digest != expected_digest:
    raise RuntimeError("resolved code artifact digest differs from remote job spec")
safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", art.digest)
root = pathlib.Path("/workspace/artifacts/code") / safe
if root.exists():
    shutil.rmtree(root)
root.parent.mkdir(parents=True, exist_ok=True)
path = art.download(root=str(root))
expected = sorted(item.name for item in art.files())
actual = sorted(item.relative_to(root).as_posix()
                for item in root.rglob("*") if item.is_file())
if actual != expected:
    raise RuntimeError(f"code artifact membership mismatch: {actual} vs {expected}")
files = [{
    "relative_path": name,
    "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
    "size_bytes": (root / name).stat().st_size,
} for name in expected]
source = {"artifact_ref": resolved_ref,
          "artifact_digest": art.digest}
normalized = {"version": "wandb_artifact_download_v1", "source": source,
              "files": files}
manifest = {**normalized, "download_root": str(root.resolve()),
            "file_manifest_sha256": hashlib.sha256(
                json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode()).hexdigest()}
manifest_path = pathlib.Path("/workspace/code_download_manifest.json")
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
pathlib.Path("/workspace/code_path.txt").write_text(str(pathlib.Path(path).resolve()))
print(f"[bootstrap] code artifact -> {path}", flush=True)
PY
  CODE_ROOT="$(cat /workspace/code_path.txt)"
  CODE_SOURCE_MANIFEST=/workspace/code_download_manifest.json
else
  CODE_ROOT=/workspace/code
fi
cd "$CODE_ROOT"
test -f tmx_gpu/pipeline_gpu.py || { echo "[bootstrap] FATAL: code tree missing"; exit 1; }

# The controller captures `lep job get --path`, publishes the two-file receipt
# artifact, and this strict worker blocks here until that immutable evidence is
# available.  Only the RUN_NAME-unique :latest alias is polled; the binding
# persisted below contains the resolved qualified :vN ref and artifact digest.
ORCHESTRATOR_ARGS=()
if [ "$STRICT_TRAINING_ENVIRONMENT" = "1" ]; then
  ORCHESTRATOR_RECEIPT_BINDING="$OUT_DIR/orchestrator_receipt_binding.json"
  export ORCHESTRATOR_RECEIPT_BINDING
  readarray -t CODE_SOURCE_IDENTITY < <(python3 - "$CODE_SOURCE_MANIFEST" <<'PY'
import json, sys
source = json.load(open(sys.argv[1], encoding="utf-8"))["source"]
print(source["artifact_ref"])
print(source["artifact_digest"])
PY
  )
  python3 -m tmx_gpu.orchestrator_receipt resolve \
    --artifact-ref "$ORCHESTRATOR_RECEIPT_ARTIFACT" \
    --output-binding "$ORCHESTRATOR_RECEIPT_BINDING" \
    --download-root "$OUT_DIR/orchestrator-receipt" \
    --expected-run-name "$RUN_NAME" \
    --expected-container-image "$TRAINING_CONTAINER_IMAGE_IDENTITY" \
    --expected-code-artifact-ref "${CODE_SOURCE_IDENTITY[0]}" \
    --expected-code-artifact-digest "${CODE_SOURCE_IDENTITY[1]}" \
    --expected-command-sha256 "$ORCHESTRATOR_COMMAND_SHA256"
  ORCHESTRATOR_ARGS=(
    --orchestrator-receipt-binding "$ORCHESTRATOR_RECEIPT_BINDING"
    --orchestrator-command-sha256 "$ORCHESTRATOR_COMMAND_SHA256"
    --orchestrator-receipt-artifact "$ORCHESTRATOR_RECEIPT_ARTIFACT"
  )
fi

MRME_CODE_ARGS=()
if [ "$SCOUTS_PER_ROUND" -gt 1 ]; then
  if [ -z "$CODE_SOURCE_MANIFEST" ] || \
     [[ ! "${CODE_ARTIFACT:-}" =~ :v[0-9]+$ ]]; then
    echo "[bootstrap] FATAL: true MR-ME requires immutable CODE_ARTIFACT name:vN evidence"
    exit 1
  fi
  # C9 adds this argument through STABILIZATION_ARGS below.  Older C8
  # MultiExplorer needs it here; never pass the same singleton flag twice.
  if [ "$EXPDIS_STABILIZATION_MODE" != "c9_selected_batch_update" ]; then
    MRME_CODE_ARGS=(--code-source-manifest "$CODE_SOURCE_MANIFEST")
  fi
fi

# ---- optional: pooled trajectories from W&B results artifacts ------------------
# TRAJ_ARTIFACTS: space-separated expdis-results artifact refs (one per scout).
# Downloads each and collects ALL of its scout explorer_trajectories.jsonl
# files (every round — hits[0] alone silently dropped rounds >= 2) into
# TRAJECTORIES for --skip-scout pooled-central runs (MultiScout). Central-GRPO
# trajectory files are never picked up, even by the fallback glob.
if [ -n "${TRAJ_ARTIFACTS:-}" ]; then
  TRAJECTORIES="$(python3 - <<'PY'
import hashlib, json, os, pathlib, re, shutil, wandb
from tmx_gpu import fresh_c8_sft
api = wandb.Api()
paths = []
sources = []
for i, ref in enumerate(os.environ["TRAJ_ARTIFACTS"].split()):
    artifact = api.artifact(ref)
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
    root = pathlib.Path(f"/workspace/artifacts/trajectory/{i}-{safe}")
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(root))
    expected = sorted(file.name for file in artifact.files())
    actual = sorted(path.relative_to(root).as_posix()
                    for path in root.rglob("*") if path.is_file())
    assert actual == expected, f"trajectory artifact membership mismatch for {ref}"
    completion_evidence = [
        json.loads((root / name).read_text())
        for name in expected
        if name.endswith("fresh_c8_scout_completion_evidence.json")
    ]
    fresh_c8_sft.reject_multiscout_component_from_generic_pool(
        artifact_metadata=dict(getattr(artifact, "metadata", {}) or {}),
        completion_evidence=completion_evidence,
    )
    hits = [str((root / name).resolve()) for name in expected
            if "/scout/" in f"/{name}" and name.endswith("explorer_trajectories.jsonl")]
    if not hits:
        hits = [str((root / name).resolve()) for name in expected
                if name.endswith("explorer_trajectories.jsonl")
                and "/central/" not in f"/{name}"]
    assert hits, f"no scout explorer_trajectories.jsonl in artifact {ref}"
    paths.extend(hits)
    sources.append({
        "artifact_ref": ref,
        "artifact_digest": artifact.digest,
        "download_root": str(root.resolve()),
        "files": [
            {
                "relative_path": pathlib.Path(path).relative_to(root).as_posix(),
                "local_path": path,
                "sha256": hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),
            }
            for path in hits
        ],
    })
pathlib.Path("/workspace/trajectory_source_manifest.json").write_text(
    json.dumps({"version": "wandb_trajectory_sources_v1", "artifacts": sources},
               indent=2, sort_keys=True) + "\n")
print(" ".join(paths))
PY
)"
  export TRAJECTORIES
  TRAJECTORY_SOURCE_MANIFEST=/workspace/trajectory_source_manifest.json
  echo "[bootstrap] pooled trajectories: $TRAJECTORIES"
  if [ "${SKIP_SCOUT:-0}" != "1" ]; then
    echo "[bootstrap] WARNING: TRAJ_ARTIFACTS set without SKIP_SCOUT=1 — forcing SKIP_SCOUT=1 (pooled trajectories imply --skip-scout; pipeline_gpu errors otherwise)"
    SKIP_SCOUT=1
  fi
fi

# ---- optional: pre-evaluated SFT checkpoint from W&B -------------------------
# The external continuation preserves the exact frozen accepted library; it
# never re-filters the c7 bank under c8. The pipeline requires SKIP_SCOUT=1 +
# the original trajectory provenance, records the external health-gate
# selection in Stage 4, then starts exact correctness-only Central GRPO.
EXTERNAL_EVIDENCE_COUNT=0
for value in "${SFT_MODEL_ARTIFACT:-}" "${SFT_ACCEPTED_ARTIFACT:-}" \
             "${SFT_GATE_BUNDLE_ARTIFACTS:-}" \
             "${SFT_SELECTION_MANIFEST_ARTIFACT:-}"; do
  [ -n "$value" ] && EXTERNAL_EVIDENCE_COUNT=$((EXTERNAL_EVIDENCE_COUNT + 1))
done
if [ -n "$DIRECT_SFT_MODEL" ]; then
  echo "[bootstrap] FATAL: direct SFT_MODEL is not allowed; use exact W&B evidence artifacts"
  exit 1
fi
if [ -n "$FROZEN_SFT_LIBRARY_ARTIFACT" ] && [ "$EXTERNAL_EVIDENCE_COUNT" -ne 0 ]; then
  echo "[bootstrap] FATAL: frozen-library SFT training and external selected-model continuation are mutually exclusive"
  exit 1
fi
if [ "$FRESH_C8_SFT_GATE" = "1" ] && \
    { [ -n "$FROZEN_SFT_LIBRARY_ARTIFACT" ] || [ "$EXTERNAL_EVIDENCE_COUNT" -ne 0 ]; }; then
  echo "[bootstrap] FATAL: fresh-c8 gated SFT training is mutually exclusive with frozen/external SFT routes"
  exit 1
fi
if [ "$FRESH_C8_MULTISCOUT_COMPONENT" = "1" ] && \
    { [ "$FRESH_C8_SFT_GATE" = "1" ] || \
      [ -n "$FROZEN_SFT_LIBRARY_ARTIFACT" ] || \
      [ "$EXTERNAL_EVIDENCE_COUNT" -ne 0 ]; }; then
  echo "[bootstrap] FATAL: fresh-c8 MultiScout component is a disjoint Scout-only evidence profile"
  exit 1
fi
if { [ -n "$C9_EXECUTION_PROFILE" ] && [ -z "$STABILIZATION_CELL" ]; } || \
   { [ -z "$C9_EXECUTION_PROFILE" ] && [ -n "$STABILIZATION_CELL" ]; }; then
  echo "[bootstrap] FATAL: C9_EXECUTION_PROFILE and STABILIZATION_CELL are required together"
  exit 1
fi
if [ -n "$C9_EXECUTION_PROFILE" ] && \
   { [ -z "$CODE_SOURCE_MANIFEST" ] || [[ ! "${CODE_ARTIFACT:-}" =~ :v[0-9]+$ ]]; }; then
  echo "[bootstrap] FATAL: preregistered C9 profiles require immutable CODE_ARTIFACT :vN evidence"
  exit 1
fi
if [ -n "$C9_MULTIEXPLORER_PROFILE" ]; then
  if [ "$C9_MULTIEXPLORER_PROFILE" != "c9_selected_batch_multi_explorer" ] || \
     [ "$SCOUTS_PER_ROUND" -le 1 ] || \
     [ "$EXPDIS_STABILIZATION_MODE" != "c9_selected_batch_update" ] || \
     [ "$NOVELTY_NORMALIZATION" != "selected_batch_zscore" ] || \
     [ "$SCIENTIFIC_VARIANT" != "expdis_c9_selected_batch_multi_explorer" ] || \
     [ -n "$C9_EXECUTION_PROFILE" ] || [ -n "$STABILIZATION_CELL" ]; then
    echo "[bootstrap] FATAL: invalid stabilized C9 multi-explorer identity"
    exit 1
  fi
fi
if [ -n "$SINGLE_SCOUT_MULTIROUND_PROFILE" ]; then
  if [ "$SINGLE_SCOUT_MULTIROUND_PROFILE" != \
       "c9_selected_batch_single_scout_multiround" ] || \
     [ "$SCOUTS_PER_ROUND" -ne 1 ] || [ "$ROUNDS" -gt 4 ] || \
     [ "$EXPDIS_STABILIZATION_MODE" != "c9_selected_batch_update" ] || \
     [ "$NOVELTY_NORMALIZATION" != "selected_batch_zscore" ] || \
     [ "$SCIENTIFIC_VARIANT" != \
       "expdis_c9_selected_batch_multi_explorer" ] || \
     [ -n "$C9_MULTIEXPLORER_PROFILE" ] || \
     [ -n "$C9_EXECUTION_PROFILE" ] || [ -n "$STABILIZATION_CELL" ]; then
    echo "[bootstrap] FATAL: invalid stabilized C9 K=1 multi-round identity"
    exit 1
  fi
fi
if [ "$EXPDIS_STABILIZATION_MODE" = "c9_selected_batch_update" ]; then
  EXPECTED_C9_VARIANT="expdis_c9_stability_screen"
  if [ -n "$C10_EXECUTION_PROFILE" ]; then
    EXPECTED_C9_VARIANT="expdis_c10_survivor_consistent_overlong_filter"
  elif [ -n "$C9_MULTIEXPLORER_PROFILE" ] || \
     [ -n "$SINGLE_SCOUT_MULTIROUND_PROFILE" ]; then
    EXPECTED_C9_VARIANT="expdis_c9_selected_batch_multi_explorer"
  elif [ "$STABILIZATION_CELL" = "C5" ] || [ -z "$STABILIZATION_CELL" ]; then
    EXPECTED_C9_VARIANT="expdis_c9_selected_batch_rnd"
  fi
  if [ "$SCIENTIFIC_VARIANT" != "$EXPECTED_C9_VARIANT" ]; then
    echo "[bootstrap] FATAL: selected stabilization profile requires SCIENTIFIC_VARIANT=$EXPECTED_C9_VARIANT"
    exit 1
  fi
  if [ "$RL_VALIDITY_MODE" != "c8" ]; then
    echo "[bootstrap] FATAL: C9 requires RL_VALIDITY_MODE=c8"
    exit 1
  fi
  if [ "$FRESH_C8_SFT_GATE" = "1" ] || \
      [ "$FRESH_C8_MULTISCOUT_COMPONENT" = "1" ]; then
    echo "[bootstrap] FATAL: C9 is disjoint from frozen fresh-c8 evidence profiles"
    exit 1
  fi
  if [ -z "$CODE_SOURCE_MANIFEST" ] || \
      [[ ! "${CODE_ARTIFACT:-}" =~ :v[0-9]+$ ]]; then
    echo "[bootstrap] FATAL: C9 requires immutable CODE_ARTIFACT :vN evidence"
    exit 1
  fi
  if [ -n "$C10_EXECUTION_PROFILE" ] && \
     { [ "$NOVELTY_NORMALIZATION" != "selected_batch_zscore" ] || \
       [ "$RUN_NAME" != "tmx-c10-survivor-canary-0e3448b184c5" ] || \
       [ "$MASK_TRUNCATED" != "1" ] || \
       [ "$TRUNCATION_BASELINE_MODE" != "unmasked_survivors" ] || \
       [ "$SCOUT_LR" != "0.000001" ] || [ "$SCOUT_STEPS" != "50" ] || \
       [ "$SCOUT_ONLY" != "1" ] || [ "$PROBE_EVERY" != "0" ] || \
       [ "$RELOAD_EVERY" != "1" ]; }; then
    echo "[bootstrap] FATAL: C10 canary requires its unique canonical run name, selected-batch z-score, survivor-consistent filtering, LR 1e-6, 50 Scout-only steps, no probes, and reload-every=1"
    exit 1
  fi
elif [ "$EXPDIS_STABILIZATION_MODE" != "c8_candidate_attempt_update" ]; then
  echo "[bootstrap] FATAL: unknown EXPDIS_STABILIZATION_MODE=$EXPDIS_STABILIZATION_MODE"
  exit 1
elif [ "$NOVELTY_NORMALIZATION" != "raw" ]; then
  echo "[bootstrap] FATAL: frozen C8 requires NOVELTY_NORMALIZATION=raw"
  exit 1
fi
if [ "$EXTERNAL_EVIDENCE_COUNT" -ne 0 ] && [ "$EXTERNAL_EVIDENCE_COUNT" -ne 4 ]; then
  echo "[bootstrap] FATAL: SFT_MODEL_ARTIFACT, SFT_ACCEPTED_ARTIFACT, profile-specific four-step SFT_GATE_BUNDLE_ARTIFACTS, and SFT_SELECTION_MANIFEST_ARTIFACT are required together"
  exit 1
fi
if [ "$FRESH_C8_SFT_GATE" = "1" ]; then
  if [ -z "$CODE_SOURCE_MANIFEST" ]; then
    echo "[bootstrap] FATAL: fresh-c8 gated SFT requires exact CODE_ARTIFACT evidence"
    exit 1
  fi
  if [[ ! "${CODE_ARTIFACT:-}" =~ :v[0-9]+$ ]]; then
    echo "[bootstrap] FATAL: fresh-c8 CODE_ARTIFACT must be pinned to immutable name:vN (aliases such as latest are forbidden)"
    exit 1
  fi
  if [ "$STOP_AFTER_SFT" != "1" ] || [ "$SFT_GRAD_ACCUM" != "1" ] || \
      [ "$SFT_SAVE_STEPS" != "125,250,500,1000" ]; then
    echo "[bootstrap] FATAL: fresh-c8 gated SFT requires STOP_AFTER_SFT=1, SFT_GRAD_ACCUM=1, SFT_SAVE_STEPS=125,250,500,1000"
    exit 1
  fi
  if [ "${SKIP_SCOUT:-0}" = "1" ] || [ -n "${TRAJ_ARTIFACTS:-}" ]; then
    echo "[bootstrap] FATAL: fresh-c8 gated SFT must follow its own in-pipeline SingleScout"
    exit 1
  fi
fi
if [ "$FRESH_C8_MULTISCOUT_COMPONENT" = "1" ]; then
  if [ -z "$CODE_SOURCE_MANIFEST" ] || \
      [[ ! "${CODE_ARTIFACT:-}" =~ :v[0-9]+$ ]]; then
    echo "[bootstrap] FATAL: fresh-c8 MultiScout component requires immutable CODE_ARTIFACT :vN evidence"
    exit 1
  fi
  if [ "$SEED" != "1" ] && [ "$SEED" != "2" ]; then
    echo "[bootstrap] FATAL: fresh-c8 MultiScout component SEED must be exactly 1 or 2"
    exit 1
  fi
  if [ "${SKIP_SCOUT:-0}" = "1" ] || [ -n "${TRAJ_ARTIFACTS:-}" ] || \
      [ "$STOP_AFTER_SFT" = "1" ] || [ -n "$SFT_SAVE_STEPS" ]; then
    echo "[bootstrap] FATAL: fresh-c8 MultiScout component must be a fresh Scout-only run"
    exit 1
  fi
fi
# ---- exact frozen-library ga8 SFT training input -----------------------------
# This route trains SFT; it is not the selected-model continuation above and
# cannot claim canonical c8. Only accepted.jsonl is taken from results:v1.
# Exact Scout-trajectory provenance remains separately bound by TRAJ_ARTIFACTS.
if [ -n "$FROZEN_SFT_LIBRARY_ARTIFACT" ]; then
  if [ -z "$TRAJECTORY_SOURCE_MANIFEST" ]; then
    echo "[bootstrap] FATAL: frozen-library ga8 SFT requires TRAJ_ARTIFACTS provenance"
    exit 1
  fi
  if [ "${SKIP_SCOUT:-0}" != "1" ] || [ "$STOP_AFTER_SFT" != "1" ]; then
    echo "[bootstrap] FATAL: frozen-library ga8 SFT requires SKIP_SCOUT=1 and STOP_AFTER_SFT=1"
    exit 1
  fi
  if [ "$SFT_GRAD_ACCUM" != "8" ] || [ "$SFT_SAVE_STEPS" != "16,32,64,126" ]; then
    echo "[bootstrap] FATAL: frozen-library ga8 SFT requires SFT_GRAD_ACCUM=8 and SFT_SAVE_STEPS=16,32,64,126"
    exit 1
  fi
  python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import re
import shutil
import wandb

ref = os.environ["FROZEN_SFT_LIBRARY_ARTIFACT"]
artifact = wandb.Api().artifact(ref)
safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
root = pathlib.Path("/workspace/artifacts/frozen-sft-library") / safe
if root.exists():
    shutil.rmtree(root)
root.parent.mkdir(parents=True, exist_ok=True)
artifact.download(root=str(root))
expected = sorted(item.name for item in artifact.files())
actual = sorted(path.relative_to(root).as_posix()
                for path in root.rglob("*") if path.is_file())
if actual != expected:
    raise RuntimeError(f"frozen SFT artifact membership mismatch: {actual} vs {expected}")
hits = [name for name in expected if name.endswith("accepted.jsonl")]
if len(hits) != 1:
    raise RuntimeError(f"expected exactly one accepted.jsonl, got {hits}")
files = [{
    "relative_path": name,
    "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
    "size_bytes": (root / name).stat().st_size,
} for name in expected]
source = {"artifact_ref": ref, "artifact_digest": artifact.digest}
normalized = {"version": "wandb_artifact_download_v1",
              "source": source, "files": files}
manifest = {
    **normalized,
    "download_root": str(root.resolve()),
    "file_manifest_sha256": hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()).hexdigest(),
}
manifest_path = pathlib.Path("/workspace/frozen_sft_library_download.json")
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
index = {
    "version": "frozen_sft_library_evidence_v1",
    "accepted": {
        "source": source,
        "download_manifest": str(manifest_path.resolve()),
        "accepted_path": str((root / hits[0]).resolve()),
    },
}
index_path = pathlib.Path("/workspace/frozen_sft_library_evidence_index.json")
index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
print(f"[bootstrap] frozen-library training evidence -> {index_path}", flush=True)
PY
  FROZEN_SFT_LIBRARY_EVIDENCE_INDEX=/workspace/frozen_sft_library_evidence_index.json
fi
if [ "$EXTERNAL_EVIDENCE_COUNT" -eq 4 ]; then
  if [ -z "$TRAJECTORY_SOURCE_MANIFEST" ]; then
    echo "[bootstrap] FATAL: external SFT continuation requires TRAJ_ARTIFACTS provenance"
    exit 1
  fi
  python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import re
import shutil
import wandb

api = wandb.Api()
profile = os.environ.get("SFT_EVIDENCE_PROFILE", "c7_reuse_ga8")
if profile not in ("c7_reuse_ga8", "fresh_c8_single_scout"):
    raise RuntimeError(f"unknown SFT_EVIDENCE_PROFILE={profile!r}")
manifest_root = pathlib.Path("/workspace/evidence_manifests")
manifest_root.mkdir(parents=True, exist_ok=True)

def download(ref, category, key):
    artifact = api.artifact(ref)
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
    root = pathlib.Path("/workspace/artifacts") / category / f"{key}-{safe}"
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(root))
    expected = sorted(file.name for file in artifact.files())
    actual = sorted(path.relative_to(root).as_posix()
                    for path in root.rglob("*") if path.is_file())
    if actual != expected:
        raise RuntimeError(f"artifact membership mismatch for {ref}: {actual} vs {expected}")
    files = [{
        "relative_path": name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "size_bytes": (root / name).stat().st_size,
    } for name in expected]
    source = {"artifact_ref": ref, "artifact_digest": artifact.digest}
    normalized = {"version": "wandb_artifact_download_v1",
                  "source": source, "files": files}
    payload = {
        **normalized,
        "download_root": str(root.resolve()),
        "file_manifest_sha256": hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode()).hexdigest(),
    }
    path = manifest_root / f"{category}-{key}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return artifact, root.resolve(), expected, path.resolve(), source

model_art, model_root, model_files, model_manifest, model_source = download(
    os.environ["SFT_MODEL_ARTIFACT"], "sft-model", "selected")
model_dirs = sorted({(model_root / name).parent.resolve() for name in model_files
                     if name.endswith("config.json")
                     and any((model_root / other).parent.resolve()
                             == (model_root / name).parent.resolve()
                             and other.endswith((".safetensors", ".bin", ".pt"))
                             for other in model_files)})
if len(model_dirs) != 1 or not (model_dirs[0] / "sft_checkpoint.json").is_file():
    raise RuntimeError(f"expected one attributed SFT checkpoint model, got {model_dirs}")

accepted_art, accepted_root, accepted_files, accepted_manifest, accepted_source = download(
    os.environ["SFT_ACCEPTED_ARTIFACT"], "accepted", profile)
accepted_hits = [str((accepted_root / name).resolve()) for name in accepted_files
                 if name.endswith("accepted.jsonl")]
if len(accepted_hits) != 1:
    raise RuntimeError(f"expected one frozen accepted.jsonl, got {accepted_hits}")

gate_tokens = os.environ["SFT_GATE_BUNDLE_ARTIFACTS"].split()
gate_refs = {}
for token in gate_tokens:
    raw_step, ref = token.split("=", 1)
    step = int(raw_step)
    if step in gate_refs:
        raise RuntimeError(f"duplicate gate step {step}")
    gate_refs[step] = ref
expected_steps = ((16, 32, 64, 126) if profile == "c7_reuse_ga8"
                  else (125, 250, 500, 1000))
if tuple(sorted(gate_refs)) != expected_steps:
    raise RuntimeError(
        f"gate artifacts must cover {expected_steps}, got {sorted(gate_refs)}")
gates = []
for step in expected_steps:
    artifact, root, files, download_manifest, source = download(
        gate_refs[step], "sft-gate", str(step))
    required = {}
    for filename in ("aggregate_metrics.json", "rollouts.jsonl",
                     "heldout_selection.jsonl"):
        hits = [name for name in files if name.endswith("/" + filename) or name == filename]
        if len(hits) != 1:
            raise RuntimeError(f"step {step}: expected one {filename}, got {hits}")
        required[filename] = (root / hits[0]).resolve()
    parents = {path.parent for path in required.values()}
    if len(parents) != 1:
        raise RuntimeError(f"step {step}: gate files do not share one bundle directory")
    gates.append({
        "optimizer_step": step,
        "source": source,
        "download_manifest": str(download_manifest),
        "bundle_dir": str(next(iter(parents))),
    })

selection_art, selection_root, selection_files, selection_download, selection_source = download(
    os.environ["SFT_SELECTION_MANIFEST_ARTIFACT"], "sft-selection", "frozen")
selection_hits = [str((selection_root / name).resolve()) for name in selection_files
                  if name.endswith("selection_manifest.json")]
if len(selection_hits) != 1:
    raise RuntimeError(f"expected one selection_manifest.json, got {selection_hits}")

index = {
    "version": ("external_sft_evidence_bundle_v2"
                if profile == "c7_reuse_ga8"
                else "fresh_c8_sft_evidence_bundle_v1"),
    "model": {"source": model_source, "download_manifest": str(model_manifest),
              "model_dir": str(model_dirs[0])},
    "accepted": {"source": accepted_source,
                 "download_manifest": str(accepted_manifest),
                 "accepted_path": accepted_hits[0]},
    "gates": gates,
    "selection": {"source": selection_source,
                  "download_manifest": str(selection_download),
                  "manifest_path": selection_hits[0]},
}
if profile == "fresh_c8_single_scout":
    code_manifest_path = pathlib.Path(os.environ.get("CODE_SOURCE_MANIFEST", ""))
    if not code_manifest_path.is_file():
        raise RuntimeError("fresh-c8 continuation requires code download manifest")
    code_manifest = json.loads(code_manifest_path.read_text())
    index["code"] = {
        "source": code_manifest["source"],
        "download_manifest": str(code_manifest_path.resolve()),
    }
index_path = pathlib.Path("/workspace/external_sft_evidence_index.json")
index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
pathlib.Path("/workspace/sft_model_path.txt").write_text(str(model_dirs[0]))
print(f"[bootstrap] complete four-gate evidence index -> {index_path}", flush=True)
PY
  SFT_MODEL="$(cat /workspace/sft_model_path.txt)"
  SFT_EVIDENCE_INDEX=/workspace/external_sft_evidence_index.json
  test -f "$SFT_MODEL/config.json" || {
    echo "[bootstrap] FATAL: SFT_MODEL is not a Hugging Face model directory: $SFT_MODEL"
    exit 1
  }
  if [ "${SKIP_SCOUT:-0}" != "1" ]; then
    echo "[bootstrap] FATAL: SFT_MODEL requires SKIP_SCOUT=1 and the original trajectories"
    exit 1
  fi
  echo "[bootstrap] selected evidence-bound SFT model=$SFT_MODEL index=$SFT_EVIDENCE_INDEX"
fi

# ---- GPU 0 trains; GPUs 1..N-1 serve vLLM (pool is managed by pipeline_gpu) ---
if [ "$NUM_GPUS" -ge 2 ]; then
  VLLM_GPUS="$(seq 1 $((NUM_GPUS - 1)) | paste -sd, -)"
else
  echo "[bootstrap] WARNING: single GPU — colocating trainer and vLLM on GPU 0"
  VLLM_GPUS="0"
fi
echo "[bootstrap] train gpu=0 vllm gpus=$VLLM_GPUS"

SKIP_SCOUT_FLAG=""
if [ "${SKIP_SCOUT:-0}" = "1" ]; then
  SKIP_SCOUT_FLAG="--skip-scout"
fi
# Lambda flags: forward only what the caller set (see LAMBDA_EXPLICIT above).
# Single-round runs always get --lambda-novelty (the schedule is per-round).
LAMBDA_ARGS=()
if [ "$ROUNDS" -gt 1 ]; then
  if [ -n "$LAMBDA_EXPLICIT" ] && [ -n "$LAMBDA_SCHEDULE_EXPLICIT" ]; then
    LAMBDA_ARGS=(--lambda-novelty "$LAMBDA" --lambda-schedule "$LAMBDA_SCHEDULE")  # pipeline errors if inconsistent
  elif [ -n "$LAMBDA_EXPLICIT" ]; then
    LAMBDA_ARGS=(--lambda-novelty "$LAMBDA")
  else
    LAMBDA_ARGS=(--lambda-schedule "$LAMBDA_SCHEDULE")
  fi
else
  LAMBDA_ARGS=(--lambda-novelty "$LAMBDA")
fi
PPO_FLAG=""
if [ "$PPO_MODE" = "1" ]; then
  PPO_FLAG="--ppo-mode"
fi
RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
  RESUME_FLAG="--resume"   # no-op on a fresh workspace (no pipeline_state.json)
fi
STOP_AFTER_SFT_FLAG=""
if [ "$STOP_AFTER_SFT" = "1" ]; then
  STOP_AFTER_SFT_FLAG="--stop-after-sft"
fi
C8_SOURCE_BALANCE_FLAG=""
if [ "$C8_SOURCE_BALANCE" = "1" ]; then
  C8_SOURCE_BALANCE_FLAG="--c8-source-balance"
fi
MASK_TRUNCATED_FLAG="--no-mask-truncated"
if [ "$MASK_TRUNCATED" = "1" ]; then
  MASK_TRUNCATED_FLAG="--mask-truncated"
fi
SFT_MODEL_ARGS=()
if [ -n "$SFT_MODEL" ]; then
  SFT_MODEL_ARGS=(
    --sft-model "$SFT_MODEL" --sft-evidence-index "$SFT_EVIDENCE_INDEX"
  )
fi
FROZEN_SFT_LIBRARY_ARGS=()
if [ -n "$FROZEN_SFT_LIBRARY_EVIDENCE_INDEX" ]; then
  FROZEN_SFT_LIBRARY_ARGS=(
    --frozen-sft-library-evidence-index "$FROZEN_SFT_LIBRARY_EVIDENCE_INDEX"
  )
fi
FRESH_C8_SFT_GATE_ARGS=()
if [ "$FRESH_C8_SFT_GATE" = "1" ]; then
  FRESH_C8_SFT_GATE_ARGS=(
    --fresh-c8-sft-gate --code-source-manifest "$CODE_SOURCE_MANIFEST"
  )
fi
FRESH_C8_MULTISCOUT_COMPONENT_ARGS=()
if [ "$FRESH_C8_MULTISCOUT_COMPONENT" = "1" ]; then
  FRESH_C8_MULTISCOUT_COMPONENT_ARGS=(
    --fresh-c8-multiscout-component
    --scout-only
    --seed "$SEED"
    --code-source-manifest "$CODE_SOURCE_MANIFEST"
  )
fi
TRAJECTORY_SOURCE_ARGS=()
if [ -n "$TRAJECTORY_SOURCE_MANIFEST" ]; then
  TRAJECTORY_SOURCE_ARGS=(--trajectory-source-manifest "$TRAJECTORY_SOURCE_MANIFEST")
fi
STABILIZATION_ARGS=(
  --expdis-stabilization-mode "$EXPDIS_STABILIZATION_MODE"
  --novelty-normalization "$NOVELTY_NORMALIZATION"
  --c9-execution-profile "$C9_EXECUTION_PROFILE"
  --c10-execution-profile "$C10_EXECUTION_PROFILE"
  --c9-multiexplorer-profile "$C9_MULTIEXPLORER_PROFILE"
  --single-scout-multiround-profile "$SINGLE_SCOUT_MULTIROUND_PROFILE"
  --stabilization-cell "$STABILIZATION_CELL"
  --scientific-variant "$SCIENTIFIC_VARIANT"
)
if [ "$EXPDIS_STABILIZATION_MODE" = "c9_selected_batch_update" ] || \
   [ -n "$C9_EXECUTION_PROFILE" ]; then
  STABILIZATION_ARGS+=(--code-source-manifest "$CODE_SOURCE_MANIFEST")
fi
LR_ARGS=()
if [ -n "$SCOUT_LR" ]; then
  LR_ARGS+=(--scout-lr "$SCOUT_LR")
fi
if [ -n "$CENTRAL_LR" ]; then
  LR_ARGS+=(--central-lr "$CENTRAL_LR")
fi
SCOUT_ONLY_FLAG=""
if [ "$SCOUT_ONLY" = "1" ]; then
  SCOUT_ONLY_FLAG="--scout-only"
fi

# C10 is one preregistered stochastic experiment, not a best-of-retries
# search.  A fixed run identity plus a remote terminal registry prevents a
# fresh OUT_DIR/RUN_NAME from replacing an observed pass or semantic failure.
# Transport-invalid outcomes remain explicitly rerunnable unchanged.
C10_LEASE_ARGS=()
if [ "$C10_EXECUTION_PROFILE" = "canary" ]; then
  python3 - <<'PY'
import json
import os
import pathlib
import re
import shutil
import wandb
from wandb.errors import CommError
from tmx_gpu import c10_evidence, common, orchestrator_receipt

# Refuse every env-derived contract mismatch before the append-only lease
# claim.  Once a claim is committed, ordinary configuration errors cannot be
# permitted to consume the preregistered one-shot stochastic experiment.
c10_evidence.validate_prelease_environment(os.environ)

entity = os.environ.get("WANDB_ENTITY", "")
if not entity:
    raise SystemExit("WANDB_ENTITY must name the W&B entity that owns the artifacts")
project = os.environ.get("WANDB_PROJECT", "two-model-explore-gpu")
name = "tmx-c10-survivor-canary-terminal-0e3448b184c5"
api = wandb.Api()
path = f"{entity}/{project}/{name}:latest"
results_path = (
    f"{entity}/{project}/"
    "tmx-c10-survivor-canary-0e3448b184c5-results:latest")
try:
    terminal_results = api.artifact(results_path, type="expdis-results")
except CommError as exc:
    if "not found" not in str(exc).lower():
        raise
    terminal_results = None
if terminal_results is not None:
    results_outcome = str((terminal_results.metadata or {}).get(
        "c10_outcome", ""))
    if results_outcome in {"passed", "semantic_failed"}:
        raise SystemExit(
            "C10 canonical results already contain a terminal outcome; "
            f"replacement is forbidden: {terminal_results.qualified_name} "
            f"{results_outcome}")
try:
    terminal_artifact = api.artifact(path, type="c10-terminal-outcome")
except CommError as exc:
    if "not found" not in str(exc).lower():
        raise
    terminal_artifact = None
if terminal_artifact is not None:
    outcome = str((terminal_artifact.metadata or {}).get(
        "c10_outcome", ""))
    if outcome not in {"passed", "semantic_failed"}:
        raise SystemExit(
            "C10 terminal registry exists with malformed outcome: "
            f"{terminal_artifact.qualified_name} {outcome!r}")
    raise SystemExit(
        "C10 already has a terminal scientific outcome; replacement is "
        f"forbidden: {terminal_artifact.qualified_name} {outcome}")

# Append-only exclusive lease.  W&B assigns a total vN order to concurrent
# claims; among currently-active holders, the earliest version wins.  A late
# claimant therefore observes and rejects the earlier holder even if both saw
# an initially empty collection.  The holder is the immutable Lepton receipt,
# not a caller-controlled process ID.
binding_path = pathlib.Path(os.environ["ORCHESTRATOR_RECEIPT_BINDING"])
receipt_binding = json.loads(binding_path.read_text(encoding="utf-8"))
orchestrator_receipt.validate_binding(
    receipt_binding, expected_run_name=c10_evidence.C10_CANONICAL_RUN_NAME)
receipt_source = receipt_binding["source"]
holder = {
    "orchestrator_binding_sha256": receipt_binding["binding_sha256"],
    "receipt_artifact_ref": receipt_source["artifact_ref"],
    "receipt_artifact_digest": receipt_source["artifact_digest"],
}
holder_sha = holder["orchestrator_binding_sha256"]
lease_name = c10_evidence.C10_EXCLUSIVE_LEASE_ARTIFACT
lease_collection = f"{entity}/{project}/{lease_name}"
claim_filename = "c10_exclusive_lease_claim.json"
claim_root = pathlib.Path(os.environ["OUT_DIR"]) / "c10-lease-claims"
claim_root.mkdir(parents=True, exist_ok=True)

def active_claim_payload():
    return c10_evidence.signed_lease_claim({
        "run_name": c10_evidence.C10_CANONICAL_RUN_NAME,
        "preregistration_sha256":
            c10_evidence.C10_PREREGISTRATION_SHA256,
        "lease_state": "active",
        "holder": holder,
    })

def publish_claim(claim):
    c10_evidence.validate_lease_claim(claim)
    state = claim["lease_state"]
    claim_holder_sha = claim["holder"]["orchestrator_binding_sha256"]
    local = claim_root / \
        f"{state}-{claim_holder_sha}-{claim['evidence_sha256']}.json"
    common.atomic_write_json(str(local), claim)
    lease_run = wandb.init(
        project=project, entity=entity,
        name=f"c10-lease-{claim_holder_sha[:12]}",
        id=re.sub(r"[^a-zA-Z0-9_-]", "-",
                  f"c10-lease-{claim_holder_sha[:40]}")[:120],
        resume="allow", job_type="c10-exclusive-lease")
    artifact = wandb.Artifact(
        lease_name, type="c10-exclusive-lease",
        metadata=c10_evidence.lease_claim_metadata(claim))
    artifact.add_file(str(local), name=claim_filename)
    logged = lease_run.log_artifact(artifact, aliases=["latest"])
    logged.wait()
    lease_run.finish()
    return claim, logged

def version_number(artifact):
    match = re.fullmatch(r"v([0-9]+)", str(artifact.version))
    if match is None:
        raise RuntimeError(
            f"C10 lease artifact has invalid version: {artifact.version!r}")
    return int(match.group(1))

def load_record(artifact):
    members = sorted(item.name for item in artifact.files())
    if members != [claim_filename]:
        raise RuntimeError(
            f"C10 lease artifact membership mismatch: {members}")
    root = claim_root / re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
    if root.exists():
        shutil.rmtree(root)
    artifact.download(root=str(root))
    claim = json.loads((root / claim_filename).read_text(encoding="utf-8"))
    c10_evidence.validate_lease_claim(claim)
    if dict(artifact.metadata or {}) != \
            c10_evidence.lease_claim_metadata(claim):
        raise RuntimeError("C10 lease artifact metadata/claim mismatch")
    return {
        "artifact_ref": artifact.qualified_name,
        "artifact_digest": artifact.digest,
        "claim": claim,
    }

def authorize_transport_release(claim):
    authorization = claim["release"]["authorization"]
    artifact = api.artifact(
        authorization["result_artifact_ref"], type="expdis-results")
    if artifact.digest != authorization["result_artifact_digest"]:
        raise RuntimeError("C10 transport release result digest drifted")
    root = claim_root / re.sub(
        r"[^a-zA-Z0-9_.-]", "-", f"result-{artifact.digest}")
    if root.exists():
        shutil.rmtree(root)
    artifact.download(root=str(root))
    c10_evidence.validate_transport_release_result(
        claim,
        result_artifact_ref=artifact.qualified_name,
        result_artifact_digest=artifact.digest,
        result_metadata=dict(artifact.metadata or {}),
        result_root=root,
    )

def lease_versions():
    try:
        api.artifact(
            f"{lease_collection}:latest", type="c10-exclusive-lease")
    except CommError as exc:
        if "not found" in str(exc).lower():
            return []
        raise
    return sorted(
        list(api.artifacts(
            "c10-exclusive-lease", lease_collection, per_page=100)),
        key=version_number)

def current_epoch_winner(versions):
    # Shared replay verifies exact active-claim targeting.  In particular, a
    # delayed release from an older same-holder attempt cannot close a newer
    # epoch, and every transport release replays its durable >5% proof.
    artifacts_by_ref = {}
    records = []
    for artifact in versions:
        record = load_record(artifact)
        artifacts_by_ref[record["artifact_ref"]] = artifact
        records.append(record)
    winner = c10_evidence.replay_exclusive_lease_records(
        records, authorize_transport_release=authorize_transport_release)
    if winner is None:
        return None, None
    artifact = artifacts_by_ref[winner["artifact_ref"]]
    return winner["claim"]["holder"][
        "orchestrator_binding_sha256"], (winner["claim"], artifact)

def recover_durable_transport_release(winner):
    """Close a crash window after result.wait() but before release.wait()."""
    if winner is None:
        return None
    winning_claim, winning_artifact = winner
    active_binding = c10_evidence.build_exclusive_lease_binding(
        claim=winning_claim,
        artifact_ref=winning_artifact.qualified_name,
        artifact_digest=winning_artifact.digest)
    result_collection = \
        f"{entity}/{project}/{c10_evidence.C10_RESULTS_ARTIFACT}"
    try:
        api.artifact(
            f"{result_collection}:latest", type="expdis-results")
    except CommError as exc:
        if "not found" in str(exc).lower():
            return None
        raise
    results = sorted(
        list(api.artifacts(
            "expdis-results", result_collection, per_page=100)),
        key=version_number)
    for result in results:
        metadata = dict(result.metadata or {})
        if metadata.get("c10_outcome") != "transport_invalid":
            continue
        scope = str(metadata.get("c10_transport_scope", ""))
        evidence_key = (
            "c10_training_transport_evidence_sha256"
            if scope == "training" else
            "c10_canary_behavior_gate_evidence_sha256")
        evidence_sha = metadata.get(evidence_key)
        if scope not in {"training", "behavior_gate"} or not isinstance(
                evidence_sha, str) or len(evidence_sha) != 64:
            raise RuntimeError(
                "C10 durable transport result has malformed metadata")
        release = c10_evidence.build_transport_release_claim(
            active_binding=active_binding,
            result_artifact_ref=result.qualified_name,
            result_artifact_digest=result.digest,
            transport_scope=scope,
            transport_evidence_sha256=evidence_sha)
        root = claim_root / re.sub(
            r"[^a-zA-Z0-9_.-]", "-", f"recovery-{result.digest}")
        if root.exists():
            shutil.rmtree(root)
        result.download(root=str(root))
        try:
            c10_evidence.validate_transport_release_result(
                release,
                result_artifact_ref=result.qualified_name,
                result_artifact_digest=result.digest,
                result_metadata=metadata,
                result_root=root)
        except ValueError as exc:
            if "different lease epoch" in str(exc):
                continue
            raise
        publish_claim(release)
        return result.qualified_name
    return None

winner_holder, winner = current_epoch_winner(lease_versions())
recovered_result = recover_durable_transport_release(winner)
if recovered_result is not None:
    # Never train in the same bootstrap that repaired a missing release.  A
    # fresh immutable receipt/output directory must own the unchanged rerun.
    raise SystemExit(
        "C10 recovered a durable transport-invalid result and released its "
        f"exact lease epoch; submit a fresh unchanged canary: {recovered_result}")
if winner is not None:
    raise SystemExit(
        "C10 already has a pre-existing active lease; re-bootstrap and "
        "same-holder resume are forbidden because state provenance cannot "
        "prove the original sampler is dead: "
        f"{winner[1].qualified_name} holder={winner_holder}")
published_claim, published_artifact = publish_claim(active_claim_payload())
api = wandb.Api()
winner_holder, winner = current_epoch_winner(lease_versions())
if winner is None:
    raise SystemExit("C10 active lease claim disappeared after commit")
if winner_holder != holder_sha:
    # A losing active version is permanently ignored by epoch replay.  It is
    # intentionally not allowed to publish a generic release record.
    raise SystemExit(
        "C10 concurrent lease claim lost to earlier immutable holder: "
        f"{winner[1].qualified_name}")
winning_claim = winner[0]
winning_artifact = winner[1]
try:
    c10_evidence.validate_published_lease_winner(
        {
            "artifact_ref": winning_artifact.qualified_name,
            "artifact_digest": winning_artifact.digest,
            "claim": winning_claim,
        },
        published_claim=published_claim,
        published_artifact_ref=published_artifact.qualified_name,
        published_artifact_digest=published_artifact.digest)
except ValueError as exc:
    raise SystemExit(
        "C10 concurrent same-holder lease claim lost to an earlier exact "
        f"artifact version: {winning_artifact.qualified_name}") from exc
lease_binding = c10_evidence.build_exclusive_lease_binding(
    claim=winning_claim,
    artifact_ref=winning_artifact.qualified_name,
    artifact_digest=winning_artifact.digest)
lease_binding_path = pathlib.Path(os.environ["OUT_DIR"]) / \
    "c10_exclusive_lease_binding.json"
common.atomic_write_json(str(lease_binding_path), lease_binding)
print(
    "[bootstrap] C10 terminal registry clear; exclusive lease="
    f"{winning_artifact.qualified_name} holder={holder_sha}")
PY
  C10_EXCLUSIVE_LEASE_BINDING="$OUT_DIR/c10_exclusive_lease_binding.json"
  export C10_EXCLUSIVE_LEASE_BINDING
  C10_LEASE_ARGS=(
    --c10-exclusive-lease-binding "$C10_EXCLUSIVE_LEASE_BINDING"
  )
fi

STATUS=0
# Keep every mutable runtime file outside the downloaded code artifact.  The
# fresh-c8 validator intentionally re-hashes that artifact before update 1 and
# on resume; running from the artifact root let W&B create `wandb/latest-run`
# there and correctly tripped the immutable-code guard.
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORK"
# shellcheck disable=SC2086
python3 "$CODE_ROOT/tmx_gpu/pipeline_gpu.py" \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --container-image-identity "$TRAINING_CONTAINER_IMAGE_IDENTITY" \
  "${ORCHESTRATOR_ARGS[@]}" \
  --run-name "$RUN_NAME" \
  --output-dir "$OUT_DIR" \
  --vllm-gpus "$VLLM_GPUS" \
  --vllm-max-model-len "$MAX_MODEL_LEN" \
  --rounds "$ROUNDS" \
  --scouts-per-round "$SCOUTS_PER_ROUND" \
  --scout-steps "$SCOUT_STEPS" \
  --central-steps "$CENTRAL_STEPS" \
  "${STABILIZATION_ARGS[@]}" \
  "${C10_LEASE_ARGS[@]}" \
  "${LR_ARGS[@]}" \
  "${LAMBDA_ARGS[@]}" \
  --policy "$POLICY" \
  $C8_SOURCE_BALANCE_FLAG \
  --min-accepted "$MIN_ACCEPTED" \
  --max-accepted "$MAX_ACCEPTED" \
  --harvest-max-passes "$HARVEST_MAX_PASSES" \
  --reload-every "$RELOAD_EVERY" \
  --rl-validity-mode "$RL_VALIDITY_MODE" \
  --truncation-baseline-mode "$TRUNCATION_BASELINE_MODE" \
  --save-every "$SAVE_EVERY" \
  --keep-checkpoints "$KEEP_CHECKPOINTS" \
  --ckpt-artifact-every "$CKPT_ARTIFACT_EVERY" \
  --probe-every "$PROBE_EVERY" \
  --dynamic-max-attempts "$DYNAMIC_MAX_ATTEMPTS" \
  --prompts-per-step "$PROMPTS_PER_STEP" \
  --num-generations "$NUM_GENERATIONS" \
  --dataset "$DATASET" \
  --max-train-examples "$MAX_TRAIN_EXAMPLES" \
  --sync-dir "$WORK" \
  --seed "$SEED" \
  --completion-budget "$COMPLETION_BUDGET" \
  --updates-per-rollout "$UPDATES_PER_ROLLOUT" \
  --optimizer "$OPTIMIZER" \
  --sft-optimizer "$SFT_OPTIMIZER" \
  --sft-grad-accum "$SFT_GRAD_ACCUM" \
  --sft-termination-mode "$SFT_TERMINATION_MODE" \
  --sft-save-steps "$SFT_SAVE_STEPS" \
  "${SFT_MODEL_ARGS[@]}" \
  "${FROZEN_SFT_LIBRARY_ARGS[@]}" \
  "${TRAJECTORY_SOURCE_ARGS[@]}" \
  "${MRME_CODE_ARGS[@]}" \
  $MASK_TRUNCATED_FLAG \
  $PPO_FLAG \
  $STOP_AFTER_SFT_FLAG \
  $SCOUT_ONLY_FLAG \
  $SKIP_SCOUT_FLAG \
  $RESUME_FLAG \
  ${TRAJECTORIES:+--trajectories $TRAJECTORIES} \
  "${EXTRA_ARGV[@]}" \
  "${FRESH_C8_SFT_GATE_ARGS[@]}" \
  "${FRESH_C8_MULTISCOUT_COMPONENT_ARGS[@]}" \
  --wandb || STATUS=$?

# ---- push result JSONs + final model to W&B (durable state) -------------------
# The pipeline already pushes a versioned results artifact after every stage;
# this final push adds the last state plus the final model directory.
export PIPELINE_EXIT_STATUS="$STATUS"
python3 - <<'PY' || STATUS=$((STATUS == 0 ? 1 : STATUS))
import glob, json, os, re, wandb
from tmx_gpu import common

run_name = os.environ["RUN_NAME"]
out_dir = os.environ.get("OUT_DIR", "/workspace/runs/" + run_name)
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "two-model-explore-gpu"),
    entity=os.environ.get("WANDB_ENTITY") or None,
    name=f"{run_name}-artifacts",
    group=run_name,
    id=re.sub(r"[^a-zA-Z0-9_-]", "-", f"{run_name}-artifacts")[:120],
    resume="allow",
    job_type="artifacts",
)
artifact_metadata = {}
single_scout_multiround = bool(
    os.environ.get("SINGLE_SCOUT_MULTIROUND_PROFILE", ""))
true_mrme = (
    int(os.environ.get("SCOUTS_PER_ROUND", "1")) > 1
    or single_scout_multiround)
if true_mrme:
    rounds = int(os.environ.get("ROUNDS", "1"))
    artifact_metadata = {
        "method": (
            "disjoint_shard_annealed_single_scout_multiround"
            if single_scout_multiround else
            "integrated_multiscout" if rounds == 1 else
            "disjoint_shard_annealed_mrme"),
        "method_is_failure_targeted_curriculum": False,
        "stabilization_profile": (
            "c9_selected_batch_multi_explorer"
            if single_scout_multiround else
            os.environ.get("C9_MULTIEXPLORER_PROFILE") or
            "c8_candidate_attempt_multi_explorer"),
        "rounds": rounds,
        "scouts_per_round": int(os.environ["SCOUTS_PER_ROUND"]),
        "base_seed": int(os.environ.get("SEED", "0")),
    }
    if single_scout_multiround:
        artifact_metadata.update({
            "method_version":
                "gpu_disjoint_shard_annealed_single_scout_multiround_v1",
            "mrme": False,
            "multi_explorer": False,
            "single_scout_multiround": True,
        })
elif os.environ.get("FRESH_C8_MULTISCOUT_COMPONENT", "0") == "1":
    artifact_metadata = {
        "fresh_c8_evidence_profile": "fresh_c8_multiscout_component",
        "component_seed": int(os.environ["SEED"]),
    }
elif os.environ.get("FRESH_C8_SFT_GATE", "0") == "1":
    artifact_metadata = {
        "fresh_c8_evidence_profile": "fresh_c8_single_scout",
        "seed": int(os.environ.get("SEED", "0")),
    }
elif (os.environ.get("C9_EXECUTION_PROFILE") or
      os.environ.get("EXPDIS_STABILIZATION_MODE") == "c9_selected_batch_update"):
    artifact_metadata = {
        "expdis_stabilization_mode": os.environ["EXPDIS_STABILIZATION_MODE"],
        "c9_execution_profile": os.environ.get("C9_EXECUTION_PROFILE", ""),
        "c10_execution_profile": os.environ.get("C10_EXECUTION_PROFILE", ""),
        "stabilization_cell": os.environ.get("STABILIZATION_CELL", ""),
        "scientific_variant": os.environ["SCIENTIFIC_VARIANT"],
        "novelty_normalization": os.environ["NOVELTY_NORMALIZATION"],
        "scout_lr": (float(os.environ["SCOUT_LR"])
                     if os.environ.get("SCOUT_LR") else 5e-6),
        "truncation_baseline_mode": os.environ.get(
            "TRUNCATION_BASELINE_MODE", "all_rows"),
        "mask_truncated": os.environ.get("MASK_TRUNCATED", "0") == "1",
    }

# A successful C9 full artifact is a scientific claim, not just a directory.
# Validate the complete signed handoff chain before attaching claim metadata or
# allowing a final model upload.  Failed retries may still upload diagnostic
# results, but can never publish an old run_summary's stale model.
pipeline_status = int(os.environ.get("PIPELINE_EXIT_STATUS", "1"))
c9_publishable_final = None
c10_publishable_final = None
mrme_publishable_final = None
if true_mrme and pipeline_status == 0:
    from tmx_gpu import common, mrme_contract

    completed = mrme_contract.validate_completed_campaign(out_dir)
    mrme_publishable_final = completed["final_model"]
    common.atomic_write_json(
        os.path.join(out_dir, "mrme_completed_campaign.json"), completed)
    portable = mrme_contract.build_portable_completed_campaign(out_dir)
    common.atomic_write_json(
        os.path.join(out_dir, "mrme_portable_campaign.json"), portable)
    artifact_metadata.update({
        "method": completed["method"],
        "stabilization_profile": completed["stabilization_profile"],
        "scientific_variant": completed["scientific_variant"],
        "external_p1_required": True,
        "headline_eligible": False,
        "mrme_contract_sha256": completed["contract_sha256"],
        "mrme_partition_sha256": completed["partition_sha256"],
        "mrme_completion_evidence_sha256": completed["evidence_sha256"],
        "mrme_portable_campaign_evidence_sha256": portable[
            "evidence_sha256"],
        "mrme_final_model_identity_sha256": completed[
            "final_model_identity_sha256"],
    })
    if completed.get("single_scout_multiround") is True:
        artifact_metadata.update({
            "method_version": completed["method_version"],
            "mrme": False,
            "multi_explorer": False,
            "single_scout_multiround": True,
        })
elif (os.environ.get("C9_EXECUTION_PROFILE") == "full"
        and os.environ.get("STABILIZATION_CELL") == "C5"
        and pipeline_status == 0):
    from tmx_gpu import c9_evidence, common

    state_path = os.path.join(out_dir, "pipeline_state.json")
    summary_path = os.path.join(out_dir, "run_summary.json")
    lineage_path = os.path.join(out_dir, "round_01", "c9_central_lineage.json")
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(lineage_path, encoding="utf-8") as handle:
        lineage = json.load(handle)
    required_stages = {
        "r01/scout", "r01/harvest", "r01/filter",
        "r01/scout_behavior_gate", "r01/sft", "r01/sft_behavior_gate",
        "r01/central",
    }
    missing = sorted(required_stages - set(state.get("stages", {})))
    if missing:
        raise RuntimeError(f"C9 final publication lacks completed stages: {missing}")
    execution = state.get("c9_execution_contract", {})
    unsigned_execution = dict(execution)
    execution_sha = unsigned_execution.pop("contract_sha256", None)
    if (execution.get("version") != c9_evidence.PIPELINE_EXECUTION_CONTRACT_VERSION
            or execution_sha != common.canonical_json_sha256(unsigned_execution)):
        raise RuntimeError("C9 final publication has invalid execution contract")
    c9_evidence.validate_central_lineage(lineage)
    central_stage = state["stages"]["r01/central"]
    central_probes = central_stage.get("c9_probe_history", {})
    if (central_probes.get("probe_protocol_id")
            != c9_evidence.C9_PROBE_PROTOCOL_ID
            or central_probes.get("phase") != "central_grpo"
            or central_probes.get("expected_global_steps")
            != [225, 250, 275, 300]
            or central_probes.get("row_count") != 4
            or not isinstance(
                central_probes.get("canonical_rows_sha256"), str)
            or len(central_probes["canonical_rows_sha256"]) != 64):
        raise RuntimeError("C9 final publication lacks exact Central probes")
    c9_publishable_final = str(summary.get("final_model", ""))
    if (not c9_publishable_final
            or os.path.realpath(c9_publishable_final)
            != os.path.realpath(str(central_stage.get("final", "")))):
        raise RuntimeError("C9 run_summary final model differs from Central stage")
    summary_args = summary.get("args", {})
    if (summary_args.get("c9_execution_profile") != "full"
            or summary_args.get("stabilization_cell") != "C5"
            or summary_args.get("scientific_variant")
            != c9_evidence.C9_SCIENTIFIC_VARIANT):
        raise RuntimeError("C9 run_summary identity mismatch")
    trainer_path = os.path.join(c9_publishable_final, "trainer_state.json")
    with open(trainer_path, encoding="utf-8") as handle:
        trainer = json.load(handle)
    if (trainer.get("phase") != "central" or trainer.get("step") != 100
            or trainer.get("external_lineage") != lineage
            or trainer.get("external_lineage_sha256") != lineage["evidence_sha256"]
            or float(trainer.get("lambda", -1.0)) != 0.0
            or float(trainer.get("runtime_contract", {}).get(
                "lambda_novelty", -1.0)) != 0.0
            or os.path.exists(os.path.join(c9_publishable_final, "rnd_state.pt"))):
        raise RuntimeError("C9 final Central trainer_state/novelty contract mismatch")
    common.verify_directory_file_manifest(
        c9_publishable_final, trainer["checkpoint_file_manifest"],
        exclude_relative_paths=("trainer_state.json",))
    artifact_metadata.update({
        "c9_pipeline_execution_contract_sha256": execution["contract_sha256"],
        "c9_central_lineage_sha256": lineage["evidence_sha256"],
        "c9_scout_completion_evidence_sha256": lineage[
            "sft_training_evidence"]["scout_completion_evidence"][
                "evidence_sha256"],
        "c9_scout_behavior_gate_evidence_sha256": lineage[
            "sft_training_evidence"]["scout_behavior_gate"]["evidence_sha256"],
        "c9_sft_training_evidence_sha256": lineage[
            "sft_training_evidence"]["evidence_sha256"],
        "c9_sft_behavior_gate_evidence_sha256": lineage[
            "sft_behavior_gate"]["evidence_sha256"],
    })
elif (os.environ.get("C9_EXECUTION_PROFILE") == "canary"
      and pipeline_status == 0):
    from tmx_gpu import c9_evidence, common

    with open(os.path.join(out_dir, "pipeline_state.json"),
              encoding="utf-8") as handle:
        state = json.load(handle)
    with open(os.path.join(out_dir, "run_summary.json"),
              encoding="utf-8") as handle:
        summary = json.load(handle)
    required_stages = {"r01/scout", "r01/canary_behavior_gate"}
    missing = sorted(required_stages - set(state.get("stages", {})))
    if missing:
        raise RuntimeError(
            f"C9 canary publication lacks completed stages: {missing}")
    execution = state.get("c9_execution_contract", {})
    unsigned_execution = dict(execution)
    execution_sha = unsigned_execution.pop("contract_sha256", None)
    if (execution.get("version")
            != c9_evidence.PIPELINE_EXECUTION_CONTRACT_VERSION
            or execution_sha != common.canonical_json_sha256(
                unsigned_execution)):
        raise RuntimeError("C9 canary publication has invalid execution contract")
    scout_stage = state["stages"]["r01/scout"]
    gate_stage = state["stages"]["r01/canary_behavior_gate"]
    with open(str(gate_stage.get("evidence_path", "")),
              encoding="utf-8") as handle:
        gate = json.load(handle)
    c9_evidence.validate_behavior_gate_evidence(
        gate, expected_stage="scout_step50", expected_profile="canary",
        expected_cell=os.environ["STABILIZATION_CELL"])
    if gate_stage.get("evidence_sha256") != gate["evidence_sha256"]:
        raise RuntimeError("C9 canary pipeline-state gate SHA mismatch")
    c9_publishable_final = str(summary.get("final_model", ""))
    if (not c9_publishable_final
            or os.path.realpath(c9_publishable_final)
            != os.path.realpath(str(scout_stage.get("final", "")))):
        raise RuntimeError("C9 canary final model differs from Scout stage")
    with open(os.path.join(c9_publishable_final, "trainer_state.json"),
              encoding="utf-8") as handle:
        trainer = json.load(handle)
    if (trainer.get("phase") != "scout" or trainer.get("step") != 50
            or trainer.get("runtime_contract")
            != gate.get("model_runtime_contract")):
        raise RuntimeError("C9 canary trainer state/runtime mismatch")
    common.verify_directory_file_manifest(
        c9_publishable_final, trainer["checkpoint_file_manifest"],
        exclude_relative_paths=("trainer_state.json",))
    artifact_metadata.update({
        "c9_pipeline_execution_contract_sha256": execution["contract_sha256"],
        "c9_canary_behavior_gate_evidence_sha256": gate["evidence_sha256"],
    })
elif (os.environ.get("C10_EXECUTION_PROFILE") == "canary"
      and pipeline_status == 0):
    from tmx_gpu import c10_evidence

    validated = c10_evidence.validate_successful_canary_publication(out_dir)
    execution = validated["execution_contract"]
    gate = validated["behavior_gate"]
    endpoint = validated["endpoint_replay"]
    c10_publishable_final = validated["final_model"]
    artifact_metadata.update({
        "c10_outcome": "passed",
        "c10_pipeline_execution_contract_sha256":
            execution["contract_sha256"],
        "c10_endpoint_replay_evidence_sha256":
            endpoint["evidence_sha256"],
        "c10_canary_behavior_gate_evidence_sha256":
            gate["evidence_sha256"],
        "c10_preregistration_sha256":
            c10_evidence.C10_PREREGISTRATION_SHA256,
    })
elif os.environ.get("C10_EXECUTION_PROFILE") == "canary":
    from tmx_gpu import c10_evidence

    state_path = os.path.join(out_dir, "pipeline_state.json")
    terminal_gate_outcome = ""
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as handle:
            failed_state = json.load(handle)
        gate_stage = failed_state.get("stages", {}).get(
            "r01/c10_canary_behavior_gate")
        if isinstance(gate_stage, dict):
            terminal_gate_outcome = str(gate_stage.get("gate_outcome", ""))
    training_failure_path = os.path.join(
        out_dir, "round_01", "scout",
        "c10_training_terminal_outcome.json")
    training_transport_path = os.path.join(
        out_dir, "round_01", "scout",
        "c10_training_transport_invalid.json")
    behavior_transport_path = os.path.join(
        out_dir, "round_01", "c10_behavior_gates", "scout_step50",
        "evidence.json")
    if os.path.isfile(training_failure_path):
        validated = c10_evidence.validate_terminal_training_failure(out_dir)
        execution = validated["execution_contract"]
        failure = validated["training_failure"]
        artifact_metadata.update({
            "c10_outcome": "semantic_failed",
            "c10_terminal_scientific_failure": True,
            "c10_training_failure_kind": failure["failure_kind"],
            "c10_pipeline_execution_contract_sha256":
                execution["contract_sha256"],
            "c10_training_failure_evidence_sha256":
                failure["evidence_sha256"],
            "c10_preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
        })
    elif terminal_gate_outcome == "semantic_failed":
        validated = c10_evidence.validate_terminal_semantic_failure(out_dir)
        execution = validated["execution_contract"]
        gate = validated["behavior_gate"]
        endpoint = validated["endpoint_replay"]
        artifact_metadata.update({
            "c10_outcome": "semantic_failed",
            "c10_terminal_scientific_failure": True,
            "c10_pipeline_execution_contract_sha256":
                execution["contract_sha256"],
            "c10_endpoint_replay_evidence_sha256":
                endpoint["evidence_sha256"],
            "c10_canary_behavior_gate_evidence_sha256":
                gate["evidence_sha256"],
            "c10_preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
        })
    elif terminal_gate_outcome == "passed":
        # A passing fixed-panel draw is terminal even if a later summary or
        # artifact step failed.  Record the pass to prevent replacement, but
        # do not authorize model publication without the full success proof.
        validated = c10_evidence.validate_terminal_behavior_pass(out_dir)
        execution = validated["execution_contract"]
        gate = validated["behavior_gate"]
        endpoint = validated["endpoint_replay"]
        artifact_metadata.update({
            "c10_outcome": "passed",
            "c10_terminal_scientific_failure": False,
            "c10_model_publication_authorized": False,
            "c10_pipeline_execution_contract_sha256":
                execution["contract_sha256"],
            "c10_endpoint_replay_evidence_sha256":
                endpoint["evidence_sha256"],
            "c10_canary_behavior_gate_evidence_sha256":
                gate["evidence_sha256"],
            "c10_preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
        })
    elif terminal_gate_outcome:
        raise RuntimeError(
            "C10 staged behavior gate has invalid outcome: "
            f"{terminal_gate_outcome!r}")
    elif os.path.isfile(training_transport_path):
        validated = c10_evidence.validate_training_transport_bundle(out_dir)
        execution = validated["execution_contract"]
        transport = validated["training_transport"]
        artifact_metadata.update({
            "c10_outcome": "transport_invalid",
            "c10_terminal_scientific_failure": False,
            "c10_transport_scope": "training",
            "c10_pipeline_execution_contract_sha256":
                execution["contract_sha256"],
            "c10_training_transport_evidence_sha256":
                transport["evidence_sha256"],
            "c10_preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
        })
    elif os.path.isfile(behavior_transport_path):
        validated = c10_evidence.validate_unstaged_behavior_bundle(out_dir)
        execution = validated["execution_contract"]
        gate = validated["behavior_gate"]
        endpoint = validated["endpoint_replay"]
        raw_outcome = gate["gate_outcome"]
        if raw_outcome not in {
                "passed", "semantic_failed", "transport_invalid"}:
            raise RuntimeError(
                f"C10 unstaged gate has invalid outcome {raw_outcome!r}")
        artifact_metadata.update({
            "c10_outcome": raw_outcome,
            "c10_terminal_scientific_failure":
                raw_outcome == "semantic_failed",
            "c10_model_publication_authorized": False,
            "c10_transport_scope": (
                "behavior_gate" if raw_outcome == "transport_invalid"
                else None),
            "c10_pipeline_execution_contract_sha256":
                execution["contract_sha256"],
            "c10_endpoint_replay_evidence_sha256":
                endpoint["evidence_sha256"],
            "c10_canary_behavior_gate_evidence_sha256":
                gate["evidence_sha256"],
            "c10_preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
        })
    else:
        # Operationally incomplete attempts make no scientific claim and do
        # not release the exclusive lease automatically.
        artifact_metadata.update({
            "c10_outcome": "incomplete",
            "c10_terminal_scientific_failure": False,
            "c10_preregistration_sha256":
                c10_evidence.C10_PREREGISTRATION_SHA256,
        })
c10_terminal_registry_path = None
if artifact_metadata.get("c10_outcome") in {"passed", "semantic_failed"}:
    terminal_payload = {
        "version": "c10_terminal_outcome_v1",
        "run_name": run_name,
        "c10_outcome": artifact_metadata["c10_outcome"],
        "preregistration_sha256":
            artifact_metadata["c10_preregistration_sha256"],
        "execution_contract_sha256": artifact_metadata[
            "c10_pipeline_execution_contract_sha256"],
        "endpoint_replay_evidence_sha256": artifact_metadata.get(
            "c10_endpoint_replay_evidence_sha256"),
        "behavior_gate_evidence_sha256": artifact_metadata.get(
            "c10_canary_behavior_gate_evidence_sha256"),
        "training_failure_evidence_sha256": artifact_metadata.get(
            "c10_training_failure_evidence_sha256"),
    }
    terminal_payload["evidence_sha256"] = common.canonical_json_sha256(
        terminal_payload)
    c10_terminal_registry_path = os.path.join(
        out_dir, "c10_terminal_outcome.json")
    common.atomic_write_json(c10_terminal_registry_path, terminal_payload)
art = wandb.Artifact(
    re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-results"),
    type="expdis-results", metadata=artifact_metadata)
patterns = ["metrics.jsonl", "pipeline_state.json", "run_summary.json",
            "c10_terminal_outcome.json",
            "c10_exclusive_lease_binding.json",
            "orchestrator_receipt_binding.json",
            "mrme_contract.json", "mrme_shard_partition.json",
            "mrme_completed_campaign.json",
            "mrme_portable_campaign.json",
            "round_*/dataset_shard_manifest.json",
            "round_*/mrme_scout_sources.json",
            "round_*/mrme_trajectory_source_manifest.json",
            "round_*/scouts/*/mrme_scout_source.json",
            "round_*/scouts/*/explorer_trajectories.jsonl",
            "round_*/scouts/*/final/trainer_state.json",
            "round_*/scouts/*/final_artifact_binding.json",
            "round_*/central/final_artifact_binding.json",
            "round_*/scouts/*/c9_behavior_gates/*/heldout_selection.jsonl",
            "round_*/scouts/*/c9_behavior_gates/*/rollouts.jsonl",
            "round_*/scouts/*/c9_behavior_gates/*/aggregate_metrics.json",
            "round_*/scouts/*/c9_behavior_gates/*/evidence.json",
            "round_*/accepted.jsonl", "round_*/harvest_trajectories.jsonl",
            "round_*/*/explorer_trajectories.jsonl",
            "round_*/scout/fresh_c8_scout_completion_evidence.json",
            "round_*/scout/final_artifact_binding.json",
            "round_*/scout/final/trainer_state.json",
            "round_*/sft/actual_sft_final/sft_provenance.json",
            "round_*/sft_model_artifact.json",
            "round_*/fresh_c8_sft_training_evidence.json",
            "round_*/scout/c9_scout_completion_evidence.json",
            "round_*/scout/c10_training_terminal_outcome.json",
            "round_*/scout/c10_training_transport_invalid.json",
            "round_*/scout/c10_dynamic_sampling_failure_journal.json",
            "round_*/scout/c10_failed_step_candidates.jsonl",
            "round_*/scout/c10_failed_selected_batch.jsonl",
            "round_*/scout/c10_failed_selected_batch_manifest.json",
            "round_*/c9_sft_training_evidence.json",
            "round_*/c9_central_lineage.json",
            "round_*/c9_multi_explorer_central_lineage.json",
            "round_*/c9_behavior_gates/*/heldout_selection.jsonl",
            "round_*/c9_behavior_gates/*/rollouts.jsonl",
            "round_*/c9_behavior_gates/*/aggregate_metrics.json",
            "round_*/c9_behavior_gates/*/evidence.json",
            "round_*/c10_behavior_gates/*/heldout_selection.jsonl",
            "round_*/c10_behavior_gates/*/rollouts.jsonl",
            "round_*/c10_behavior_gates/*/aggregate_metrics.json",
            "round_*/c10_behavior_gates/*/evidence.json",
            "round_*/c10_endpoint_replay.json",
            "round_*/sft/sft_completed_run.json",
            "round_*/sft_selection_provenance.json",
            "round_*/external_central_lineage.json"]
n = 0
for pat in patterns:
    for path in glob.glob(os.path.join(out_dir, pat)):
        art.add_file(path, name=os.path.relpath(path, out_dir))
        n += 1
print(f"[artifacts] {n} result files", flush=True)
logged_results = run.log_artifact(art)
# A portable MR-ME completion is only durable after the thin evidence package
# itself has committed.  Do not let process exit/background SDK flushing be
# the scientific persistence boundary.
logged_results.wait()
if artifact_metadata.get("c10_outcome") == "transport_invalid":
    # Release only after the signed transport bundle itself is durable.  An
    # incomplete/unknown crash deliberately leaves the lease active and must
    # resume under the same immutable orchestrator receipt.
    from tmx_gpu import c10_evidence

    with open(os.path.join(
            out_dir, "c10_exclusive_lease_binding.json"),
            encoding="utf-8") as handle:
        lease_binding = json.load(handle)
    c10_evidence.validate_exclusive_lease_binding(lease_binding)
    transport_scope = artifact_metadata["c10_transport_scope"]
    evidence_key = (
        "c10_training_transport_evidence_sha256"
        if transport_scope == "training" else
        "c10_canary_behavior_gate_evidence_sha256")
    release_claim = c10_evidence.build_transport_release_claim(
        active_binding=lease_binding,
        result_artifact_ref=logged_results.qualified_name,
        result_artifact_digest=logged_results.digest,
        transport_scope=transport_scope,
        transport_evidence_sha256=artifact_metadata[evidence_key],
    )
    c10_evidence.validate_transport_release_result(
        release_claim,
        result_artifact_ref=logged_results.qualified_name,
        result_artifact_digest=logged_results.digest,
        result_metadata=artifact_metadata,
        result_root=out_dir,
    )
    release_path = os.path.join(out_dir, "c10_lease_release.json")
    common.atomic_write_json(release_path, release_claim)
    release_artifact = wandb.Artifact(
        c10_evidence.C10_EXCLUSIVE_LEASE_ARTIFACT,
        type="c10-exclusive-lease",
        metadata=c10_evidence.lease_claim_metadata(release_claim))
    release_artifact.add_file(
        release_path, name="c10_exclusive_lease_claim.json")
    logged_release = run.log_artifact(
        release_artifact, aliases=["latest"])
    logged_release.wait()
if c10_terminal_registry_path is not None:
    registry = wandb.Artifact(
        "tmx-c10-survivor-canary-terminal-0e3448b184c5",
        type="c10-terminal-outcome", metadata=artifact_metadata)
    registry.add_file(
        c10_terminal_registry_path, name="c10_terminal_outcome.json")
    logged_registry = run.log_artifact(registry, aliases=["latest"])
    logged_registry.wait()

summary_path = os.path.join(out_dir, "run_summary.json")
final_model = None
if pipeline_status == 0 and os.path.exists(summary_path):
    with open(summary_path) as f:
        final_model = json.load(f).get("final_model")
if os.environ.get("C9_EXECUTION_PROFILE") in {"full", "canary"}:
    final_model = c9_publishable_final
if os.environ.get("C10_EXECUTION_PROFILE") == "canary":
    final_model = c10_publishable_final
if true_mrme:
    final_model = mrme_publishable_final
if final_model and os.path.isdir(final_model) and os.environ.get("PUSH_MODEL_ARTIFACT", "1") == "1":
    final_model_manifest = common.directory_file_manifest(final_model)
    final_model_metadata = {
        **artifact_metadata,
        "model_file_manifest_sha256": final_model_manifest["sha256"],
    }
    m = wandb.Artifact(
        re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-final-model"),
        type="model", metadata=final_model_metadata)
    m.add_dir(final_model)
    logged_model = run.log_artifact(m)
    logged_model.wait()
    model_ref = str(
        getattr(logged_model, "qualified_name", "")
        or getattr(logged_model, "name", "") or "")
    common.require_immutable_wandb_artifact_ref(
        model_ref, field="published final-model artifact")
    if not str(getattr(logged_model, "digest", "") or ""):
        raise RuntimeError("published final-model artifact lacks a remote digest")
    if not callable(getattr(logged_model, "files", None)):
        raise RuntimeError("published final-model artifact cannot enumerate members")
    remote_names = sorted(str(item.name) for item in logged_model.files())
    expected_names = sorted(
        str(item["relative_path"])
        for item in final_model_manifest["files"])
    if remote_names != expected_names:
        raise RuntimeError(
            "published final-model artifact membership mismatch after upload")
    common.verify_directory_file_manifest(final_model, final_model_manifest)
    print(
        f"[artifacts] durably pushed final model {model_ref} from {final_model}",
        flush=True)

# Opt-in SFT behavior-gate checkpoints are full model directories.  Upload
# each as its own artifact so they remain available after the ephemeral node
# exits; no directories exist (and therefore no extra artifacts are created)
# under the default final-only contract.
checkpoint_dirs = (sorted(glob.glob(
    os.path.join(out_dir, "round_*", "sft", "optimizer_step_*")))
    if os.environ.get("PUSH_MODEL_ARTIFACT", "1") == "1" else [])
for checkpoint_dir in checkpoint_dirs:
    if not os.path.isdir(checkpoint_dir):
        continue
    metadata_path = os.path.join(checkpoint_dir, "sft_checkpoint.json")
    completed_path = os.path.join(
        os.path.dirname(checkpoint_dir), "sft_completed_run.json")
    if not os.path.isfile(metadata_path) or not os.path.isfile(completed_path):
        print(f"[artifacts] skipping incomplete SFT checkpoint {checkpoint_dir}: "
              "completed-run evidence is absent", flush=True)
        continue
    with open(metadata_path, encoding="utf-8") as f:
        checkpoint_metadata = json.load(f)
    with open(completed_path, encoding="utf-8") as f:
        completed_evidence = json.load(f)
    unsigned_completed = dict(completed_evidence)
    recorded_completed_sha = unsigned_completed.pop("evidence_sha256", None)
    import hashlib
    recomputed_completed_sha = hashlib.sha256(
        json.dumps(unsigned_completed, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()).hexdigest()
    if recorded_completed_sha != recomputed_completed_sha or \
            checkpoint_metadata.get("completed_run_evidence") != completed_evidence or \
            checkpoint_metadata.get("training_completed") is not True:
        print(f"[artifacts] skipping incomplete SFT checkpoint {checkpoint_dir}: "
              "completed-run evidence is invalid", flush=True)
        continue
    expected_dirs = {
        os.path.join(os.path.dirname(checkpoint_dir), f"optimizer_step_{int(step):06d}")
        for step in completed_evidence.get("saved_optimizer_steps", [])
    }
    if not expected_dirs or not all(os.path.isdir(path) for path in expected_dirs):
        print(f"[artifacts] skipping incomplete SFT checkpoint {checkpoint_dir}: "
              "the completed snapshot set is absent", flush=True)
        continue
    rel = os.path.relpath(checkpoint_dir, out_dir)
    match = re.search(r"round_(\d+)/sft/optimizer_step_(\d+)$", rel)
    suffix = (f"r{match.group(1)}-sft-step-{match.group(2)}"
              if match else re.sub(r"[^a-zA-Z0-9_.-]", "-", rel))
    checkpoint_artifact = wandb.Artifact(
        re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-{suffix}"),
        type="model",
        metadata={"checkpoint_kind": "post_optimizer_step_hf_model",
                  "relative_path": rel},
    )
    checkpoint_artifact.add_dir(checkpoint_dir)
    run.log_artifact(checkpoint_artifact)
    print(f"[artifacts] pushed SFT checkpoint dir {checkpoint_dir}", flush=True)
run.finish()
PY

echo "[bootstrap] done, status=$STATUS"
exit $STATUS
