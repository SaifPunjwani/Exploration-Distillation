#!/usr/bin/env bash
# Lepton job entrypoint: base-model benchmark evals on one 8-GPU node.
# Expects env: WANDB_API_KEY (secret), immutable CODE_ARTIFACT and digest,
# RUN_NAME, EVAL_CONTAINER_IMAGE_IDENTITY, ORCHESTRATOR_RECEIPT_ARTIFACT,
# ORCHESTRATOR_COMMAND_SHA256, and optionally MODEL/BENCHMARKS/MAX_PROBLEMS.
set -euo pipefail

# --- durable logging: mirror everything to a file and push to W&B on exit ----
exec > >(tee -a /workspace/job.log) 2>&1
upload_logs() {
  status=$?
  if [ "$status" -ne 0 ] && [ -n "${PUBLICATION_MATRIX:-}" ]; then
    rm -f "$PUBLICATION_MATRIX"
  fi
  echo "[bootstrap] exit status=$status — uploading logs to W&B"
  python3 - "$status" <<'PY' || true
import glob, sys, os, wandb
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "two-model-explore-gpu"),
                 entity=os.environ.get("WANDB_ENTITY") or None,
                 name=f"joblogs-{os.environ.get('RUN_NAME','unknown')}",
                 job_type="job-logs", resume="allow",
                 id=f"joblogs-{os.environ.get('RUN_NAME','unknown')}")
art = wandb.Artifact(f"joblogs-{os.environ.get('RUN_NAME','unknown')}", type="logs")
for f in ["/workspace/job.log"] + sorted(glob.glob("/workspace/vllm_*.log")):
    if os.path.exists(f):
        art.add_file(f, name=os.path.basename(f))
run.log_artifact(art)
run.summary["exit_status"] = int(sys.argv[1])
run.finish()
print("[bootstrap] logs uploaded")
PY
  if [ "$status" -ne 0 ]; then
    python3 - "$status" <<'PY' || true
import glob, os, sys, wandb
files = sorted(glob.glob(
    f"/workspace/code/baseline_runs_gpu/results/"
    f"{os.environ.get('RUN_NAME','unknown')}/**/rollouts.jsonl",
    recursive=True,
))
files += sorted(glob.glob(
    f"/workspace/code/baseline_runs_gpu/results/"
    f"{os.environ.get('RUN_NAME','unknown')}/**/evaluation_manifest.json",
    recursive=True,
))
if files:
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "two-model-explore-gpu"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=f"eval-recovery-{os.environ.get('RUN_NAME','unknown')}",
        job_type="eval-recovery", resume="allow",
        id=f"eval-recovery-{os.environ.get('RUN_NAME','unknown')}",
    )
    artifact = wandb.Artifact(
        f"eval-recovery-{os.environ.get('RUN_NAME','unknown')}",
        type="eval-incomplete",
        metadata={
            "headline_eligible": False,
            "evaluation_status": "incomplete",
            "exit_status": int(sys.argv[1]),
        },
    )
    for path in files:
        artifact.add_file(path, name=os.path.relpath(
            path, "/workspace/code/baseline_runs_gpu/results"))
    run.log_artifact(artifact)
    run.summary["headline_eligible"] = False
    run.summary["evaluation_status"] = "incomplete"
    run.finish(exit_code=int(sys.argv[1]))
PY
  fi
}
trap upload_logs EXIT

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODEL_SOURCE_NAME_INPUT="${MODEL_SOURCE_NAME:-}"
MODEL_SOURCE_NAME="${MODEL_SOURCE_NAME_INPUT:-$MODEL}"
MODEL_FAMILY="${MODEL_FAMILY:-}"
if [ -z "$MODEL_FAMILY" ]; then
  MODEL_FAMILY_IDENTITY="${MODEL_ARTIFACT:-$MODEL_SOURCE_NAME}"
  case "$(printf '%s' "$MODEL_FAMILY_IDENTITY" | tr '[:upper:]' '[:lower:]')" in
    *qwen3-1.7b*|*qwen3-1p7b*) MODEL_FAMILY="qwen3-1.7b" ;;
    *qwen3-4b*) MODEL_FAMILY="qwen3-4b" ;;
    *ministral-3-3b*) MODEL_FAMILY="ministral-3-3b" ;;
  esac
fi
if [ -n "${MODEL_ARTIFACT:-}" ] && [ -z "$MODEL_SOURCE_NAME_INPUT" ]; then
  case "$MODEL_FAMILY" in
    qwen3-1.7b) MODEL_SOURCE_NAME="Qwen/Qwen3-1.7B" ;;
    qwen3-4b) MODEL_SOURCE_NAME="Qwen/Qwen3-4B" ;;
    ministral-3-3b) MODEL_SOURCE_NAME="mistralai/Ministral-3-3B-Instruct-2512" ;;
  esac
fi
RUN_NAME="${RUN_NAME:?RUN_NAME required}"
PROTOCOL="${PROTOCOL:-P1}"
if [ -z "${BENCHMARKS:-}" ]; then
  if [ "$PROTOCOL" = "P2" ]; then
    BENCHMARKS="AIME24 AIME25"
  else
    BENCHMARKS="AIME24 AIME25 AMC23 MATH500 Minerva-Math GSM8K"
  fi
fi
export PROTOCOL BENCHMARKS
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
if [ "$NUM_GPUS" -ne 8 ]; then
  echo "[bootstrap] FATAL: publication evaluation requires exactly 8 visible GPUs, got $NUM_GPUS"
  exit 1
fi
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"   # 2k prompt + 32k completion + headroom
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-0.24.0}"
FROZEN_EVAL_CONTAINER_IMAGE_IDENTITY="vllm/vllm-openai:v0.24.0@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
EVAL_CONTAINER_IMAGE_IDENTITY="${EVAL_CONTAINER_IMAGE_IDENTITY:-}"
if [ "$EVAL_CONTAINER_IMAGE_IDENTITY" != "$FROZEN_EVAL_CONTAINER_IMAGE_IDENTITY" ]; then
  echo "[bootstrap] FATAL: EVAL_CONTAINER_IMAGE_IDENTITY must equal the immutable publication image $FROZEN_EVAL_CONTAINER_IMAGE_IDENTITY"
  exit 1
fi
if [ "$EXPECTED_VLLM_VERSION" != "0.24.0" ]; then
  echo "[bootstrap] FATAL: EXPECTED_VLLM_VERSION must remain frozen at 0.24.0"
  exit 1
fi
CODE_ARTIFACT_REF="${CODE_ARTIFACT:-}"
CODE_ARTIFACT_DIGEST="${CODE_ARTIFACT_DIGEST:-}"
ORCHESTRATOR_RECEIPT_ARTIFACT="${ORCHESTRATOR_RECEIPT_ARTIFACT:-}"
ORCHESTRATOR_COMMAND_SHA256="${ORCHESTRATOR_COMMAND_SHA256:-}"
if [ -z "$ORCHESTRATOR_RECEIPT_ARTIFACT" ] || \
   [ -z "$ORCHESTRATOR_COMMAND_SHA256" ]; then
  echo "[bootstrap] FATAL: ORCHESTRATOR_RECEIPT_ARTIFACT and ORCHESTRATOR_COMMAND_SHA256 are required"
  exit 1
fi
MODEL_ARTIFACT_REF="${MODEL_ARTIFACT:-}"
MODEL_ARTIFACT_DIGEST="${MODEL_ARTIFACT_DIGEST:-}"
METHOD_NAME_INPUT="${METHOD_NAME:-}"
METHOD_NAME_RESOLVED="${METHOD_NAME_INPUT:-Base}"
UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_REF="${UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_REF:-}"
UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_DIGEST="${UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_DIGEST:-}"
UPSTREAM_FINAL_MODEL_ARTIFACT_DIGEST="${UPSTREAM_FINAL_MODEL_ARTIFACT_DIGEST:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
if [ -z "$MODEL_REVISION" ]; then
  case "$MODEL_FAMILY" in
    qwen3-1.7b) MODEL_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" ;;
    qwen3-4b) MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c" ;;
    ministral-3-3b) MODEL_REVISION="7046a0e237b436c8fb4927061ab3773772e53741" ;;
  esac
fi
EVAL_RECOVERY_ARTIFACT="${EVAL_RECOVERY_ARTIFACT:-}"
# PROTOCOL=P2 requests 38912 completion tokens: servers must fit 2048+38912.
if [ "${PROTOCOL:-P1}" = "P2" ] && [ "$MAX_MODEL_LEN" -lt 40960 ]; then
  echo "[bootstrap] PROTOCOL=P2 needs 2048+38912 tokens: raising MAX_MODEL_LEN $MAX_MODEL_LEN -> 40960"
  MAX_MODEL_LEN=40960
fi
if [ "${PROTOCOL:-P1}" = "P2" ]; then
  for benchmark in $BENCHMARKS; do
    case "$benchmark" in
      AIME24|AIME25) ;;
      *) echo "[bootstrap] FATAL: P2 is restricted to AIME24/AIME25, got $benchmark"; exit 1 ;;
    esac
  done
fi
PORT_BASE=8000
WORK=/workspace
ORCHESTRATOR_RECEIPT_BINDING="$WORK/evaluation_orchestrator_receipt_binding.json"
EVALUATION_EXECUTION_ENVIRONMENT="$WORK/evaluation_execution_environment.json"
mkdir -p "$WORK" && cd "$WORK"

echo "[bootstrap] $(date -u) model=$MODEL gpus=$NUM_GPUS run=$RUN_NAME"
nvidia-smi -L || true

# The image digest freezes torch/vLLM/transformers/tokenizers.  Install only
# the lightweight packages used by the evaluated, empirically probed runtime;
# publication mode below verifies the complete eleven-package identity before
# any model server is launched.
export DEBIAN_FRONTEND=noninteractive
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m pip install --disable-pip-version-check --no-cache-dir \
  "wandb==0.28.0" \
  "datasets==5.0.0" \
  "hf-transfer==0.1.9" \
  "requests==2.34.2" \
  "huggingface-hub==1.21.0" \
  "numpy==2.2.6" \
  "safetensors==0.8.0"

# ---- code is fetched by the job command before this script runs --------------
cd /workspace/code
test -f tmx_gpu/eval_gpu.py || { echo "[bootstrap] FATAL: code tree missing"; exit 1; }
test -f tmx_gpu/eval_matrix.py || { echo "[bootstrap] FATAL: matrix verifier missing"; exit 1; }
EVAL_ROOT="/workspace/code/baseline_runs_gpu/results/$RUN_NAME"
if [ "$PROTOCOL" = "P2" ]; then
  PUBLICATION_MATRIX="$EVAL_ROOT/qwen_report_p2_anchor_matrix.json"
else
  PUBLICATION_MATRIX="$EVAL_ROOT/publication_matrix.json"
fi
# A prior promotion must not survive a rerun which later fails or is killed.
rm -f "$PUBLICATION_MATRIX"

if [ -z "$CODE_ARTIFACT_REF" ]; then
  echo "[bootstrap] FATAL: CODE_ARTIFACT is required for a publication evaluation"
  exit 1
fi
python3 - "$CODE_ARTIFACT_REF" "$CODE_ARTIFACT_DIGEST" <<'PY'
import hashlib, json, pathlib, re, sys, wandb
from tmx_gpu.eval_gpu import EXECUTING_CODE_FILES

ref, expected = sys.argv[1:]
if re.fullmatch(r"[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+", ref) is None:
    raise SystemExit("code artifact ref must be an immutable entity/project/name:vN")
artifact = wandb.Api().artifact(ref)
if artifact.qualified_name != ref:
    raise SystemExit(
        f"code artifact resolved identity {artifact.qualified_name!r} differs from {ref!r}")
if expected and artifact.digest != expected:
    raise SystemExit(
        f"code artifact digest mismatch: resolved {artifact.digest}, expected {expected}")
root = pathlib.Path(artifact.download(root="/workspace/eval_code_verification"))
manifest_members = set(artifact.manifest.entries)
missing_required = sorted(set(EXECUTING_CODE_FILES) - manifest_members)
if missing_required:
    raise SystemExit(
        f"code artifact is missing evaluator dependency files: {missing_required}")
code_members = sorted(
    relative for relative in manifest_members
    if relative.startswith(("tmx_gpu/", "tmx_jax/")))
if not code_members:
    raise SystemExit("code artifact has no attributed tmx_gpu/tmx_jax tree")
for relative in code_members:
    downloaded = root / relative
    executing = pathlib.Path("/workspace/code") / relative
    if not downloaded.is_file() or not executing.is_file():
        raise SystemExit(f"executing code tree is missing artifact member {relative}")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(downloaded) != digest(executing):
        raise SystemExit(f"executing code does not match code artifact: {relative}")
pathlib.Path("/workspace/code_artifact_digest.txt").write_text(
    artifact.digest + "\n")
pathlib.Path("/workspace/code_artifact_ref.txt").write_text(
    artifact.qualified_name + "\n")
PY
CODE_ARTIFACT_DIGEST="$(tr -d '\r\n' < /workspace/code_artifact_digest.txt)"
CODE_ARTIFACT_REF="$(tr -d '\r\n' < /workspace/code_artifact_ref.txt)"

# Fail before artifact downloads/server allocation if either the immutable
# image declaration or any observed package version differs from the frozen
# evaluator environment.
python3 - "$EVAL_CONTAINER_IMAGE_IDENTITY" "$EXPECTED_VLLM_VERSION" <<'PY'
import sys

from tmx_gpu.eval_gpu import (
    PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY,
    PUBLICATION_RUNTIME_STACK_VERSIONS,
    PUBLICATION_VLLM_VERSION,
    runtime_stack_versions,
)

image, expected_vllm = sys.argv[1:]
if image != PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY:
    raise SystemExit(
        f"evaluation image identity {image!r} differs from frozen "
        f"{PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY!r}")
if expected_vllm != PUBLICATION_VLLM_VERSION:
    raise SystemExit(
        f"expected vLLM {expected_vllm!r} differs from frozen "
        f"{PUBLICATION_VLLM_VERSION!r}")
actual = runtime_stack_versions()
if actual != PUBLICATION_RUNTIME_STACK_VERSIONS:
    mismatches = {
        name: {"actual": actual.get(name), "expected": expected}
        for name, expected in PUBLICATION_RUNTIME_STACK_VERSIONS.items()
        if actual.get(name) != expected
    }
    raise SystemExit(f"frozen evaluation runtime mismatch: {mismatches}")
print(f"[bootstrap] immutable evaluation environment verified: {image}")
PY

# Resolve the controller-captured raw Lepton job spec to one immutable :vN
# artifact before model download or server startup.  Then capture all eight
# native runtime facts while the complete scheduled device set is still idle.
python3 -m tmx_gpu.orchestrator_receipt resolve \
  --profile strict_publication_evaluation_v1 \
  --artifact-ref "$ORCHESTRATOR_RECEIPT_ARTIFACT" \
  --output-binding "$ORCHESTRATOR_RECEIPT_BINDING" \
  --download-root "$WORK/evaluation-orchestrator-receipt" \
  --expected-run-name "$RUN_NAME" \
  --expected-container-image "$FROZEN_EVAL_CONTAINER_IMAGE_IDENTITY" \
  --expected-code-artifact-ref "$CODE_ARTIFACT_REF" \
  --expected-code-artifact-digest "$CODE_ARTIFACT_DIGEST" \
  --expected-command-sha256 "$ORCHESTRATOR_COMMAND_SHA256"

python3 - "$ORCHESTRATOR_RECEIPT_BINDING" \
  "$EVALUATION_EXECUTION_ENVIRONMENT" "$RUN_NAME" \
  "$FROZEN_EVAL_CONTAINER_IMAGE_IDENTITY" "$CODE_ARTIFACT_REF" \
  "$CODE_ARTIFACT_DIGEST" "$ORCHESTRATOR_COMMAND_SHA256" \
  "$ORCHESTRATOR_RECEIPT_ARTIFACT" <<'PY'
import sys

from tmx_gpu import common, orchestrator_receipt
from tmx_gpu.eval_gpu import (
    build_publication_evaluation_execution_environment,
)

(
    binding_path,
    output_path,
    run_name,
    image,
    code_ref,
    code_digest,
    command_sha256,
    polling_ref,
) = sys.argv[1:]
binding = orchestrator_receipt.load_binding(
    binding_path,
    expected_run_name=run_name,
    expected_container_image=image,
    expected_code_artifact_ref=code_ref,
    expected_code_artifact_digest=code_digest,
    expected_command_sha256=command_sha256,
    expected_polling_artifact_ref=polling_ref,
    profile=orchestrator_receipt.EVALUATION_PROFILE,
)
native = orchestrator_receipt.observed_native_compatibility_facts()
environment = build_publication_evaluation_execution_environment(
    binding, native)
common.atomic_write_json(output_path, environment)
print(
    "[bootstrap] receipt-bound evaluation runtime captured: "
    f"{binding['source']['artifact_ref']} / "
    f"{native['visible_gpu_count']}x {native['gpu_family']}",
    flush=True,
)
PY

if [ -n "$EVAL_RECOVERY_ARTIFACT" ]; then
  echo "[bootstrap] restoring incomplete evaluation $EVAL_RECOVERY_ARTIFACT"
  python3 - "$EVAL_RECOVERY_ARTIFACT" "$RUN_NAME" <<'PY'
import pathlib, sys, wandb

ref, run_name = sys.argv[1:]
artifact = wandb.Api().artifact(ref)
if artifact.type != "eval-incomplete" or artifact.metadata.get(
        "headline_eligible") is not False:
    raise SystemExit("recovery artifact is not a non-headline eval-incomplete artifact")
entries = sorted(artifact.manifest.entries)
if not entries or any(not name.startswith(run_name + "/") for name in entries):
    raise SystemExit("recovery artifact paths do not match RUN_NAME")
root = pathlib.Path("/workspace/code/baseline_runs_gpu/results")
root.mkdir(parents=True, exist_ok=True)
artifact.download(root=str(root))
PY
fi

# ---- optional: evaluate a trained checkpoint from a W&B model artifact --------
if [ -n "$MODEL_ARTIFACT_REF" ]; then
  echo "[bootstrap] fetching model artifact $MODEL_ARTIFACT_REF"
  python3 - "$MODEL_ARTIFACT_REF" "$MODEL_ARTIFACT_DIGEST" <<'PY'
import json, os, pathlib, re, shutil, sys, wandb
ref, expected = sys.argv[1:]
if re.fullmatch(r"[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+", ref) is None:
    raise SystemExit("model artifact ref must be an immutable entity/project/name:vN")
art = wandb.Api().artifact(ref)
if art.qualified_name != ref:
    raise SystemExit(
        f"model artifact resolved identity {art.qualified_name!r} differs from {ref!r}")
if expected and art.digest != expected:
    raise SystemExit(
        f"model artifact digest mismatch: resolved {art.digest}, expected {expected}")
shutil.rmtree("/workspace/model", ignore_errors=True)
art.download(root="/workspace/model")
pathlib.Path("/workspace/model_artifact_resolution.json").write_text(json.dumps({
    "artifact_ref": art.qualified_name,
    "artifact_digest": art.digest,
    "artifact_type": art.type,
    "metadata": dict(art.metadata or {}),
}, sort_keys=True) + "\n")
print("[bootstrap] model artifact -> /workspace/model:", sorted(os.listdir("/workspace/model"))[:8])
PY
  MODEL_ARTIFACT_REF="$(python3 -c 'import json; print(json.load(open("/workspace/model_artifact_resolution.json"))["artifact_ref"])')"
  MODEL_ARTIFACT_DIGEST="$(python3 -c 'import json; print(json.load(open("/workspace/model_artifact_resolution.json"))["artifact_digest"])')"
  MODEL=/workspace/model
  test -f /workspace/model/config.json || { echo "[bootstrap] FATAL: model artifact lacks config.json"; exit 1; }
fi

# A trained ExpDis checkpoint is identified by its immutable model artifact and
# signed campaign evidence, never by METHOD_NAME.  Download and validate the
# complete campaign/model/code binding before starting an expensive server.
if [ -n "$UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_REF" ]; then
  if [ -z "$MODEL_ARTIFACT_REF" ]; then
    echo "[bootstrap] FATAL: training campaign evidence requires MODEL_ARTIFACT"
    exit 1
  fi
  python3 - "$UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_REF" \
    "$UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_DIGEST" <<'PY'
import json, pathlib, re, shutil, sys, wandb
ref, expected = sys.argv[1:]
if re.fullmatch(r"[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+", ref) is None:
    raise SystemExit(
        "training campaign artifact ref must be an immutable entity/project/name:vN")
art = wandb.Api().artifact(ref)
if art.qualified_name != ref:
    raise SystemExit(
        f"campaign artifact resolved identity {art.qualified_name!r} differs from {ref!r}")
if expected and art.digest != expected:
    raise SystemExit(
        f"campaign artifact digest mismatch: resolved {art.digest}, expected {expected}")
root = pathlib.Path("/workspace/training_campaign_evidence")
shutil.rmtree(root, ignore_errors=True)
art.download(root=str(root))
pathlib.Path("/workspace/training_campaign_artifact_resolution.json").write_text(
    json.dumps({
        "artifact_ref": art.qualified_name,
        "artifact_digest": art.digest,
        "artifact_type": art.type,
        "metadata": dict(art.metadata or {}),
    }, sort_keys=True) + "\n")
PY
fi

# SingleScout Central lineage names a separate immutable SFT model artifact.
# Resolve and download those bytes before scientific validation; a signed JSON
# binding without the referenced model is not a portable end-to-end replay.
rm -f /workspace/sft_model_artifact_resolution.json \
      /workspace/scout_model_artifact_resolution.json \
      /workspace/central_training_artifact_resolution.json \
      /workspace/dapo_results_artifact_resolution.json \
      /workspace/dapo_checkpoint_artifact_resolution.json
rm -rf /workspace/sft_model_artifact \
       /workspace/scout_model_artifact \
       /workspace/central_training_artifact \
       /workspace/dapo_results_artifact \
       /workspace/dapo_checkpoint_artifact
if [ -f /workspace/training_campaign_artifact_resolution.json ] && \
   [ -f /workspace/model_artifact_resolution.json ]; then
  python3 <<'PY'
import json
import pathlib
import re
import shutil

import wandb

from tmx_gpu.eval_matrix import (
    DAPO_METHOD,
    EXPDIS_SINGLESCOUT_METHOD,
    infer_expdis_method_from_artifact_metadata,
)
from tmx_gpu import dapo_evidence

campaign_resolution = json.loads(pathlib.Path(
    "/workspace/training_campaign_artifact_resolution.json").read_text())
model_resolution = json.loads(pathlib.Path(
    "/workspace/model_artifact_resolution.json").read_text())
campaign_method = None
if campaign_resolution["metadata"].get("method") == DAPO_METHOD:
    campaign_method = DAPO_METHOD
    if campaign_resolution["artifact_type"] != \
            dapo_evidence.DAPO_CAMPAIGN_ARTIFACT_TYPE:
        raise SystemExit("DAPO campaign has the wrong artifact type")
    campaign_root = pathlib.Path("/workspace/training_campaign_evidence")
    completion = dapo_evidence.validate_completion_binding(json.loads(
        (campaign_root / dapo_evidence.DAPO_COMPLETION_BINDING_FILENAME).read_text()))
    source = completion["results_artifact"]["source"]
    ref = str(source["artifact_ref"])
    artifact = wandb.Api().artifact(ref)
    if artifact.qualified_name != ref or \
            artifact.digest != source["artifact_digest"] or \
            artifact.type != dapo_evidence.DAPO_RESULTS_ARTIFACT_TYPE:
        raise SystemExit("DAPO results artifact resolution mismatch")
    root = pathlib.Path("/workspace/dapo_results_artifact")
    shutil.rmtree(root, ignore_errors=True)
    artifact.download(root=str(root))
    pathlib.Path("/workspace/dapo_results_artifact_resolution.json").write_text(
        json.dumps({
            "artifact_ref": artifact.qualified_name,
            "artifact_digest": artifact.digest,
            "artifact_type": artifact.type,
            "metadata": dict(artifact.metadata or {}),
        }, sort_keys=True) + "\n")
    checkpoint_source = completion["checkpoint_artifact"]["source"]
    checkpoint_ref = str(checkpoint_source["artifact_ref"])
    checkpoint = wandb.Api().artifact(checkpoint_ref)
    if checkpoint.qualified_name != checkpoint_ref or \
            checkpoint.digest != checkpoint_source["artifact_digest"] or \
            checkpoint.type != dapo_evidence.DAPO_CHECKPOINT_ARTIFACT_TYPE:
        raise SystemExit("DAPO step-100 checkpoint artifact resolution mismatch")
    checkpoint_root = pathlib.Path("/workspace/dapo_checkpoint_artifact")
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    checkpoint.download(root=str(checkpoint_root))
    pathlib.Path(
        "/workspace/dapo_checkpoint_artifact_resolution.json").write_text(
        json.dumps({
            "artifact_ref": checkpoint.qualified_name,
            "artifact_digest": checkpoint.digest,
            "artifact_type": checkpoint.type,
            "metadata": dict(checkpoint.metadata or {}),
        }, sort_keys=True) + "\n")
else:
    campaign_method = infer_expdis_method_from_artifact_metadata(
        campaign_resolution["metadata"], where="campaign artifact")
    model_method = infer_expdis_method_from_artifact_metadata(
        model_resolution["metadata"], where="model artifact")
    if campaign_method != model_method:
        raise SystemExit(
            "campaign/model method mismatch before SFT artifact resolution")
if campaign_method == EXPDIS_SINGLESCOUT_METHOD:
    campaign_root = pathlib.Path("/workspace/training_campaign_evidence")

    def resolve_binding(binding_path, *, expected_type, root, resolution_path):
        if not binding_path.is_file():
            raise SystemExit(
                f"SingleScout campaign lacks {binding_path.relative_to(campaign_root)}")
        binding = json.loads(binding_path.read_text())
        source = binding.get("source")
        if not isinstance(source, dict):
            raise SystemExit(f"{binding_path.name} lacks immutable source")
        ref = str(source.get("artifact_ref", ""))
        expected = str(source.get("artifact_digest", ""))
        if re.fullmatch(r"[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+", ref) is None:
            raise SystemExit(f"{binding_path.name} artifact ref is not immutable")
        artifact = wandb.Api().artifact(ref)
        if artifact.qualified_name != ref or artifact.digest != expected or \
                artifact.type != expected_type:
            raise SystemExit(f"{binding_path.name} artifact resolution mismatch")
        shutil.rmtree(root, ignore_errors=True)
        artifact.download(root=str(root))
        resolution_path.write_text(json.dumps({
            "artifact_ref": artifact.qualified_name,
            "artifact_digest": artifact.digest,
            "artifact_type": artifact.type,
            "metadata": dict(artifact.metadata or {}),
        }, sort_keys=True) + "\n")

    resolve_binding(
        campaign_root / "round_01" / "sft_model_artifact.json",
        expected_type="sft-model",
        root=pathlib.Path("/workspace/sft_model_artifact"),
        resolution_path=pathlib.Path(
            "/workspace/sft_model_artifact_resolution.json"),
    )
    resolve_binding(
        campaign_root / "round_01" / "scout" /
        "final_artifact_binding.json",
        expected_type="model",
        root=pathlib.Path("/workspace/scout_model_artifact"),
        resolution_path=pathlib.Path(
            "/workspace/scout_model_artifact_resolution.json"),
    )
    resolve_binding(
        campaign_root / "round_01" / "central" /
        "final_artifact_binding.json",
        expected_type="model",
        root=pathlib.Path("/workspace/central_training_artifact"),
        resolution_path=pathlib.Path(
            "/workspace/central_training_artifact_resolution.json"),
    )
PY
fi

python3 - "$METHOD_NAME_INPUT" "$CODE_ARTIFACT_REF" \
  "$CODE_ARTIFACT_DIGEST" "$UPSTREAM_FINAL_MODEL_ARTIFACT_DIGEST" <<'PY'
import json, pathlib, sys
from tmx_gpu.eval_matrix import (
    DAPO_METHOD,
    infer_expdis_method_from_artifact_metadata,
    validate_training_campaign_evidence,
)

requested_method, code_ref, code_digest, expected_model_digest = sys.argv[1:]
model_resolution_path = pathlib.Path("/workspace/model_artifact_resolution.json")
campaign_resolution_path = pathlib.Path(
    "/workspace/training_campaign_artifact_resolution.json")
binding_path = pathlib.Path("/workspace/training_campaign_binding.json")
binding_path.unlink(missing_ok=True)
derived_method = None
binding = None
if model_resolution_path.is_file():
    model = json.loads(model_resolution_path.read_text())
    campaign = (json.loads(campaign_resolution_path.read_text())
                if campaign_resolution_path.is_file() else None)
    is_dapo_campaign = bool(
        campaign and campaign["metadata"].get("method") == DAPO_METHOD)
    model_method = (None if is_dapo_campaign else
                    infer_expdis_method_from_artifact_metadata(
                        model["metadata"], where="model artifact"))
    if expected_model_digest and expected_model_digest != model["artifact_digest"]:
        raise SystemExit(
            "UPSTREAM_FINAL_MODEL_ARTIFACT_DIGEST differs from resolved model artifact")
    if campaign is not None:
        sft_resolution_path = pathlib.Path(
            "/workspace/sft_model_artifact_resolution.json")
        sft = (json.loads(sft_resolution_path.read_text())
               if sft_resolution_path.is_file() else None)
        scout_resolution_path = pathlib.Path(
            "/workspace/scout_model_artifact_resolution.json")
        scout = (json.loads(scout_resolution_path.read_text())
                 if scout_resolution_path.is_file() else None)
        central_resolution_path = pathlib.Path(
            "/workspace/central_training_artifact_resolution.json")
        central = (json.loads(central_resolution_path.read_text())
                   if central_resolution_path.is_file() else None)
        dapo_results_resolution_path = pathlib.Path(
            "/workspace/dapo_results_artifact_resolution.json")
        dapo_results = (json.loads(dapo_results_resolution_path.read_text())
                        if dapo_results_resolution_path.is_file() else None)
        dapo_checkpoint_resolution_path = pathlib.Path(
            "/workspace/dapo_checkpoint_artifact_resolution.json")
        dapo_checkpoint = (json.loads(
            dapo_checkpoint_resolution_path.read_text())
            if dapo_checkpoint_resolution_path.is_file() else None)
        binding = validate_training_campaign_evidence(
            pathlib.Path("/workspace/training_campaign_evidence"),
            pathlib.Path("/workspace/model"),
            campaign_artifact_ref=campaign["artifact_ref"],
            campaign_artifact_digest=campaign["artifact_digest"],
            campaign_artifact_type=campaign["artifact_type"],
            campaign_artifact_metadata=campaign["metadata"],
            model_artifact_ref=model["artifact_ref"],
            model_artifact_digest=model["artifact_digest"],
            model_artifact_type=model["artifact_type"],
            model_artifact_metadata=model["metadata"],
            code_artifact_ref=code_ref,
            code_artifact_digest=code_digest,
            sft_artifact_root=(
                pathlib.Path("/workspace/sft_model_artifact") if sft else None),
            sft_artifact_ref=(sft["artifact_ref"] if sft else None),
            sft_artifact_digest=(sft["artifact_digest"] if sft else None),
            sft_artifact_type=(sft["artifact_type"] if sft else None),
            sft_artifact_metadata=(sft["metadata"] if sft else None),
            scout_artifact_root=(
                pathlib.Path("/workspace/scout_model_artifact")
                if scout else None),
            scout_artifact_ref=(scout["artifact_ref"] if scout else None),
            scout_artifact_digest=(scout["artifact_digest"] if scout else None),
            scout_artifact_type=(scout["artifact_type"] if scout else None),
            scout_artifact_metadata=(scout["metadata"] if scout else None),
            central_training_artifact_root=(
                pathlib.Path("/workspace/central_training_artifact")
                if central else None),
            central_training_artifact_ref=(
                central["artifact_ref"] if central else None),
            central_training_artifact_digest=(
                central["artifact_digest"] if central else None),
            central_training_artifact_type=(
                central["artifact_type"] if central else None),
            central_training_artifact_metadata=(
                central["metadata"] if central else None),
            dapo_results_artifact_root=(
                pathlib.Path("/workspace/dapo_results_artifact")
                if dapo_results else None),
            dapo_results_artifact_ref=(
                dapo_results["artifact_ref"] if dapo_results else None),
            dapo_results_artifact_digest=(
                dapo_results["artifact_digest"] if dapo_results else None),
            dapo_results_artifact_type=(
                dapo_results["artifact_type"] if dapo_results else None),
            dapo_results_artifact_metadata=(
                dapo_results["metadata"] if dapo_results else None),
            dapo_checkpoint_artifact_root=(
                pathlib.Path("/workspace/dapo_checkpoint_artifact")
                if dapo_checkpoint else None),
            dapo_checkpoint_artifact_ref=(
                dapo_checkpoint["artifact_ref"] if dapo_checkpoint else None),
            dapo_checkpoint_artifact_digest=(
                dapo_checkpoint["artifact_digest"]
                if dapo_checkpoint else None),
            dapo_checkpoint_artifact_type=(
                dapo_checkpoint["artifact_type"] if dapo_checkpoint else None),
            dapo_checkpoint_artifact_metadata=(
                dapo_checkpoint["metadata"] if dapo_checkpoint else None),
        )
        derived_method = binding["method"]
        if derived_method != DAPO_METHOD and model_method != derived_method:
            raise SystemExit("model metadata method differs from signed campaign")
    elif model_method is not None:
        raise SystemExit(
            "trained ExpDis model artifact requires its immutable signed campaign artifact")
elif campaign_resolution_path.is_file():
    raise SystemExit("training campaign artifact cannot bind a registry Base model")

if derived_method is not None:
    if requested_method and requested_method != derived_method:
        raise SystemExit(
            f"METHOD_NAME={requested_method!r} contradicts immutable trained method "
            f"{derived_method!r}")
    method = derived_method
    binding_path.write_text(json.dumps(binding, sort_keys=True) + "\n")
    pathlib.Path("/workspace/training_campaign_binding_sha256.txt").write_text(
        binding["binding_sha256"] + "\n")
else:
    method = requested_method or "Base"
    if method == DAPO_METHOD:
        raise SystemExit(
            "DAPO-style correctness requires its immutable signed campaign, "
            "model, and results artifacts")
    pathlib.Path("/workspace/training_campaign_binding_sha256.txt").write_text("\n")
pathlib.Path("/workspace/method_name_resolved.txt").write_text(method + "\n")
PY
METHOD_NAME_RESOLVED="$(tr -d '\r\n' < /workspace/method_name_resolved.txt)"
TRAINING_CAMPAIGN_BINDING_SHA256="$(tr -d '\r\n' < /workspace/training_campaign_binding_sha256.txt)"
if [ -f /workspace/training_campaign_artifact_resolution.json ]; then
  UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_REF="$(python3 -c 'import json; print(json.load(open("/workspace/training_campaign_artifact_resolution.json"))["artifact_ref"])')"
  UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_DIGEST="$(python3 -c 'import json; print(json.load(open("/workspace/training_campaign_artifact_resolution.json"))["artifact_digest"])')"
  UPSTREAM_FINAL_MODEL_ARTIFACT_DIGEST="$MODEL_ARTIFACT_DIGEST"
fi

if [ -z "$MODEL_FAMILY" ]; then
  echo "[bootstrap] FATAL: MODEL_FAMILY could not be inferred; set qwen3-1.7b, qwen3-4b, or ministral-3-3b"
  exit 1
fi

# ---- start one vLLM server per GPU -------------------------------------------
for i in $(seq 0 $((NUM_GPUS - 1))); do
  REVISION_ARGS=()
  if [ -z "$MODEL_ARTIFACT_REF" ]; then
    REVISION_ARGS=(--revision "$MODEL_REVISION")
  fi
  CUDA_VISIBLE_DEVICES=$i nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" \
    "${REVISION_ARGS[@]}" \
    --port $((PORT_BASE + i)) --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    > "/workspace/vllm_$i.log" 2>&1 &
done

ENDPOINTS=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
  ENDPOINTS+="http://127.0.0.1:$((PORT_BASE + i))/v1,"
done
ENDPOINTS="${ENDPOINTS%,}"

echo "[bootstrap] waiting for $NUM_GPUS vLLM servers..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
  ok=0
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$((PORT_BASE + i))/health" >/dev/null 2>&1; then
      ok=1; break
    fi
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "[bootstrap] FATAL: vLLM server $i failed to become healthy"
    echo "----- vllm_$i.log (tail) -----"
    tail -80 "/workspace/vllm_$i.log" || true
    exit 1
  fi
done
echo "[bootstrap] all vLLM servers healthy: $ENDPOINTS"

# ---- run benchmarks sequentially (each fans out across all servers) ----------
STATUS=0
EXACT_MATRIX_FLAG="--require-exact-sample-matrix --require-publication-protocol"
if [ "${ALLOW_PARTIAL_EVAL:-0}" = "1" ]; then
  echo "[bootstrap] WARNING: diagnostic partial-matrix mode enabled; results are not headline-eligible"
  EXACT_MATRIX_FLAG=""
fi
for B in $BENCHMARKS; do
  echo "[bootstrap] ===== $B ====="
  python3 tmx_gpu/eval_gpu.py \
    --endpoints "$ENDPOINTS" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --model-source-name "$MODEL_SOURCE_NAME" \
    --model-family "$MODEL_FAMILY" \
    --model-revision "$MODEL_REVISION" \
    ${MODEL_ARTIFACT_REF:+--model-artifact-ref "$MODEL_ARTIFACT_REF"} \
    ${MODEL_ARTIFACT_DIGEST:+--model-artifact-digest "$MODEL_ARTIFACT_DIGEST"} \
    --code-artifact-ref "$CODE_ARTIFACT_REF" \
    --code-artifact-digest "$CODE_ARTIFACT_DIGEST" \
    ${TRAINING_CAMPAIGN_BINDING_SHA256:+--training-campaign-binding-sha256 "$TRAINING_CAMPAIGN_BINDING_SHA256"} \
    --expected-vllm-version "$EXPECTED_VLLM_VERSION" \
    --container-image-identity "$EVAL_CONTAINER_IMAGE_IDENTITY" \
    --evaluation-execution-environment "$EVALUATION_EXECUTION_ENVIRONMENT" \
    --benchmark "$B" \
    --run-name "$RUN_NAME" \
    --concurrency "${CONCURRENCY:-96}" \
    --server-max-model-len "$MAX_MODEL_LEN" \
    ${MAX_PROBLEMS:+--max-problems "$MAX_PROBLEMS"} \
    ${NUM_SAMPLES:+--num-samples "$NUM_SAMPLES"} \
    ${PROTOCOL:+--protocol "$PROTOCOL"} \
    --method-name "$METHOD_NAME_RESOLVED" \
    $EXACT_MATRIX_FLAG \
    --wandb --resume || STATUS=$?
done

if [ "$STATUS" -ne 0 ]; then
  rm -f "$PUBLICATION_MATRIX"
  echo "[bootstrap] benchmark failure; publication matrix withheld, status=$STATUS"
  exit "$STATUS"
fi

# ---- campaign-level signed evaluation boundary -----------------------------
# P1 emits the authoritative six-benchmark project matrix.  P2 emits a
# separate, explicitly non-headline AIME-only Qwen-report anchor matrix.
MATRIX_ARGS=(
  --eval-root "$EVAL_ROOT"
  --output "$PUBLICATION_MATRIX"
  --protocol "$PROTOCOL"
  --run-name "$RUN_NAME"
  --method-name "$METHOD_NAME_RESOLVED"
  --model-family "$MODEL_FAMILY"
)
if [ -n "$UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_REF" ]; then
  MATRIX_ARGS+=(
    --upstream-training-campaign-artifact-ref "$UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_REF"
  )
fi
if [ -f /workspace/training_campaign_binding.json ]; then
  MATRIX_ARGS+=(
    --training-campaign-binding /workspace/training_campaign_binding.json
    --require-upstream-training-evidence
  )
fi
if [ -n "$UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_DIGEST" ]; then
  MATRIX_ARGS+=(
    --upstream-training-campaign-artifact-digest "$UPSTREAM_TRAINING_CAMPAIGN_ARTIFACT_DIGEST"
  )
fi
if [ -n "$UPSTREAM_FINAL_MODEL_ARTIFACT_DIGEST" ]; then
  MATRIX_ARGS+=(
    --upstream-final-model-artifact-digest "$UPSTREAM_FINAL_MODEL_ARTIFACT_DIGEST"
  )
fi
if ! python3 -m tmx_gpu.eval_matrix "${MATRIX_ARGS[@]}"; then
  rm -f "$PUBLICATION_MATRIX"
  echo "[bootstrap] FATAL: $PROTOCOL evaluation matrix validation failed"
  exit 1
fi

# Upload, wait for W&B commit, download the committed object, and compare bytes
# before setting the run's headline flag.  A failed upload is not a promotion.
if ! python3 - "$PUBLICATION_MATRIX" <<'PY'
import hashlib
import json
import os
import pathlib
import shutil
import sys

import wandb

from tmx_gpu.eval_matrix import (
    validate_training_campaign_evidence,
    validate_training_campaign_binding,
    verify_evaluation_matrix_signature,
    verify_remote_eval_artifacts,
)

matrix_path = pathlib.Path(sys.argv[1])
matrix_bytes = matrix_path.read_bytes()
matrix = json.loads(matrix_bytes)
signature = verify_evaluation_matrix_signature(matrix)
protocol = matrix["protocol"]
is_headline = protocol == "P1"
matrix_metric_name = "hard_mean" if is_headline else "anchor_mean"
matrix_metric = matrix[matrix_metric_name]
matrix_filename = matrix_path.name
artifact_role = (
    "eval-publication-matrix" if is_headline else "eval-qwen-report-p2-anchor")
artifact_stem = (
    "eval-publication-matrix" if is_headline else "eval-qwen-report-p2-anchor")
source_files = {matrix_filename: matrix_path}
for benchmark in matrix["benchmark_order"]:
    for filename in (
        "aggregate_metrics.json",
        "evaluation_manifest.json",
        "evaluation_artifact_receipt.json",
    ):
        source_files[f"benchmarks/{benchmark}/{filename}"] = (
            matrix_path.parent / benchmark / filename)
project = os.environ.get("WANDB_PROJECT", "two-model-explore-gpu")
entity = os.environ.get("WANDB_ENTITY", "")
if not entity:
    raise SystemExit("WANDB_ENTITY must name the W&B entity that owns the artifacts")
run_name = os.environ["RUN_NAME"]
api = wandb.Api()
# Local receipts are editable files.  Promotion therefore independently
# resolves every immutable :vN artifact named by the signed matrix.
resolved_eval_artifacts = verify_remote_eval_artifacts(
    matrix, lambda ref: api.artifact(ref).digest)
upstream = matrix.get("upstream_training_evidence")
if upstream is not None:
    upstream = validate_training_campaign_binding(upstream)
    resolved_objects = {}
    artifact_bindings = [
        ("campaign", upstream["campaign_artifact"]),
        ("model", upstream["model_artifact"]),
        ("code", upstream["code_artifact"]),
    ]
    if upstream["method"] == "DAPO-style correctness":
        artifact_bindings.extend([
            ("results", upstream["results_artifact"]),
            ("checkpoint", upstream["checkpoint_artifact"]),
        ])
    if upstream["method"] == "expdis_single_scout":
        artifact_bindings.extend([
            ("sft", upstream["sft_artifact"]),
            ("scout", upstream["scout_artifact"]),
            ("central_training", upstream["central_training_artifact"]),
        ])
    for label, artifact_binding in artifact_bindings:
        resolved_upstream = api.artifact(artifact_binding["artifact_ref"])
        resolved_objects[label] = resolved_upstream
        if resolved_upstream.digest != artifact_binding["artifact_digest"]:
            raise RuntimeError(
                f"upstream {label} artifact digest does not match the signed "
                "publication matrix")
    # Re-download every byte-bearing upstream object into fresh promotion-only
    # roots.  Reusing the pre-server downloads would leave a TOCTOU gap in
    # which local campaign/model bytes could change while the long eval runs.
    promotion_roots = {}
    for label in ("campaign", "model", "sft", "scout", "central_training"):
        if label not in resolved_objects:
            continue
        root = pathlib.Path(f"/workspace/promotion_{label}_artifact")
        shutil.rmtree(root, ignore_errors=True)
        resolved_objects[label].download(root=str(root))
        promotion_roots[label] = root
    for label in ("results", "checkpoint"):
        if label in resolved_objects:
            root = pathlib.Path(f"/workspace/promotion_{label}_artifact")
            shutil.rmtree(root, ignore_errors=True)
            resolved_objects[label].download(root=str(root))
            promotion_roots[label] = root
    independently_revalidated = validate_training_campaign_evidence(
        promotion_roots["campaign"],
        promotion_roots["model"],
        campaign_artifact_ref=resolved_objects["campaign"].qualified_name,
        campaign_artifact_digest=resolved_objects["campaign"].digest,
        campaign_artifact_type=resolved_objects["campaign"].type,
        campaign_artifact_metadata=dict(
            resolved_objects["campaign"].metadata or {}),
        model_artifact_ref=resolved_objects["model"].qualified_name,
        model_artifact_digest=resolved_objects["model"].digest,
        model_artifact_type=resolved_objects["model"].type,
        model_artifact_metadata=dict(resolved_objects["model"].metadata or {}),
        code_artifact_ref=resolved_objects["code"].qualified_name,
        code_artifact_digest=resolved_objects["code"].digest,
        sft_artifact_root=promotion_roots.get("sft"),
        sft_artifact_ref=(resolved_objects["sft"].qualified_name
                          if "sft" in resolved_objects else None),
        sft_artifact_digest=(resolved_objects["sft"].digest
                             if "sft" in resolved_objects else None),
        sft_artifact_type=(resolved_objects["sft"].type
                           if "sft" in resolved_objects else None),
        sft_artifact_metadata=(dict(resolved_objects["sft"].metadata or {})
                               if "sft" in resolved_objects else None),
        scout_artifact_root=promotion_roots.get("scout"),
        scout_artifact_ref=(resolved_objects["scout"].qualified_name
                            if "scout" in resolved_objects else None),
        scout_artifact_digest=(resolved_objects["scout"].digest
                               if "scout" in resolved_objects else None),
        scout_artifact_type=(resolved_objects["scout"].type
                             if "scout" in resolved_objects else None),
        scout_artifact_metadata=(dict(
            resolved_objects["scout"].metadata or {})
            if "scout" in resolved_objects else None),
        central_training_artifact_root=promotion_roots.get(
            "central_training"),
        central_training_artifact_ref=(
            resolved_objects["central_training"].qualified_name
            if "central_training" in resolved_objects else None),
        central_training_artifact_digest=(
            resolved_objects["central_training"].digest
            if "central_training" in resolved_objects else None),
        central_training_artifact_type=(
            resolved_objects["central_training"].type
            if "central_training" in resolved_objects else None),
        central_training_artifact_metadata=(dict(
            resolved_objects["central_training"].metadata or {})
            if "central_training" in resolved_objects else None),
        dapo_results_artifact_root=promotion_roots.get("results"),
        dapo_results_artifact_ref=(
            resolved_objects["results"].qualified_name
            if "results" in resolved_objects else None),
        dapo_results_artifact_digest=(
            resolved_objects["results"].digest
            if "results" in resolved_objects else None),
        dapo_results_artifact_type=(
            resolved_objects["results"].type
            if "results" in resolved_objects else None),
        dapo_results_artifact_metadata=(dict(
            resolved_objects["results"].metadata or {})
            if "results" in resolved_objects else None),
        dapo_checkpoint_artifact_root=promotion_roots.get("checkpoint"),
        dapo_checkpoint_artifact_ref=(
            resolved_objects["checkpoint"].qualified_name
            if "checkpoint" in resolved_objects else None),
        dapo_checkpoint_artifact_digest=(
            resolved_objects["checkpoint"].digest
            if "checkpoint" in resolved_objects else None),
        dapo_checkpoint_artifact_type=(
            resolved_objects["checkpoint"].type
            if "checkpoint" in resolved_objects else None),
        dapo_checkpoint_artifact_metadata=(dict(
            resolved_objects["checkpoint"].metadata or {})
            if "checkpoint" in resolved_objects else None),
    )
    if independently_revalidated != upstream:
        raise RuntimeError(
            "publication binding differs from independent campaign replay")
run = wandb.init(
    project=project,
    entity=entity,
    name=f"{artifact_stem}-{run_name}",
    job_type=artifact_role,
    id=f"{artifact_stem}-{run_name}",
    resume="allow",
)
run.summary["headline_eligible"] = False
run.summary["publication_status"] = "uploading"
try:
    artifact = wandb.Artifact(
        f"{artifact_stem}-{run_name}",
        type=artifact_role,
        metadata={
            "promotion_boundary": "pending_download_verification",
            "matrix_signature_sha256": signature,
            matrix_metric_name: matrix_metric,
            "protocol": protocol,
            "matrix_role": matrix["matrix_role"],
            "headline_eligible": is_headline,
        },
    )
    for name, path in source_files.items():
        artifact.add_file(str(path), name=name)
    logged = run.log_artifact(artifact)
    logged.wait()
    if getattr(logged, "state", None) != "COMMITTED":
        raise RuntimeError(
            f"publication artifact did not commit: state={getattr(logged, 'state', None)!r}")
    verify_root = pathlib.Path(
        f"/workspace/{artifact_stem}_upload_verify")
    shutil.rmtree(verify_root, ignore_errors=True)
    logged.download(root=str(verify_root))
    for name, source_path in source_files.items():
        committed_path = verify_root / name
        if not committed_path.is_file() or \
                committed_path.read_bytes() != source_path.read_bytes():
            raise RuntimeError(f"committed publication source mismatch: {name}")
    committed = verify_root / matrix_filename
    committed_value = json.loads(committed.read_text())
    if verify_evaluation_matrix_signature(committed_value) != signature:
        raise RuntimeError("committed evaluation matrix signature mismatch")
    run.summary["publication_artifact_ref"] = logged.qualified_name
    run.summary["publication_artifact_digest"] = logged.digest
    run.summary["matrix_signature_sha256"] = signature
    run.summary["resolved_eval_artifact_count"] = len(resolved_eval_artifacts)
    run.summary[matrix_metric_name] = matrix_metric
    run.summary["protocol"] = protocol
    run.summary["matrix_role"] = matrix["matrix_role"]
    # This is intentionally the last scientific state transition.
    run.summary["publication_status"] = "complete"
    run.summary["headline_eligible"] = is_headline
    run.finish()
except Exception:
    run.summary["headline_eligible"] = False
    run.summary["publication_status"] = "failed"
    run.finish(exit_code=1)
    raise
PY
then
  rm -f "$PUBLICATION_MATRIX"
  echo "[bootstrap] FATAL: $PROTOCOL evaluation matrix upload/verification failed"
  exit 1
fi

if [ "$PROTOCOL" = "P1" ]; then
  echo "[bootstrap] done, status=0, authoritative project P1 matrix=$PUBLICATION_MATRIX"
else
  echo "[bootstrap] done, status=0, secondary Qwen-report P2 anchor matrix=$PUBLICATION_MATRIX"
fi
exit 0
