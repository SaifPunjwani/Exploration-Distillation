#!/usr/bin/env bash
# Exact original-paper DAPO control: Base -> 100 correctness-only DR-GRPO
# updates under the exact Central route and learning-rate contract, but from
# Base rather than an SFT handoff. Lambda is fixed to zero and no external
# lineage is supplied, so this remains the single-model correctness control.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:?RUN_NAME required}"
CODE_ARTIFACT="${CODE_ARTIFACT:?CODE_ARTIFACT required}"
CODE_ARTIFACT_DIGEST="${CODE_ARTIFACT_DIGEST:?CODE_ARTIFACT_DIGEST required}"
CANONICAL_TRAINING_CONTAINER_IMAGE="vllm/vllm-openai:v0.25.0@sha256:fc56161ee42a011aeee78b65d0a81b6683c7d04402fd40503d14d4d6c98f07cb"
TRAINING_CONTAINER_IMAGE_IDENTITY="${TRAINING_CONTAINER_IMAGE_IDENTITY:?TRAINING_CONTAINER_IMAGE_IDENTITY required}"
ORCHESTRATOR_COMMAND_SHA256="${ORCHESTRATOR_COMMAND_SHA256:?ORCHESTRATOR_COMMAND_SHA256 required}"
ORCHESTRATOR_RECEIPT_ARTIFACT="${ORCHESTRATOR_RECEIPT_ARTIFACT:?ORCHESTRATOR_RECEIPT_ARTIFACT required}"
RESUME_CHECKPOINT_ARTIFACT="${RESUME_CHECKPOINT_ARTIFACT:-}"
RESUME_RESULTS_ARTIFACT="${RESUME_RESULTS_ARTIFACT:-}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
STEPS="${STEPS:-100}"
LR="${LR:-0.000001}"
SEED="${SEED:-0}"
WORK=/workspace
OUT_DIR="$WORK/runs/$RUN_NAME"
PROVENANCE_DIR="$OUT_DIR/provenance"
CODE_MANIFEST="$PROVENANCE_DIR/dapo_code_manifest.json"
CODE_BINDING="$PROVENANCE_DIR/dapo_code_binding.json"
CODE_ROOT_MARKER="$PROVENANCE_DIR/dapo_code_root.txt"
ORCHESTRATOR_RECEIPT_BINDING="$PROVENANCE_DIR/orchestrator_receipt_binding.json"
RESUME_MANIFEST="$PROVENANCE_DIR/dapo_resume_manifest.json"
RESUME_RESULTS_MANIFEST="$PROVENANCE_DIR/dapo_resume_results_manifest.json"
RESUME_ROOT_MARKER="$PROVENANCE_DIR/dapo_resume_root.txt"
RESUME_RESULTS_ROOT_MARKER="$PROVENANCE_DIR/dapo_resume_results_root.txt"
RESUME_PAIR_RECORD="$PROVENANCE_DIR/dapo_resume_pair.json"
export RUN_NAME OUT_DIR PROVENANCE_DIR CODE_MANIFEST CODE_BINDING CODE_ROOT_MARKER
export RESUME_MANIFEST RESUME_RESULTS_MANIFEST RESUME_ROOT_MARKER
export RESUME_RESULTS_ROOT_MARKER RESUME_PAIR_RECORD
export CODE_ARTIFACT RESUME_CHECKPOINT_ARTIFACT RESUME_RESULTS_ARTIFACT
export TRAINING_CONTAINER_IMAGE_IDENTITY
export CODE_ARTIFACT_DIGEST ORCHESTRATOR_COMMAND_SHA256
export ORCHESTRATOR_RECEIPT_ARTIFACT ORCHESTRATOR_RECEIPT_BINDING
export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore-gpu}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="${WANDB_DIR:-$OUT_DIR/wandb}"
mkdir -p "$WORK" "$OUT_DIR" "$WANDB_DIR"
# A run may reuse a persistent /workspace, but provenance must never inherit
# fixed-path records from an earlier run. Scientific JSONLs are handled by the
# explicit resume path below; this directory is always rebuilt from scratch.
rm -rf "$PROVENANCE_DIR"
mkdir -p "$PROVENANCE_DIR"

exec > >(tee -a "$OUT_DIR/job.log") 2>&1

if [ "$TRAINING_CONTAINER_IMAGE_IDENTITY" != \
     "$CANONICAL_TRAINING_CONTAINER_IMAGE" ]; then
  echo "[dapo] FATAL: immutable training image must be $CANONICAL_TRAINING_CONTAINER_IMAGE" >&2
  exit 1
fi
if [ "$NUM_GPUS" != "8" ]; then
  echo "[dapo] FATAL: original-paper DAPO requires exactly 8 visible A100/H100 GPUs" >&2
  exit 1
fi

# The immutable image owns torch/vLLM/transformers/tokenizers. Install only
# the exact light bootstrap packages observed in the canonical v0.25 runtime;
# grpo_gpu verifies this full combined stack before update 1.
PKGS=("wandb==0.28.0" "datasets==5.0.0" "requests==2.34.2" \
      "huggingface_hub==1.21.0" "hf-transfer==0.1.9" \
      "numpy==2.2.6" "safetensors==0.8.0")
install_deps() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --system --quiet "${PKGS[@]}" 2>&1 | tail -1 && return 0
    echo "[dapo] uv install failed; falling back to pip"
  fi
  pip install --quiet --no-cache-dir "${PKGS[@]}" 2>&1 | tail -1 && return 0
  python3 -m pip install --quiet --no-cache-dir "${PKGS[@]}" 2>&1 | tail -1
}
install_deps || { echo "[dapo] FATAL: dependency install failed"; exit 1; }
export HF_HUB_ENABLE_HF_TRANSFER=1

# Preserve the code binding, partial metrics/trajectories, and all process logs
# on every ordinary exit.  Abrupt node loss is covered by the every-25-step
# immutable W&B checkpoint artifacts; RESUME_CHECKPOINT_ARTIFACT restores one.
upload_evidence() {
  status=$?
  trap - EXIT
  set +e
  echo "[dapo] exit status=$status — uploading durable evidence"
  python3 - "$status" <<'PY'
import glob
import json
import os
import pathlib
import re
import shutil
import sys

import wandb

run_name = os.environ["RUN_NAME"]
out_dir = os.environ["OUT_DIR"]
status = int(sys.argv[1])
code_binding_path = pathlib.Path(os.environ["CODE_BINDING"])
code_source = {"artifact_ref": os.environ["CODE_ARTIFACT"], "artifact_digest": ""}
if code_binding_path.is_file():
    code_source = json.loads(code_binding_path.read_text())["source"]
run_id = re.sub(r"[^a-zA-Z0-9_-]", "-", f"{run_name}-evidence")[:120]
run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity=os.environ["WANDB_ENTITY"],
    name=f"{run_name}-evidence",
    job_type="artifacts",
    id=run_id,
    resume="allow",
)
artifact = wandb.Artifact(
    re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-results"),
    type="dapo-results",
    metadata={
        "exit_status": status,
        "completion_status": "complete" if status == 0 else "incomplete",
        "method": "DAPO-style correctness",
        "scientific_variant": "original_paper_dapo_100_central_lr",
        "code_artifact_ref": code_source["artifact_ref"],
        "code_artifact_digest": code_source["artifact_digest"],
        "resume_checkpoint_artifact": os.environ.get("RESUME_CHECKPOINT_ARTIFACT", ""),
        "resume_results_artifact": os.environ.get("RESUME_RESULTS_ARTIFACT", ""),
    },
)
paths = [
    os.environ["CODE_MANIFEST"],
    os.environ["CODE_BINDING"],
    os.environ["RESUME_MANIFEST"],
    os.environ["RESUME_RESULTS_MANIFEST"],
    os.environ["RESUME_PAIR_RECORD"],
    os.environ["ORCHESTRATOR_RECEIPT_BINDING"],
]
patterns = (
    "job.log",
    "metrics.jsonl",
    "explorer_trajectories.jsonl",
    "rollouts_all.jsonl",
    "final/trainer_state.json",
    "final_artifact_binding.json",
    "step_*/trainer_state.json",
    "vllm_logs/*.log",
)
paths.extend(
    path for pattern in patterns
    for path in sorted(glob.glob(os.path.join(out_dir, pattern)))
)
seen = set()
for path in paths:
    if os.path.isfile(path) and path not in seen:
        seen.add(path)
        if path.startswith(out_dir + os.sep):
            name = os.path.relpath(path, out_dir)
        else:
            name = os.path.basename(path)
        artifact.add_file(path, name=name)
logged = run.log_artifact(artifact, aliases=["latest"])
logged.wait()
run.summary["exit_status"] = status
run.summary["evidence_member_count"] = len(seen)

if status == 0:
    from tmx_gpu import dapo_evidence

    def artifact_identity(value, expected_type):
        ref = str(
            getattr(value, "qualified_name", "")
            or getattr(value, "name", "") or "")
        if not re.fullmatch(
                r"[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+", ref):
            raise RuntimeError(f"artifact ref is not immutable/qualified: {ref!r}")
        digest = str(getattr(value, "digest", "") or "")
        if not digest or str(value.type) != expected_type:
            raise RuntimeError(f"artifact {ref!r} lacks exact type/digest")
        return ref, digest

    results_ref, results_digest = artifact_identity(logged, "dapo-results")
    api = wandb.Api()
    results_remote = api.artifact(results_ref)
    if str(results_remote.digest) != results_digest or \
            str(results_remote.type) != "dapo-results":
        raise RuntimeError("committed DAPO results receipt changed after upload")
    results_root = pathlib.Path("/workspace/dapo-results-completion-verify")
    if results_root.exists():
        shutil.rmtree(results_root)
    results_remote.download(root=str(results_root))

    final_binding = json.loads(
        (pathlib.Path(out_dir) / "final_artifact_binding.json").read_text())
    model_source = final_binding.get("source", {})
    model_ref = str(model_source.get("artifact_ref", ""))
    model_remote = api.artifact(model_ref)
    resolved_model_ref, model_digest = artifact_identity(model_remote, "model")
    if resolved_model_ref != model_ref or \
            model_digest != str(model_source.get("artifact_digest", "")):
        raise RuntimeError("final-model binding differs from committed model artifact")
    model_root = pathlib.Path("/workspace/dapo-model-completion-verify")
    if model_root.exists():
        shutil.rmtree(model_root)
    model_remote.download(root=str(model_root))

    checkpoint_name = re.sub(
        r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-central-r1-ckpt")
    checkpoint_remote = api.artifact(
        f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/"
        f"{checkpoint_name}:latest")
    checkpoint_ref, checkpoint_digest = artifact_identity(
        checkpoint_remote, dapo_evidence.DAPO_CHECKPOINT_ARTIFACT_TYPE)
    checkpoint_root = pathlib.Path("/workspace/dapo-checkpoint-completion-verify")
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    checkpoint_remote.download(root=str(checkpoint_root))

    completion = dapo_evidence.build_completion_binding(
        model_root=str(model_root), checkpoint_root=str(checkpoint_root),
        results_root=str(results_root),
        model_artifact_ref=model_ref, model_artifact_digest=model_digest,
        model_artifact_type=str(model_remote.type),
        model_artifact_metadata=dict(model_remote.metadata or {}),
        checkpoint_artifact_ref=checkpoint_ref,
        checkpoint_artifact_digest=checkpoint_digest,
        checkpoint_artifact_type=str(checkpoint_remote.type),
        checkpoint_artifact_metadata=dict(checkpoint_remote.metadata or {}),
        results_artifact_ref=results_ref,
        results_artifact_digest=results_digest,
        results_artifact_type=str(results_remote.type),
        results_artifact_metadata=dict(results_remote.metadata or {}),
    )
    completion_path = pathlib.Path(os.environ["PROVENANCE_DIR"]) / \
        dapo_evidence.DAPO_COMPLETION_BINDING_FILENAME
    completion_path.write_text(json.dumps(
        completion, indent=2, sort_keys=True) + "\n")
    campaign_metadata = {
        "completion_status": "complete",
        "method": dapo_evidence.DAPO_METHOD,
        "scientific_variant": dapo_evidence.SCIENTIFIC_VARIANT,
        "completion_binding_sha256": completion["binding_sha256"],
        "model_artifact_ref": model_ref,
        "model_artifact_digest": model_digest,
        "checkpoint_artifact_ref": checkpoint_ref,
        "checkpoint_artifact_digest": checkpoint_digest,
        "results_artifact_ref": results_ref,
        "results_artifact_digest": results_digest,
        "code_artifact_ref": completion["code_artifact_binding"]["source"][
            "artifact_ref"],
        "code_artifact_digest": completion["code_artifact_binding"]["source"][
            "artifact_digest"],
    }
    campaign = wandb.Artifact(
        re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-training-campaign"),
        type=dapo_evidence.DAPO_CAMPAIGN_ARTIFACT_TYPE,
        metadata=campaign_metadata,
    )
    campaign.add_file(
        str(completion_path),
        name=dapo_evidence.DAPO_COMPLETION_BINDING_FILENAME)
    logged_campaign = run.log_artifact(campaign, aliases=["latest"])
    logged_campaign.wait()
    campaign_ref, campaign_digest = artifact_identity(
        logged_campaign, dapo_evidence.DAPO_CAMPAIGN_ARTIFACT_TYPE)
    campaign_remote = api.artifact(campaign_ref)
    if str(campaign_remote.digest) != campaign_digest:
        raise RuntimeError("committed DAPO campaign receipt changed after upload")
    campaign_root = pathlib.Path("/workspace/dapo-campaign-completion-verify")
    if campaign_root.exists():
        shutil.rmtree(campaign_root)
    campaign_remote.download(root=str(campaign_root))
    dapo_evidence.validate_completion_campaign(
        campaign_root=str(campaign_root), model_root=str(model_root),
        checkpoint_root=str(checkpoint_root),
        results_root=str(results_root),
        campaign_artifact_type=str(campaign_remote.type),
        campaign_artifact_metadata=dict(campaign_remote.metadata or {}),
        model_artifact_ref=model_ref, model_artifact_digest=model_digest,
        model_artifact_type=str(model_remote.type),
        model_artifact_metadata=dict(model_remote.metadata or {}),
        checkpoint_artifact_ref=checkpoint_ref,
        checkpoint_artifact_digest=checkpoint_digest,
        checkpoint_artifact_type=str(checkpoint_remote.type),
        checkpoint_artifact_metadata=dict(checkpoint_remote.metadata or {}),
        results_artifact_ref=results_ref,
        results_artifact_digest=results_digest,
        results_artifact_type=str(results_remote.type),
        results_artifact_metadata=dict(results_remote.metadata or {}),
    )
    run.summary["training_campaign_artifact_ref"] = campaign_ref
    run.summary["training_campaign_artifact_digest"] = campaign_digest
    run.summary["final_model_artifact_ref"] = model_ref
    run.summary["final_model_artifact_digest"] = model_digest
    run.summary["step_100_checkpoint_artifact_ref"] = checkpoint_ref
    run.summary["step_100_checkpoint_artifact_digest"] = checkpoint_digest
    run.summary["results_artifact_ref"] = results_ref
    run.summary["results_artifact_digest"] = results_digest
    print(f"[dapo] signed completion campaign committed: {campaign_ref}", flush=True)

run.finish()
print(f"[dapo] durable evidence uploaded ({len(seen)} files)", flush=True)
PY
  upload_status=$?
  if [ "$status" -eq 0 ] && [ "$upload_status" -ne 0 ]; then
    echo "[dapo] FATAL: training succeeded but evidence upload failed" >&2
    exit "$upload_status"
  fi
  exit "$status"
}
trap upload_evidence EXIT

if [[ ! "$CODE_ARTIFACT" =~ ^[^/[:space:]]+/[^/[:space:]]+/[^/:[:space:]]+:v[0-9]+$ ]]; then
  echo "[dapo] FATAL: CODE_ARTIFACT must be fully qualified entity/project/name:vN" >&2
  exit 1
fi
if [ -n "$RESUME_CHECKPOINT_ARTIFACT" ] && \
   [[ ! "$RESUME_CHECKPOINT_ARTIFACT" =~ :v[0-9]+$ ]]; then
  echo "[dapo] FATAL: RESUME_CHECKPOINT_ARTIFACT must be immutable name:vN" >&2
  exit 1
fi
if { [ -n "$RESUME_CHECKPOINT_ARTIFACT" ] && [ -z "$RESUME_RESULTS_ARTIFACT" ]; } || \
   { [ -z "$RESUME_CHECKPOINT_ARTIFACT" ] && [ -n "$RESUME_RESULTS_ARTIFACT" ]; }; then
  echo "[dapo] FATAL: exact checkpoint and results artifacts are required together" >&2
  exit 1
fi
if [ -n "$RESUME_RESULTS_ARTIFACT" ] && \
   [[ ! "$RESUME_RESULTS_ARTIFACT" =~ :v[0-9]+$ ]]; then
  echo "[dapo] FATAL: RESUME_RESULTS_ARTIFACT must be immutable name:vN" >&2
  exit 1
fi
if [ "$MODEL" != "Qwen/Qwen3-1.7B" ] || [ "$STEPS" != "100" ] || \
   [ "$LR" != "0.000001" ] || [ "$SEED" != "0" ] || \
   [ "$NUM_GPUS" -lt 2 ]; then
  echo "[dapo] FATAL: original-paper contract drift" >&2
  exit 1
fi

# Download to a digest-scoped root and bind every member before update 1.
python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import re
import shutil
import wandb

ref = os.environ["CODE_ARTIFACT"]
artifact = wandb.Api().artifact(ref)
resolved_ref = str(
    getattr(artifact, "qualified_name", "")
    or getattr(artifact, "name", "") or "")
if not re.fullmatch(r"[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+", resolved_ref):
    raise RuntimeError(
        f"resolved code artifact is not an immutable qualified ref: {resolved_ref!r}")
if artifact.digest != os.environ["CODE_ARTIFACT_DIGEST"]:
    raise RuntimeError("resolved code artifact digest differs from remote job spec")
safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
root = pathlib.Path("/workspace/artifacts/code") / safe
if root.exists():
    shutil.rmtree(root)
root.parent.mkdir(parents=True, exist_ok=True)
artifact.download(root=str(root))
expected = sorted(item.name for item in artifact.files())
actual = sorted(path.relative_to(root).as_posix()
                for path in root.rglob("*") if path.is_file())
if actual != expected:
    raise RuntimeError("code artifact membership mismatch")
files = [{
    "relative_path": name,
    "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
    "size_bytes": (root / name).stat().st_size,
} for name in expected]
source = {"artifact_ref": resolved_ref, "artifact_digest": artifact.digest}
normalized = {"version": "wandb_artifact_download_v1", "source": source,
              "files": files}
manifest = {
    **normalized,
    "download_root": str(root.resolve()),
    "file_manifest_sha256": hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()).hexdigest(),
}
pathlib.Path(os.environ["CODE_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
pathlib.Path(os.environ["CODE_ROOT_MARKER"]).write_text(str(root.resolve()))
PY

CODE_ROOT="$(cat "$CODE_ROOT_MARKER")"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
VLLM_GPUS="$(seq -s, 1 $((NUM_GPUS - 1)))"
cd "$WORK"

# Revalidate the downloader's record with the exact implementation the
# trainer will use, and persist the machine-independent identity under the
# run directory before publishing it.
python3 - <<'PY'
import json
import os

from tmx_gpu import common

manifest_path = os.environ["CODE_MANIFEST"]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = common.validate_artifact_download_manifest(json.load(handle))
binding = common.stable_artifact_download_identity(manifest)
common.atomic_write_json(
    os.environ["CODE_BINDING"], binding)
print("[dapo] code manifest independently validated", flush=True)
PY

# Block before any learner update until the controller's authoritative
# `lep job get` spec is present as a RUN_NAME-unique immutable artifact.
python3 -m tmx_gpu.orchestrator_receipt resolve \
  --artifact-ref "$ORCHESTRATOR_RECEIPT_ARTIFACT" \
  --output-binding "$ORCHESTRATOR_RECEIPT_BINDING" \
  --download-root "$PROVENANCE_DIR/orchestrator-receipt" \
  --expected-run-name "$RUN_NAME" \
  --expected-container-image "$TRAINING_CONTAINER_IMAGE_IDENTITY" \
  --expected-code-artifact-ref "$(python3 -c 'import json,os; print(json.load(open(os.environ["CODE_MANIFEST"]))["source"]["artifact_ref"])')" \
  --expected-code-artifact-digest "$(python3 -c 'import json,os; print(json.load(open(os.environ["CODE_MANIFEST"]))["source"]["artifact_digest"])')" \
  --expected-command-sha256 "$ORCHESTRATOR_COMMAND_SHA256"

# Publish the exact, independently re-hashed code binding before update 1.
# This deliberately creates results:v0 even if model loading or generation
# fails before the first learner update.
python3 - <<'PY'
import json
import os
import re

import wandb

run_name = os.environ["RUN_NAME"]
manifest_path = os.environ["CODE_MANIFEST"]
binding_path = os.environ["CODE_BINDING"]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
run_id = re.sub(r"[^a-zA-Z0-9_-]", "-", f"{run_name}-evidence")[:120]
run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity=os.environ["WANDB_ENTITY"],
    name=f"{run_name}-evidence",
    job_type="artifacts",
    id=run_id,
    resume="allow",
)
artifact = wandb.Artifact(
    re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-results"),
    type="dapo-results",
    metadata={
        "evidence_stage": "pre_update_1",
        "scientific_variant": "original_paper_dapo_100_central_lr",
        "code_artifact_ref": manifest["source"]["artifact_ref"],
        "code_artifact_digest": manifest["source"]["artifact_digest"],
        "code_file_manifest_sha256": manifest["file_manifest_sha256"],
        "code_file_count": len(manifest["files"]),
    },
)
artifact.add_file(manifest_path, name="dapo_code_manifest.json")
artifact.add_file(binding_path, name="dapo_code_binding.json")
logged = run.log_artifact(artifact, aliases=["pre-update-1"])
logged.wait()
run.summary["pre_update_code_binding_published"] = True
run.summary["code_artifact_ref"] = manifest["source"]["artifact_ref"]
run.summary["code_artifact_digest"] = manifest["source"]["artifact_digest"]
run.summary["code_file_manifest_sha256"] = manifest["file_manifest_sha256"]
run.finish()
print("[dapo] pre-update-1 code binding published", flush=True)
PY

RESUME_ARGS=()
if [ -n "$RESUME_CHECKPOINT_ARTIFACT" ]; then
  # Restore the append-only metrics and trajectory evidence paired with the
  # checkpoint.  A checkpoint alone is intentionally insufficient for a
  # canonical resume because it would leave an incomplete training curve.
  python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import re
import shutil

import wandb

ref = os.environ["RESUME_RESULTS_ARTIFACT"]
artifact = wandb.Api().artifact(ref)
safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
root = pathlib.Path("/workspace/artifacts/resume-results") / safe
if root.exists():
    shutil.rmtree(root)
root.parent.mkdir(parents=True, exist_ok=True)
artifact.download(root=str(root))
expected = sorted(item.name for item in artifact.files())
actual_paths = sorted(path for path in root.rglob("*") if path.is_file())
actual = [path.relative_to(root).as_posix() for path in actual_paths]
if actual != expected:
    raise RuntimeError("resume results artifact membership mismatch")
if any(path.is_symlink() for path in root.rglob("*")):
    raise RuntimeError("resume results artifact contains a symlink")
required = {
    "dapo_code_binding.json",
    "metrics.jsonl",
    "explorer_trajectories.jsonl",
    "rollouts_all.jsonl",
}
if not required.issubset(set(actual)):
    raise RuntimeError(
        f"resume results artifact lacks required evidence: {sorted(required - set(actual))}")
if len([name for name in actual if name.endswith("trainer_state.json")]) != 1:
    raise RuntimeError("resume results artifact must contain exactly one trainer state")
files = [{
    "relative_path": path.relative_to(root).as_posix(),
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "size_bytes": path.stat().st_size,
} for path in actual_paths]
manifest = {
    "version": "wandb_results_download_v1",
    "source": {"artifact_ref": ref, "artifact_digest": artifact.digest},
    "artifact_type": artifact.type,
    "artifact_metadata": dict(artifact.metadata or {}),
    "files": files,
}
pathlib.Path(os.environ["RESUME_RESULTS_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
pathlib.Path(os.environ["RESUME_RESULTS_ROOT_MARKER"]).write_text(str(root.resolve()))
print("[dapo] exact prior metrics/trajectory artifact downloaded", flush=True)
PY

  # Download an exact immutable checkpoint, verify artifact membership and
  # its code binding before any checkpoint model bytes can be loaded, and let
  # grpo_gpu perform the full runtime/file-manifest compatibility check.
  python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import re
import shutil

import wandb

ref = os.environ["RESUME_CHECKPOINT_ARTIFACT"]
artifact = wandb.Api().artifact(ref)
safe = re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
root = pathlib.Path("/workspace/artifacts/resume") / safe
if root.exists():
    shutil.rmtree(root)
root.parent.mkdir(parents=True, exist_ok=True)
artifact.download(root=str(root))
expected = sorted(item.name for item in artifact.files())
actual = sorted(path.relative_to(root).as_posix()
                for path in root.rglob("*") if path.is_file())
if actual != expected:
    raise RuntimeError("resume artifact membership mismatch")
if any(path.is_symlink() for path in root.rglob("*")):
    raise RuntimeError("resume artifact contains a symlink")
state_paths = [path for path in root.rglob("trainer_state.json") if path.is_file()]
if len(state_paths) != 1:
    raise RuntimeError(
        f"resume artifact must contain exactly one trainer_state.json; got {len(state_paths)}")
state_path = state_paths[0]
checkpoint = state_path.parent.resolve()
with open(os.environ["CODE_MANIFEST"], encoding="utf-8") as handle:
    code = json.load(handle)
code_identity = {
    key: code[key]
    for key in ("version", "source", "files", "file_manifest_sha256")
}
with open(state_path, encoding="utf-8") as handle:
    state = json.load(handle)
if state.get("code_artifact_binding") != code_identity:
    raise RuntimeError("resume checkpoint code binding differs from current exact code")
if state.get("phase") != "central" or state.get("model") != "Qwen/Qwen3-1.7B":
    raise RuntimeError("resume checkpoint is not the original-paper DAPO route")
if state.get("rl_validity_mode") != "c8" or \
        state.get("optimizer_state_saved") is not True:
    raise RuntimeError("resume checkpoint lacks c8 optimizer-complete evidence")
step = int(state.get("step", -1))
if step < 1 or step > 100:
    raise RuntimeError(f"resume checkpoint step must be in [1,100], got {step}")
files = [{
    "relative_path": name,
    "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
    "size_bytes": (root / name).stat().st_size,
} for name in expected]
manifest = {
    "version": "wandb_checkpoint_download_v1",
    "source": {"artifact_ref": ref, "artifact_digest": artifact.digest},
    "artifact_type": artifact.type,
    "artifact_metadata": dict(artifact.metadata or {}),
    "files": files,
    "checkpoint_root": str(checkpoint),
    "trainer_step": step,
}
pathlib.Path(os.environ["RESUME_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
pathlib.Path(os.environ["RESUME_ROOT_MARKER"]).write_text(str(checkpoint))
print(f"[dapo] exact resume checkpoint bound at step {step}", flush=True)
PY

  # Only this pairing step may copy prior JSONLs into OUT_DIR. It proves the
  # results trainer state is byte-for-byte the selected checkpoint state,
  # checks both artifact types/metadata and full runtime/code/file manifests,
  # and requires all three evidence streams to reach the selected step.
  python3 -m tmx_gpu.dapo_evidence \
    --code-binding "$CODE_BINDING" \
    --checkpoint-root "$(cat "$RESUME_ROOT_MARKER")" \
    --checkpoint-manifest "$RESUME_MANIFEST" \
    --results-root "$(cat "$RESUME_RESULTS_ROOT_MARKER")" \
    --results-manifest "$RESUME_RESULTS_MANIFEST" \
    --output-dir "$OUT_DIR" \
    --pair-record "$RESUME_PAIR_RECORD"
  RESUME_ARGS=(--resume-from "$(cat "$RESUME_ROOT_MARKER")")

  # The resume bindings must themselves be durable before the first resumed
  # update, not merely present in the pod that is about to consume them.
  python3 - <<'PY'
import json
import os
import re

import wandb

run_name = os.environ["RUN_NAME"]
run_id = re.sub(r"[^a-zA-Z0-9_-]", "-", f"{run_name}-evidence")[:120]
run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity=os.environ["WANDB_ENTITY"],
    name=f"{run_name}-evidence",
    job_type="artifacts",
    id=run_id,
    resume="allow",
)
artifact = wandb.Artifact(
    re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-results"),
    type="dapo-results",
    metadata={
        "evidence_stage": "pre_resumed_update",
        "scientific_variant": "original_paper_dapo_100_central_lr",
        "resume_checkpoint_artifact": os.environ["RESUME_CHECKPOINT_ARTIFACT"],
        "resume_results_artifact": os.environ["RESUME_RESULTS_ARTIFACT"],
    },
)
for path in (
    os.environ["CODE_MANIFEST"],
    os.environ["CODE_BINDING"],
    os.environ["RESUME_MANIFEST"],
    os.environ["RESUME_RESULTS_MANIFEST"],
    os.environ["RESUME_PAIR_RECORD"],
):
    artifact.add_file(path, name=os.path.basename(path))
logged = run.log_artifact(artifact, aliases=["pre-resumed-update"])
logged.wait()
run.summary["pre_resumed_update_binding_published"] = True
run.finish()
print("[dapo] pre-resumed-update bindings published", flush=True)
PY
fi

STATUS=0
python3 "$CODE_ROOT/tmx_gpu/grpo_gpu.py" \
  --model "$MODEL" \
  --container-image-identity "$TRAINING_CONTAINER_IMAGE_IDENTITY" \
  --orchestrator-receipt-binding "$ORCHESTRATOR_RECEIPT_BINDING" \
  --orchestrator-command-sha256 "$ORCHESTRATOR_COMMAND_SHA256" \
  --orchestrator-receipt-artifact "$ORCHESTRATOR_RECEIPT_ARTIFACT" \
  --output-dir "$OUT_DIR" \
  --run-name "$RUN_NAME" \
  --phase central \
  --scientific-variant original_paper_dapo_100_central_lr \
  --steps "$STEPS" \
  --lr "$LR" \
  --lambda-novelty 0 \
  --prompts-per-step 4 \
  --num-generations 16 \
  --max-prompt-len 2048 \
  --max-completion-len 16384 \
  --completion-budget 0 \
  --loss-denominator-len 16384 \
  --soft-overlong-expected-len 13107 \
  --soft-overlong-cache-len 3277 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --updates-per-rollout 1 \
  --clip-epsilon-low 0.2 \
  --clip-epsilon-high 0.28 \
  --rl-validity-mode c8 \
  --dynamic-max-attempts 64 \
  --optimizer adamw \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --weight-decay 0 \
  --grad-clip 1.0 \
  --incorrect-novelty-scale 0.0 \
  --no-mask-truncated \
  --dataset dapo_math_17k \
  --max-train-examples 20000 \
  --seed "$SEED" \
  --round 1 \
  --global-step-offset 0 \
  --num-scouts 1 \
  --device cuda:0 \
  --param-dtype float32 \
  --vllm-gpus "$VLLM_GPUS" \
  --vllm-max-model-len 36864 \
  --gen-timeout 3600 \
  --gen-concurrency 8 \
  --max-consecutive-gen-failures 3 \
  --chunk-tokens 256 \
  --sync-dir "$WORK" \
  --save-every 25 \
  --keep-checkpoints 1 \
  --ckpt-artifact-every 25 \
  --probe-every 25 \
  --reload-every 10 \
  --code-source-manifest "$CODE_MANIFEST" \
  --metrics-jsonl "$OUT_DIR/metrics.jsonl" \
  "${RESUME_ARGS[@]}" \
  --wandb || STATUS=$?

exit "$STATUS"
