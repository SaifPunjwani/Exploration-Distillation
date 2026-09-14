#!/usr/bin/env bash
# Lepton entrypoint for the benchmark-independent SFT behavior gate.
#
# The outer job bootstrap must download CODE_ARTIFACT into /workspace/code.
# Required env:
#   RUN_NAME
#   ACCEPTED_ARTIFACTS   space-separated W&B result artifacts containing
#                        round_*/accepted.jsonl
# Optional env:
#   MODEL=Qwen/Qwen3-1.7B
#   MODEL_ARTIFACT       W&B model artifact to evaluate (empty evaluates MODEL)
#   NUM_GPUS=all visible, MAX_MODEL_LEN=36864
#   NUM_PROMPTS=32, SAMPLES_PER_PROMPT=2, REQUIRE_PASS=1
#   MIN_TERMINATION_RATE=.80 MAX_CLIPPED_RATE=.20 MIN_VALID_BOXED_RATE=.90
#   MAX_MEDIAN_TOKEN_FRACTION=.75 MAX_REPETITION_RATE=.109375
#   MAX_UNCLOSED_THINK_RATE=.20 MAX_GENERATION_ERROR_RATE=.05
set -euo pipefail

exec > >(tee -a /workspace/job.log) 2>&1

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MODEL_ARTIFACT="${MODEL_ARTIFACT:-}"
RUN_NAME="${RUN_NAME:?RUN_NAME required}"
ACCEPTED_ARTIFACTS="${ACCEPTED_ARTIFACTS:?ACCEPTED_ARTIFACTS required}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-32768}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-2}"
REQUIRE_PASS="${REQUIRE_PASS:-1}"
SELECTION_SEED="${SELECTION_SEED:-20260713}"
GENERATION_SEED="${GENERATION_SEED:-20260713}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_TERMINATION_RATE="${MIN_TERMINATION_RATE:-0.80}"
MAX_CLIPPED_RATE="${MAX_CLIPPED_RATE:-0.20}"
MIN_VALID_BOXED_RATE="${MIN_VALID_BOXED_RATE:-0.90}"
MAX_MEDIAN_TOKEN_FRACTION="${MAX_MEDIAN_TOKEN_FRACTION:-0.75}"
# Frozen after the Base calibration run, before checkpoint inspection: the
# paper-prompt Base control had 7/64 repeated generations.  Equality at the
# Base rate is admitted.  Override explicitly for unrelated diagnostic uses.
MAX_REPETITION_RATE="${MAX_REPETITION_RATE:-0.109375}"
MAX_UNCLOSED_THINK_RATE="${MAX_UNCLOSED_THINK_RATE:-0.20}"
MAX_GENERATION_ERROR_RATE="${MAX_GENERATION_ERROR_RATE:-0.05}"
PORT_BASE="${PORT_BASE:-8000}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-600}"
if ! [[ "$SERVER_READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[sft-gate-job] FATAL: SERVER_READY_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 1
fi
# Keep vLLM's internal engine timeout at least as long as the launcher's
# identity-checked health wait. Cold Base-model downloads can legitimately
# take more than ten minutes when parallel servers share a fresh HF cache.
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$SERVER_READY_TIMEOUT_SECONDS}"
export VLLM_ENGINE_READY_TIMEOUT_S
SERVER_READY_POLL_COUNT=$(( (SERVER_READY_TIMEOUT_SECONDS + 4) / 5 ))
OUT_ROOT="/workspace/sft_gate/bootstrap-${RUN_NAME//[^a-zA-Z0-9_.-]/-}"
export RUN_NAME OUT_ROOT

upload_durable_state() {
  status="${1:-$?}"
  echo "[sft-gate-job] exit status=$status — uploading durable state"
  python3 - "$status" <<'PY' || true
import glob
import json
import os
import re
import sys
import wandb

run_name = os.environ.get("RUN_NAME", "unknown")
artifact_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{run_name}-sft-gate")
run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "two-model-explore-gpu"),
    entity=os.environ.get("WANDB_ENTITY") or None,
    name=f"sft-gate-{run_name}",
    job_type="sft-behavior-gate",
    id=re.sub(r"[^a-zA-Z0-9_-]", "-", f"sft-gate-{run_name}")[:120],
    resume="allow",
)
artifact = wandb.Artifact(artifact_name, type="sft-gate-results")
for path in ["/workspace/job.log"] + sorted(glob.glob("/workspace/vllm_*.log")):
    if os.path.isfile(path):
        artifact.add_file(path, name=os.path.basename(path))
out_root = os.environ.get("OUT_ROOT", "/workspace/sft_gate")
if os.path.isdir(out_root):
    artifact.add_dir(out_root, name="sft_gate")
aggregates = sorted(glob.glob(os.path.join(out_root, "**", "aggregate_metrics.json"), recursive=True))
if aggregates:
    with open(aggregates[-1], encoding="utf-8") as handle:
        aggregate = json.load(handle)
    gate = aggregate.get("health_gate", {})
    run.summary["health_gate_passed"] = bool(gate.get("passed"))
    run.summary["metric_contract_version"] = aggregate.get("metric_contract_version")
    for mode, metrics in aggregate.get("metrics_by_prompt_mode", {}).items():
        for key in (
            "termination_rate", "clipped_rate", "valid_boxed_answer_rate",
            "median_completion_tokens", "repetition_rate", "unclosed_think_rate",
            "generation_error_rate", "correctness_rate_diagnostic_only",
        ):
            if key in metrics:
                run.summary[f"{mode}/{key}"] = metrics[key]
run.summary["exit_status"] = int(sys.argv[1])
run.log_artifact(artifact)
run.finish()
print("[sft-gate-job] durable state uploaded", flush=True)
PY
}
SERVER_PIDS=()
cleanup_servers() {
  for pid in "${SERVER_PIDS[@]:-}"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
  done
}
on_exit() {
  status=$?
  cleanup_servers
  upload_durable_state "$status"
}
trap on_exit EXIT

mkdir -p /workspace/code /workspace/artifacts /workspace/evidence_manifests "$OUT_ROOT"
cd /workspace/code
test -f tmx_gpu/sft_gate_gpu.py || {
  echo "[sft-gate-job] FATAL: code artifact lacks sft_gate_gpu.py" >&2
  exit 1
}

if [ -n "$MODEL_ARTIFACT" ]; then
  echo "[sft-gate-job] downloading model artifact $MODEL_ARTIFACT"
  python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import re
import shutil
import wandb

ref = os.environ["MODEL_ARTIFACT"]
artifact = wandb.Api().artifact(ref)
safe_digest = re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
root = pathlib.Path("/workspace/artifacts/model") / safe_digest
if root.exists():
    shutil.rmtree(root)
root.parent.mkdir(parents=True, exist_ok=True)
artifact.download(root=str(root))
expected = sorted(file.name for file in artifact.files())
actual = sorted(
    path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
assert actual == expected, f"model artifact membership mismatch: {actual} vs {expected}"
files = [{
    "relative_path": name,
    "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
    "size_bytes": (root / name).stat().st_size,
} for name in expected]
source = {"artifact_ref": ref, "artifact_digest": artifact.digest}
normalized = {"version": "wandb_artifact_download_v1", "source": source, "files": files}
payload = {
    **normalized,
    "download_root": str(root.resolve()),
    "file_manifest_sha256": hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()).hexdigest(),
}
manifest_path = pathlib.Path("/workspace/evidence_manifests/model.json")
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
pathlib.Path("/workspace/model_path.txt").write_text(str(root.resolve()))
pathlib.Path("/workspace/model_source_digest.txt").write_text(artifact.digest)
PY
  MODEL="$(cat /workspace/model_path.txt)"
  test -f "$MODEL/config.json" || {
    echo "[sft-gate-job] FATAL: model artifact lacks config.json" >&2
    exit 1
  }
  MODEL_SOURCE_REF="$MODEL_ARTIFACT"
  MODEL_SOURCE_DIGEST="$(cat /workspace/model_source_digest.txt)"
  MODEL_FILES_MANIFEST=/workspace/evidence_manifests/model.json
else
  MODEL_SOURCE_REF=""
  MODEL_SOURCE_DIGEST=""
  MODEL_FILES_MANIFEST=""
fi

export ACCEPTED_ARTIFACTS
mapfile -t ACCEPTED_PATHS < <(python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import re
import shutil
import wandb

paths = []
download_manifests = []
for index, ref in enumerate(os.environ["ACCEPTED_ARTIFACTS"].split()):
    artifact = wandb.Api().artifact(ref)
    safe_digest = re.sub(r"[^a-zA-Z0-9_.-]", "-", artifact.digest)
    root = pathlib.Path(f"/workspace/artifacts/accepted/{index}-{safe_digest}")
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(root))
    expected = sorted(file.name for file in artifact.files())
    actual = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    if actual != expected:
        raise RuntimeError(f"accepted artifact membership mismatch for {ref}")
    hits = [str((root / name).resolve()) for name in expected
            if name.endswith("accepted.jsonl")]
    if not hits:
        raise RuntimeError(f"accepted.jsonl not found in {ref}")
    paths.extend(hits)
    files = [{
        "relative_path": name,
        "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        "size_bytes": (root / name).stat().st_size,
    } for name in expected]
    source = {"artifact_ref": ref, "artifact_digest": artifact.digest}
    normalized = {"version": "wandb_artifact_download_v1", "source": source,
                  "files": files}
    download_manifests.append({
        **normalized,
        "download_root": str(root.resolve()),
        "file_manifest_sha256": hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode()).hexdigest(),
    })
pathlib.Path("/workspace/accepted_source_manifest.json").write_text(
    json.dumps({"version": "wandb_accepted_sources_v2",
                "download_manifests": download_manifests},
               indent=2, sort_keys=True) + "\n")
for path in paths:
    print(path)
PY
)
if [ "${#ACCEPTED_PATHS[@]}" -eq 0 ]; then
  echo "[sft-gate-job] FATAL: no accepted JSONLs resolved" >&2
  exit 1
fi

echo "[sft-gate-job] $(date -u) model=$MODEL gpus=$NUM_GPUS run=$RUN_NAME"
nvidia-smi -L || true

if [ -n "$MODEL_SOURCE_DIGEST" ]; then
  SERVED_MODEL_NAME="sft-gate-${MODEL_SOURCE_DIGEST//[^a-zA-Z0-9]/-}"
else
  SERVED_MODEL_NAME="sft-gate-base-${RUN_NAME//[^a-zA-Z0-9]/-}"
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:0:120}"
# Never append into a gate bundle left by an earlier process. Namespace the
# output by the exact model digest (or explicit Base identity) and run name,
# then start from an empty directory. The EXIT trap follows the updated export.
GATE_SCOPE="${MODEL_SOURCE_DIGEST:-base-${MODEL//[^a-zA-Z0-9_.-]/-}}-${RUN_NAME//[^a-zA-Z0-9_.-]/-}"
OUT_ROOT="/workspace/sft_gate/${GATE_SCOPE:0:220}"
export OUT_ROOT
rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  port=$((PORT_BASE + gpu))
  # A responder left by an earlier job must never satisfy this gate's health
  # check. Kill the process owning the port when possible, then prove the port
  # is free before launching and track the exact child PID.
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  fi
  port_free=0
  for _ in $(seq 1 20); do
    if python3 - "$port" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    s.close()
PY
    then
      port_free=1
      break
    fi
    sleep 1
  done
  if [ "$port_free" != "1" ]; then
    echo "[sft-gate-job] FATAL: stale process still owns port $port" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME" \
    --port "$port" --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 > "/workspace/vllm_$gpu.log" 2>&1 &
  pid=$!
  SERVER_PIDS+=("$pid")
  echo "[sft-gate-job] launched gpu=$gpu pid=$pid port=$port served=$SERVED_MODEL_NAME"
done

ENDPOINTS=""
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  ENDPOINTS+="http://127.0.0.1:$((PORT_BASE + gpu))/v1,"
done
ENDPOINTS="${ENDPOINTS%,}"

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  healthy=0
  for _ in $(seq 1 "$SERVER_READY_POLL_COUNT"); do
    pid="${SERVER_PIDS[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[sft-gate-job] FATAL: newly launched vLLM pid $pid exited" >&2
      tail -100 "/workspace/vllm_$gpu.log" || true
      exit 1
    fi
    if curl -sf "http://127.0.0.1:$((PORT_BASE + gpu))/health" >/dev/null 2>&1; then
      if curl -sf "http://127.0.0.1:$((PORT_BASE + gpu))/v1/models" | \
          python3 -c 'import json,sys; expected=sys.argv[1]; data=json.load(sys.stdin); ids=[x.get("id") for x in data.get("data",[])]; raise SystemExit(0 if ids == [expected] else 1)' \
          "$SERVED_MODEL_NAME"; then
        healthy=1
        break
      fi
    fi
    sleep 5
  done
  if [ "$healthy" != "1" ]; then
    echo "[sft-gate-job] FATAL: vLLM server $gpu did not become healthy" >&2
    tail -100 "/workspace/vllm_$gpu.log" || true
    exit 1
  fi
done

ACCEPTED_ARGS=()
for path in "${ACCEPTED_PATHS[@]}"; do
  ACCEPTED_ARGS+=(--accepted "$path")
done
REQUIRE_PASS_ARG=()
if [ "$REQUIRE_PASS" = "1" ]; then
  REQUIRE_PASS_ARG=(--require-pass)
fi
MODEL_MANIFEST_ARGS=()
if [ -n "$MODEL_FILES_MANIFEST" ]; then
  MODEL_MANIFEST_ARGS=(--model-files-manifest "$MODEL_FILES_MANIFEST")
fi

python3 tmx_gpu/sft_gate_gpu.py \
  --endpoints "$ENDPOINTS" \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --model-source-ref "$MODEL_SOURCE_REF" \
  --model-source-digest "$MODEL_SOURCE_DIGEST" \
  "${MODEL_MANIFEST_ARGS[@]}" \
  --accepted-source-manifest /workspace/accepted_source_manifest.json \
  "${ACCEPTED_ARGS[@]}" \
  --num-prompts "$NUM_PROMPTS" \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --selection-seed "$SELECTION_SEED" \
  --generation-seed "$GENERATION_SEED" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
  --min-termination-rate "$MIN_TERMINATION_RATE" \
  --max-clipped-rate "$MAX_CLIPPED_RATE" \
  --min-valid-boxed-rate "$MIN_VALID_BOXED_RATE" \
  --max-median-token-fraction "$MAX_MEDIAN_TOKEN_FRACTION" \
  --max-repetition-rate "$MAX_REPETITION_RATE" \
  --max-unclosed-think-rate "$MAX_UNCLOSED_THINK_RATE" \
  --max-generation-error-rate "$MAX_GENERATION_ERROR_RATE" \
  --server-max-model-len "$MAX_MODEL_LEN" \
  --out-dir "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  "${REQUIRE_PASS_ARG[@]}"
