#!/usr/bin/env bash
set -euo pipefail

GCS_MODEL_PATH="${GCS_MODEL_PATH:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-}"
HF_REVISION="${HF_REVISION:-}"
MODEL_TAR_URL="${MODEL_TAR_URL:-}"
WORKER_ID="${WORKER_ID:?Must set WORKER_ID}"
LOCAL_MODEL_DIR_BASENAME="${LOCAL_MODEL_DIR_BASENAME:-expdis_vllm_model}"
LOCAL_MODEL_DIR="/tmp/${LOCAL_MODEL_DIR_BASENAME}_w${WORKER_ID}"
STAGING_DIR="${LOCAL_MODEL_DIR}.staging"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
VLLM_TPU_BUCKET_PADDING_GAP="${VLLM_TPU_BUCKET_PADDING_GAP:-}"
VLLM_TPU_MOST_MODEL_LEN="${VLLM_TPU_MOST_MODEL_LEN:-}"
VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-$HOME/.cache/vllm/xla_cache}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"
LOG_FILE="${LOG_FILE:-$HOME/vllm_reload_worker.log}"
MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-auto}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}"
VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-tp${TP_SIZE}_pda${VLLM_V1_USE_PREFILL_DECODE_ATTENTION}_mlen${MAX_MODEL_LEN}_mbt${MAX_NUM_BATCHED_TOKENS}}"
if [ -z "${TPU_VISIBLE_CHIPS:-}" ]; then
  if [ "$TP_SIZE" = "1" ]; then
    TPU_VISIBLE_CHIPS="0"
  else
    TPU_VISIBLE_CHIPS="0,1,2,3"
  fi
fi
TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
if [ -z "${TPU_CHIPS_PER_PROCESS_BOUNDS:-}" ]; then
  if [ "$TP_SIZE" = "1" ]; then
    TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1"
  else
    TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1"
  fi
fi
if [ -z "${TPU_CHIPS_PER_HOST_BOUNDS:-}" ]; then
  TPU_CHIPS_PER_HOST_BOUNDS="$TPU_CHIPS_PER_PROCESS_BOUNDS"
fi
TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"

if [ -z "$MODEL_TAR_URL" ] && [ -z "$GCS_MODEL_PATH" ] && { [ -z "$HF_REPO_ID" ] || [ -z "$HF_PATH_IN_REPO" ]; }; then
  echo "[reload-worker] ERROR: set MODEL_TAR_URL, or GCS_MODEL_PATH, or HF_REPO_ID+HF_PATH_IN_REPO" >&2
  exit 2
fi

exec > >(tee -a "$LOG_FILE") 2>&1
if [ -n "$MODEL_TAR_URL" ]; then
  MODEL_SOURCE="$MODEL_TAR_URL"
elif [ -n "$HF_REPO_ID" ] && [ -n "$HF_PATH_IN_REPO" ]; then
  MODEL_SOURCE="hf://${HF_REPO_ID}/${HF_PATH_IN_REPO}"
else
  MODEL_SOURCE="$GCS_MODEL_PATH"
fi
echo "[reload-worker] start $(date -Is) worker=$WORKER_ID model=$MODEL_SOURCE local=$LOCAL_MODEL_DIR"

ensure_vllm_tpu_runtime_stack() {
  [ "${EXPDIS_VLLM_ENSURE_JAX_STACK:-1}" = "0" ] && return 0
  local expected current
  expected="0.8.1 0.8.1 0.0.31 2.3.5"
  current="$("$VLLM_VENV_DIR/bin/python" - <<'PY'
import importlib.metadata as m
pkgs = ["jax", "jaxlib", "libtpu", "numpy"]
vals = []
for pkg in pkgs:
    try:
        vals.append(m.version(pkg))
    except Exception:
        vals.append("missing")
print(" ".join(vals))
PY
)"
  if [ "$current" != "$expected" ]; then
    echo "[reload-worker] repairing TPU runtime stack current=[$current] expected=[$expected]"
    "$VLLM_VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall \
      --extra-index-url https://storage.googleapis.com/libtpu-wheels/index.html \
      --extra-index-url https://storage.googleapis.com/jax-releases/libtpu_releases.html \
      "jax==0.8.1" "jaxlib==0.8.1" "libtpu==0.0.31" "numpy==2.3.5"
  else
    echo "[reload-worker] TPU runtime stack ok [$current]"
  fi
}

patch_vllm_tpu_rpa_kernel() {
  "$VLLM_VENV_DIR/bin/python" - <<'PY'
import re
import sysconfig
from pathlib import Path

purelib = Path(sysconfig.get_paths()["purelib"])
paths = [
    purelib / "tpu_inference/kernels/ragged_paged_attention/v3/kernel.py",
    purelib / "tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py",
]
wrapper_path = purelib / "tpu_inference/models/vllm/vllm_model_wrapper.py"

old_implicit = re.compile(
    r"bkv_x2_int32_ref = bkv_x2_ref\.bitcast\(jnp\.int32\)\.reshape\(\n"
    r"\s+\(2, -1, 8, 128\)\)"
)
old_product = re.compile(
    r"bkv_x2_int32_ref = bkv_x2_ref\.bitcast\(jnp\.int32\)\.reshape\(\n"
    r"\s+\(2, \(bkv_x2_ref\.shape\[1\] \* bkv_x2_ref\.shape\[2\] \* "
    r"bkv_x2_ref\.shape\[3\]\) // 8, 8, 128\)\)"
)
old_hardcoded = re.compile(
    r"bkv_x2_int32_ref = bkv_x2_ref\.bitcast\(jnp\.int32\)\.reshape\(\n"
    r"\s+\(2, bkv_x2_ref\.shape\[1\], 8, 128\)\)"
)
old_dynamic_mult = re.compile(
    r"bkv_x2_int32_ref = bkv_x2_ref\.bitcast\(jnp\.int32\)\.reshape\(\n"
    r"\s+\(2, bkv_x2_ref\.shape\[1\], bkv_x2_ref\.shape\[2\] \* "
    r"bkv_x2_ref\.shape\[3\], 128\)\)"
)
replacement = (
    "bkv_x2_int32_ref = bkv_x2_ref.bitcast(jnp.int32).reshape(\n"
    "      (2, bkv_x2_ref.shape[1], bkv_x2_ref.shape[2], 128))"
)

for path in paths:
    if not path.exists():
        print(f"[reload-worker] WARN missing TPU RPA kernel {path}")
        continue
    text = path.read_text()
    updated = old_implicit.sub(replacement, text)
    updated = old_product.sub(replacement, updated)
    updated = old_hardcoded.sub(replacement, updated)
    updated = old_dynamic_mult.sub(replacement, updated)
    if updated != text:
        path.write_text(updated)
        print(f"[reload-worker] patched TPU RPA kernel {path}")
    elif replacement in text:
        print(f"[reload-worker] TPU RPA kernel already patched {path}")
    else:
        print(f"[reload-worker] WARN TPU RPA kernel patch pattern not found {path}")

if wrapper_path.exists():
    text = wrapper_path.read_text()
    updated = text.replace('"post_spmd_conservative"', '"POST_SPMD"')
    if updated != text:
        wrapper_path.write_text(updated)
        print(f"[reload-worker] patched TPU collective matmul compiler options {wrapper_path}")
    elif '"POST_SPMD"' in text:
        print(f"[reload-worker] TPU collective matmul compiler options already patched {wrapper_path}")
    else:
        print(f"[reload-worker] WARN TPU collective matmul patch pattern not found {wrapper_path}")
else:
    print(f"[reload-worker] WARN missing TPU vLLM model wrapper {wrapper_path}")
PY
}

ensure_vllm_tpu_runtime_stack
patch_vllm_tpu_rpa_kernel || echo "[reload-worker] WARN TPU RPA patch failed; continuing"

pkill -9 -f "[g]cloud.*storage cp" 2>/dev/null || true
pkill -9 -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
pkill -9 -f "[v]llm serve" 2>/dev/null || true
tmux kill-session -t vllm_server 2>/dev/null || true
fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
rm -f /tmp/libtpu_lockfile 2>/dev/null || true
rm -rf /tmp/expdis_external_*_w"${WORKER_ID}" /tmp/expdis_vllm_reload_* 2>/dev/null || true
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
# vLLM reloads run on many workers at once. gcloud's default sliced/parallel
# downloader has intermittently failed on TPU VMs with missing *.gstmp files for
# model.safetensors. Use a single-stream copy here; reload is slower but safe.
export CLOUDSDK_STORAGE_PROCESS_COUNT="${CLOUDSDK_STORAGE_PROCESS_COUNT:-1}"
export CLOUDSDK_STORAGE_THREAD_COUNT="${CLOUDSDK_STORAGE_THREAD_COUNT:-1}"
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD="${CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD:-150G}"
hf_download_ok=0
if [ -n "$MODEL_TAR_URL" ]; then
  echo "[reload-worker] downloading internal model tar $MODEL_TAR_URL"
  "$VLLM_VENV_DIR/bin/python" - "$MODEL_TAR_URL" "$STAGING_DIR" <<'PY'
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

url = sys.argv[1]
staging = Path(sys.argv[2])
staging.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(prefix="expdis_vllm_reload_", suffix=".tar", delete=False) as f:
    tmp = Path(f.name)
try:
    with urllib.request.urlopen(url, timeout=7200) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    with tarfile.open(tmp, "r") as tar:
        tar.extractall(staging)
finally:
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
PY
  hf_download_ok=1
elif [ -n "$HF_REPO_ID" ] && [ -n "$HF_PATH_IN_REPO" ]; then
  export STAGING_DIR HF_REPO_ID HF_REPO_TYPE HF_PATH_IN_REPO HF_REVISION
  set +e
  "$VLLM_VENV_DIR/bin/python" - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE") or "dataset"
subpath = os.environ["HF_PATH_IN_REPO"].strip("/")
revision = os.environ.get("HF_REVISION") or None
staging = Path(os.environ["STAGING_DIR"])
tmp = staging.parent / (staging.name + ".hf_download")

shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
    allow_patterns=[subpath + "/*"],
    local_dir=str(tmp),
)
src = tmp / subpath
if not src.exists():
    matches = list(tmp.rglob("model.safetensors"))
    if not matches:
        raise FileNotFoundError(f"missing HF subpath {subpath!r} and model.safetensors")
    src = matches[0].parent
for child in src.iterdir():
    dst = staging / child.name
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(child), str(dst))
shutil.rmtree(tmp, ignore_errors=True)
PY
  hf_rc=$?
  set -e
  if [ "$hf_rc" -eq 0 ]; then
    hf_download_ok=1
  elif [ -n "$GCS_MODEL_PATH" ]; then
    echo "[reload-worker] WARN: HF download failed rc=$hf_rc; falling back to GCS $GCS_MODEL_PATH"
    rm -rf "$STAGING_DIR"
    mkdir -p "$STAGING_DIR"
  else
    echo "[reload-worker] ERROR: HF download failed rc=$hf_rc and no GCS fallback is configured" >&2
    exit "$hf_rc"
  fi
fi
if [ "$hf_download_ok" != "1" ]; then
  gcloud storage cp --recursive "${GCS_MODEL_PATH%/}/*" "$STAGING_DIR/"
fi
test -s "$STAGING_DIR/model.safetensors"
rm -rf "$LOCAL_MODEL_DIR"
mv "$STAGING_DIR" "$LOCAL_MODEL_DIR"
echo "[reload-worker] copied $(du -sh "$LOCAL_MODEL_DIR" | awk '{print $1}')"

sleep 3

if [ -z "${VLLM_XLA_CACHE_PATH:-}" ] || [ "$VLLM_XLA_CACHE_PATH" = "$HOME/.cache/vllm/xla_cache" ]; then
  safe_cache_tag="$(printf '%s' "$VLLM_XLA_CACHE_TAG" | tr -c 'A-Za-z0-9_.-' '_')"
  VLLM_XLA_CACHE_PATH="$HOME/.cache/vllm/xla_cache_${safe_cache_tag}_tp${TP_SIZE}"
fi

cat > /tmp/run_vllm_reloaded.sh <<RUNEOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="$VLLM_VENV_DIR/bin:\$PATH"
export PJRT_DEVICE="\${PJRT_DEVICE:-TPU}"
export TPU_SKIP_MDS_QUERY="\${TPU_SKIP_MDS_QUERY:-1}"
export TPU_VISIBLE_CHIPS="$TPU_VISIBLE_CHIPS"
export TPU_PROCESS_BOUNDS="$TPU_PROCESS_BOUNDS"
export TPU_CHIPS_PER_PROCESS_BOUNDS="$TPU_CHIPS_PER_PROCESS_BOUNDS"
export TPU_CHIPS_PER_HOST_BOUNDS="$TPU_CHIPS_PER_HOST_BOUNDS"
export TPU_HOST_BOUNDS="$TPU_HOST_BOUNDS"
export VLLM_USE_V1="$VLLM_USE_V1"
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION="$VLLM_V1_USE_PREFILL_DECODE_ATTENTION"
export TPU_WORKER_HOSTNAMES="\${TPU_WORKER_HOSTNAMES:-localhost}"
export MODEL_IMPL_TYPE="$MODEL_IMPL_TYPE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_XLA_CACHE_PATH="\${VLLM_XLA_CACHE_PATH:-$VLLM_XLA_CACHE_PATH}"
export XLA_PERSISTENT_CACHE_PATH="\${XLA_PERSISTENT_CACHE_PATH:-\$VLLM_XLA_CACHE_PATH}"
if [ -n "\${EXPDIS_VLLM_SAFE_XLA_FLAGS:-}" ]; then
  export XLA_FLAGS="\$EXPDIS_VLLM_SAFE_XLA_FLAGS"
else
  unset XLA_FLAGS
fi
if [ -n "$VLLM_TPU_BUCKET_PADDING_GAP" ]; then
  export VLLM_TPU_BUCKET_PADDING_GAP="$VLLM_TPU_BUCKET_PADDING_GAP"
else
  unset VLLM_TPU_BUCKET_PADDING_GAP
fi
if [ -n "$VLLM_TPU_MOST_MODEL_LEN" ]; then
  export VLLM_TPU_MOST_MODEL_LEN="$VLLM_TPU_MOST_MODEL_LEN"
else
  unset VLLM_TPU_MOST_MODEL_LEN
fi
mkdir -p "\$VLLM_XLA_CACHE_PATH"
extra_args=()
if [ "$DISABLE_LOG_REQUESTS" != "0" ]; then
  extra_args+=(--disable-log-requests)
fi
if [ "$VLLM_ENABLE_CHUNKED_PREFILL" != "0" ]; then
  extra_args+=(--enable-chunked-prefill)
else
  extra_args+=(--no-enable-chunked-prefill)
fi
exec "$VLLM_VENV_DIR/bin/vllm" serve "$LOCAL_MODEL_DIR" \\
  --host 0.0.0.0 --port "$PORT" \\
  --tensor-parallel-size "$TP_SIZE" \\
  --dtype bfloat16 \\
  --max-model-len "$MAX_MODEL_LEN" \\
  --max-num-seqs "$MAX_NUM_SEQS" \\
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \\
  --gpu-memory-utilization 0.85 \\
  --generation-config vllm \\
  --served-model-name "$SERVED_MODEL_NAME" \\
  "\${extra_args[@]}"
RUNEOF
chmod +x /tmp/run_vllm_reloaded.sh

tmux new-session -d -s vllm_server "bash /tmp/run_vllm_reloaded.sh > $HOME/vllm_server.log 2>&1"
for attempt in $(seq 1 120); do
  if curl -fsS --connect-timeout 3 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "[reload-worker] HEALTHY after ${attempt} attempts $(date -Is)"
    exit 0
  fi
  if ! tmux has-session -t vllm_server 2>/dev/null; then
    echo "[reload-worker] ERROR: tmux session exited"
    tail -200 "$HOME/vllm_server.log" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

echo "[reload-worker] ERROR: vLLM did not become healthy"
tail -200 "$HOME/vllm_server.log" 2>/dev/null || true
exit 1
