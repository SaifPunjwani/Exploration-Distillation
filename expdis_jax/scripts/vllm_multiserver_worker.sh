#!/usr/bin/env bash
# Launch one TP=1 vLLM server per selected TPU chip on a single TPU VM.
# If GCS_MODEL_PATH or HF_REPO_ID+HF_PATH_IN_REPO is provided, the model bundle
# is staged once per host and shared by all chip-local vLLM processes.
set -euo pipefail

WORKER_ID="${WORKER_ID:?Must set WORKER_ID}"
CHIPS="${CHIPS:-0 1 2 3}"
PORT_BASE="${PORT_BASE:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
GCS_MODEL_PATH="${GCS_MODEL_PATH:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_PATH_IN_REPO="${HF_PATH_IN_REPO:-}"
HF_REVISION="${HF_REVISION:-}"
LOCAL_MODEL_DIR_BASENAME="${LOCAL_MODEL_DIR_BASENAME:-expdis_vllm_model}"
LOCAL_MODEL_DIR="/tmp/${LOCAL_MODEL_DIR_BASENAME}_w${WORKER_ID}"
STAGING_DIR="${LOCAL_MODEL_DIR}.staging"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/vllm_miniforge}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-22528}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_TPU_BUCKET_PADDING_GAP="${VLLM_TPU_BUCKET_PADDING_GAP:-4096}"
VLLM_TPU_MOST_MODEL_LEN="${VLLM_TPU_MOST_MODEL_LEN:-$MAX_MODEL_LEN}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}"
VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-rpa_patch_v2_mlen${MAX_MODEL_LEN}_mbt${MAX_NUM_BATCHED_TOKENS}_pda${VLLM_V1_USE_PREFILL_DECODE_ATTENTION}}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"
MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-auto}"
EXPDIS_VLLM_SUPERVISE="${EXPDIS_VLLM_SUPERVISE:-1}"
EXPDIS_VLLM_RESTART_SLEEP_SECONDS="${EXPDIS_VLLM_RESTART_SLEEP_SECONDS:-5}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-5}"
LOG_FILE="${LOG_FILE:-$HOME/vllm_multiserver_worker_${WORKER_ID}.log}"

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[vllm-multi-worker] start $(date -Is) worker=$WORKER_ID chips=[$CHIPS] port_base=$PORT_BASE"

install_vllm_tpu_env() {
  echo "[vllm-multi-worker] installing vLLM TPU env at $VLLM_VENV_DIR"
  rm -rf "$VLLM_VENV_DIR" 2>/dev/null || true
  if [ ! -d "$MINIFORGE_DIR" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge_vllm.sh
    bash /tmp/miniforge_vllm.sh -b -p "$MINIFORGE_DIR"
  fi
  # shellcheck disable=SC1091
  source "$MINIFORGE_DIR/bin/activate"
  conda create -y -p "$VLLM_VENV_DIR" python=3.11
  conda activate "$VLLM_VENV_DIR"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install \
    --extra-index-url https://storage.googleapis.com/libtpu-wheels/index.html \
    --extra-index-url https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    'vllm-tpu==0.13.3' datasets wandb transformers sentencepiece tiktoken
}

if [ ! -x "$VLLM_VENV_DIR/bin/vllm" ]; then
  install_vllm_tpu_env
fi

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
    echo "[vllm-multi-worker] repairing TPU runtime stack current=[$current] expected=[$expected]"
    "$VLLM_VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall \
      --extra-index-url https://storage.googleapis.com/libtpu-wheels/index.html \
      --extra-index-url https://storage.googleapis.com/jax-releases/libtpu_releases.html \
      "jax==0.8.1" "jaxlib==0.8.1" "libtpu==0.0.31" "numpy==2.3.5"
  else
    echo "[vllm-multi-worker] TPU runtime stack ok [$current]"
  fi
}

patch_vllm_tpu_transformers_compat() {
  "$VLLM_VENV_DIR/bin/python" - <<'PY'
import py_compile
import sysconfig
from pathlib import Path

purelib = Path(sysconfig.get_paths()["purelib"])
replacements = {
    purelib / "tpu_inference/models/jax/qwen2.py": {
        "from transformers import Qwen2Config, modeling_flax_utils":
            "from transformers import Qwen2Config\nfrom tpu_inference.layers.jax.layers import modeling_flax_utils",
    },
    purelib / "tpu_inference/models/jax/qwen2_5_vl.py": {
        "from transformers import modeling_flax_utils\n":
            "from tpu_inference.layers.jax.layers import modeling_flax_utils\n",
    },
    purelib / "tpu_inference/models/jax/llama3.py": {
        "from transformers import LlamaConfig, modeling_flax_utils":
            "from transformers import LlamaConfig\nfrom tpu_inference.layers.jax.layers import modeling_flax_utils",
    },
}

changed = []
for path, path_replacements in replacements.items():
    if not path.exists():
        print(f"[vllm-multi-worker] WARN: missing tpu_inference compat target {path}")
        continue
    text = path.read_text()
    updated = text
    for old, new in path_replacements.items():
        updated = updated.replace(old, new)
    if updated != text:
        backup = path.with_suffix(path.suffix + ".tmx_transformers_compat_bak")
        if not backup.exists():
            backup.write_text(text)
        path.write_text(updated)
        changed.append(str(path))
    py_compile.compile(str(path), doraise=True)

transformers_base = purelib / "vllm/model_executor/models/transformers/base.py"
if transformers_base.exists():
    text = transformers_base.read_text()
    original_text = text
    marker = "# TMX text-only Gemma compatibility: ignore unsupported audio tower weights."
    suffix_line = "            self.ignore_unexpected_suffixes.append(\".layer_scalar\")\n"
    if marker not in text:
        needle = (
            "        if self.quant_config:\n"
            "            quant_method_name = self.quant_config.get_name()\n"
        )
        patch = (
            f"        {marker}\n"
            "        if getattr(self.config, \"model_type\", \"\") in {\"gemma4\", \"gemma3n\"}:\n"
            "            self.ignore_unexpected_prefixes.extend([\n"
            "                \"model.audio_tower.\",\n"
            "                \"audio_tower.\",\n"
            "            ])\n"
            "            self.ignore_unexpected_suffixes.append(\".layer_scalar\")\n\n"
        )
        if needle not in text:
            raise RuntimeError(f"cannot patch {transformers_base}: quant_config marker not found")
        text = text.replace(needle, patch + needle)
    elif suffix_line not in text:
        suffix_needle = (
            "            self.ignore_unexpected_prefixes.extend([\n"
            "                \"model.audio_tower.\",\n"
            "                \"audio_tower.\",\n"
            "            ])\n"
        )
        if suffix_needle not in text:
            raise RuntimeError(f"cannot patch {transformers_base}: Gemma audio prefix block not found")
        text = text.replace(suffix_needle, suffix_needle + suffix_line)
    if text != original_text:
        backup = transformers_base.with_suffix(transformers_base.suffix + ".tmx_gemma_audio_compat_bak")
        if not backup.exists():
            backup.write_text(original_text)
        transformers_base.write_text(text)
        changed.append(str(transformers_base))
    py_compile.compile(str(transformers_base), doraise=True)

for module in (
    "tpu_inference.models.jax.qwen2",
    "tpu_inference.models.jax.qwen2_5_vl",
    "tpu_inference.models.jax.llama3",
):
    __import__(module)

if changed:
    print("[vllm-multi-worker] patched transformers modeling_flax_utils compatibility: " + ", ".join(changed))
else:
    print("[vllm-multi-worker] transformers modeling_flax_utils compatibility already ok")
PY
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
        print(f"[vllm-multi-worker] WARN: missing TPU RPA kernel {path}")
        continue
    text = path.read_text()
    updated = old_implicit.sub(replacement, text)
    updated = old_product.sub(replacement, updated)
    updated = old_hardcoded.sub(replacement, updated)
    updated = old_dynamic_mult.sub(replacement, updated)
    if updated != text:
        path.write_text(updated)
        print(f"[vllm-multi-worker] patched TPU RPA kernel {path}")
    elif replacement in text:
        print(f"[vllm-multi-worker] TPU RPA kernel already patched {path}")
    else:
        print(f"[vllm-multi-worker] WARN: TPU RPA kernel patch pattern not found {path}")

if wrapper_path.exists():
    text = wrapper_path.read_text()
    updated = text.replace('"post_spmd_conservative"', '"POST_SPMD"')
    if updated != text:
        wrapper_path.write_text(updated)
        print(f"[vllm-multi-worker] patched TPU collective matmul compiler options {wrapper_path}")
    elif '"POST_SPMD"' in text:
        print(f"[vllm-multi-worker] TPU collective matmul compiler options already patched {wrapper_path}")
    else:
        print(f"[vllm-multi-worker] WARN: TPU collective matmul patch pattern not found {wrapper_path}")
else:
    print(f"[vllm-multi-worker] WARN: missing TPU vLLM model wrapper {wrapper_path}")
PY
}

ensure_vllm_tpu_runtime_stack
patch_vllm_tpu_transformers_compat || echo "[vllm-multi-worker] WARN: transformers compatibility patch failed; continuing"
patch_vllm_tpu_rpa_kernel || echo "[vllm-multi-worker] WARN: TPU RPA patch failed; continuing"

resolve_cached_hf_model() {
  MODEL_TO_RESOLVE="$1" "$VLLM_VENV_DIR/bin/python" - <<'PY'
import os
import sys
from pathlib import Path

model = os.environ["MODEL_TO_RESOLVE"].strip()
if not model:
    raise SystemExit(0)
if Path(model).exists():
    print(model)
    raise SystemExit(0)
if "/" not in model or model.startswith("."):
    raise SystemExit(0)

try:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model, local_files_only=True)
except Exception as exc:
    print(f"[vllm-multi-worker] WARN: local HF cache miss for {model}: {exc}", file=sys.stderr)
    raise SystemExit(0)
print(path)
PY
}

pkill -9 -f "[g]cloud.*storage cp" 2>/dev/null || true
pkill -9 -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
pkill -9 -f "[v]llm serve" 2>/dev/null || true
(tmux ls 2>/dev/null || true) | awk -F: '/^vllm_server/ {print $1}' | while read -r session; do
  [ -n "$session" ] && tmux kill-session -t "$session" 2>/dev/null || true
done
rm -f /tmp/libtpu_lockfile 2>/dev/null || true
sleep 2

model_arg="$MODEL"
if [ -n "$GCS_MODEL_PATH" ] || { [ -n "$HF_REPO_ID" ] && [ -n "$HF_PATH_IN_REPO" ]; }; then
  rm -rf "$STAGING_DIR"
  mkdir -p "$STAGING_DIR"
  export CLOUDSDK_STORAGE_PROCESS_COUNT="${CLOUDSDK_STORAGE_PROCESS_COUNT:-1}"
  export CLOUDSDK_STORAGE_THREAD_COUNT="${CLOUDSDK_STORAGE_THREAD_COUNT:-1}"
  export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD="${CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD:-150G}"

  hf_download_ok=0
  if [ -n "$HF_REPO_ID" ] && [ -n "$HF_PATH_IN_REPO" ]; then
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
      echo "[vllm-multi-worker] WARN: HF download failed rc=$hf_rc; falling back to GCS"
      rm -rf "$STAGING_DIR"
      mkdir -p "$STAGING_DIR"
    else
      echo "[vllm-multi-worker] ERROR: HF download failed rc=$hf_rc and no GCS fallback is configured" >&2
      exit "$hf_rc"
    fi
  fi
  if [ "$hf_download_ok" != "1" ]; then
    gcloud storage cp --recursive "${GCS_MODEL_PATH%/}/*" "$STAGING_DIR/"
  fi
  test -s "$STAGING_DIR/model.safetensors"
  rm -rf "$LOCAL_MODEL_DIR"
  mv "$STAGING_DIR" "$LOCAL_MODEL_DIR"
  echo "[vllm-multi-worker] copied $(du -sh "$LOCAL_MODEL_DIR" | awk '{print $1}')"
  model_arg="$LOCAL_MODEL_DIR"
else
  cache_warn="/tmp/vllm_cached_model_${WORKER_ID}.warn"
  cached_model="$(resolve_cached_hf_model "$MODEL" 2>"$cache_warn" || true)"
  if [ -n "$cached_model" ] && [ -d "$cached_model" ]; then
    model_arg="$cached_model"
    echo "[vllm-multi-worker] using local HF snapshot for $MODEL -> $model_arg"
  elif [ -s "$cache_warn" ]; then
    cat "$cache_warn"
  fi
fi

for chip in $CHIPS; do
  port=$((PORT_BASE + chip))
  session="vllm_server_chip_${chip}"
  run_script="/tmp/run_vllm_chip_${chip}.sh"
  safe_cache_tag="$(printf '%s' "$VLLM_XLA_CACHE_TAG" | tr -c 'A-Za-z0-9_.-' '_')"
  cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="$VLLM_VENV_DIR/bin:\$PATH"
export PJRT_DEVICE=TPU
export TPU_SKIP_MDS_QUERY=1
export TPU_VISIBLE_CHIPS="$chip"
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_WORKER_HOSTNAMES=localhost
export VLLM_USE_V1="$VLLM_USE_V1"
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION="$VLLM_V1_USE_PREFILL_DECODE_ATTENTION"
export EXPDIS_VLLM_SUPERVISE="$EXPDIS_VLLM_SUPERVISE"
export EXPDIS_VLLM_RESTART_SLEEP_SECONDS="$EXPDIS_VLLM_RESTART_SLEEP_SECONDS"
export MODEL_IMPL_TYPE="$MODEL_IMPL_TYPE"
if [ -d "$model_arg" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
export VLLM_XLA_CACHE_PATH="$HOME/.cache/vllm/xla_cache_${safe_cache_tag}_chip_${chip}"
export XLA_PERSISTENT_CACHE_PATH="\$VLLM_XLA_CACHE_PATH"
if [ -n "\${EXPDIS_VLLM_SAFE_XLA_FLAGS:-}" ]; then
  export XLA_FLAGS="\$EXPDIS_VLLM_SAFE_XLA_FLAGS"
else
  unset XLA_FLAGS
fi
export VLLM_TPU_BUCKET_PADDING_GAP="$VLLM_TPU_BUCKET_PADDING_GAP"
export VLLM_TPU_MOST_MODEL_LEN="$VLLM_TPU_MOST_MODEL_LEN"
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
if [ -n "$MODEL_IMPL_TYPE" ] && [ "$MODEL_IMPL_TYPE" != "auto" ]; then
  extra_args+=(--model-impl "$MODEL_IMPL_TYPE")
fi
while true; do
  rc=0
  "$VLLM_VENV_DIR/bin/vllm" serve "$model_arg" \\
    --host 0.0.0.0 --port "$port" \\
    --tensor-parallel-size 1 \\
    --dtype bfloat16 \\
    --max-model-len "$MAX_MODEL_LEN" \\
    --max-num-seqs "$MAX_NUM_SEQS" \\
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \\
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \\
    --generation-config vllm \\
    --served-model-name "$SERVED_MODEL_NAME" \\
    "\${extra_args[@]}" || rc="\$?"
  rc="\${rc:-0}"
  echo "[vllm-multi-worker] vLLM exited chip=$chip port=$port rc=\$rc supervise=\${EXPDIS_VLLM_SUPERVISE:-1} at \$(date -Is)"
  if [ "\${EXPDIS_VLLM_SUPERVISE:-1}" = "0" ]; then
    exit "\$rc"
  fi
  sleep "\${EXPDIS_VLLM_RESTART_SLEEP_SECONDS:-5}"
done
EOF
  chmod +x "$run_script"
  tmux new-session -d -s "$session" "bash '$run_script' > '$HOME/vllm_server_chip_${chip}.log' 2>&1"
  echo "[vllm-multi-worker] launched chip=$chip port=$port session=$session"

  chip_deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
  chip_healthy=0
  while [ "$(date +%s)" -lt "$chip_deadline" ]; do
    if curl -fsS --connect-timeout 3 "http://127.0.0.1:${port}/health" >/dev/null; then
      echo "[vllm-multi-worker] HEALTHY chip=$chip port=$port"
      chip_healthy=1
      break
    elif ! tmux has-session -t "$session" 2>/dev/null; then
      echo "[vllm-multi-worker] ERROR: session exited chip=$chip port=$port"
      tail -160 "$HOME/vllm_server_chip_${chip}.log" 2>/dev/null || true
      exit 1
    fi
    sleep "$POLL_SECONDS"
  done

  if [ "$chip_healthy" != "1" ]; then
    echo "[vllm-multi-worker] ERROR: timed out waiting for chip=$chip port=$port"
    tail -160 "$HOME/vllm_server_chip_${chip}.log" 2>/dev/null || true
    exit 1
  fi
done

echo "[vllm-multi-worker] all chip servers healthy worker=$WORKER_ID chips=[$CHIPS]"
