#!/usr/bin/env bash
# Bootstrap + run vLLM-TPU on a single TPU worker.
# The operational fast path is a local env; legacy GCS env cache is opt-in only.
set -euo pipefail

VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
VLLM_ENV_CACHE_GCS_URI="${VLLM_ENV_CACHE_GCS_URI:-}"
TMX_ALLOW_GCS_ARTIFACTS="${TMX_ALLOW_GCS_ARTIFACTS:-0}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/vllm_miniforge}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_TPU_BUCKET_PADDING_GAP="${VLLM_TPU_BUCKET_PADDING_GAP:-}"
VLLM_TPU_MOST_MODEL_LEN="${VLLM_TPU_MOST_MODEL_LEN:-}"
VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-$HOME/.cache/vllm/xla_cache}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}"
VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-tp${TP}_pda${VLLM_V1_USE_PREFILL_DECODE_ATTENTION}_mlen${MAX_MODEL_LEN}_mbt${MAX_NUM_BATCHED_TOKENS}}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-auto}"
TMX_VLLM_SUPERVISE="${TMX_VLLM_SUPERVISE:-1}"
TMX_VLLM_RESTART_SLEEP_SECONDS="${TMX_VLLM_RESTART_SLEEP_SECONDS:-5}"
if [ -z "${TPU_VISIBLE_CHIPS:-}" ]; then
  if [ "$TP" = "1" ]; then
    TPU_VISIBLE_CHIPS="0"
  else
    TPU_VISIBLE_CHIPS="0,1,2,3"
  fi
fi
TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
if [ -z "${TPU_CHIPS_PER_PROCESS_BOUNDS:-}" ]; then
  if [ "$TP" = "1" ]; then
    TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1"
  else
    TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1"
  fi
fi
if [ -z "${TPU_CHIPS_PER_HOST_BOUNDS:-}" ]; then
  TPU_CHIPS_PER_HOST_BOUNDS="$TPU_CHIPS_PER_PROCESS_BOUNDS"
fi
TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"

stop_existing_vllm() {
  pkill -9 -f "[v]llm serve" 2>/dev/null || true
  pkill -9 -f "[v]llm.entrypoints.openai.api_server" 2>/dev/null || true
  pkill -9 -f "[V]LLM::EngineCore" 2>/dev/null || true
  tmux kill-session -t vllm_server 2>/dev/null || true
  if tmux has-session -t vllm_server 2>/dev/null; then
    tmux kill-server 2>/dev/null || true
  fi
  rm -rf "/tmp/tmux-$(id -u)" 2>/dev/null || true
  fuser -k /dev/vfio/0 /dev/vfio/1 /dev/vfio/2 /dev/vfio/3 2>/dev/null || true
  rm -f /tmp/libtpu_lockfile 2>/dev/null || true
  sleep 2
}

restore_cached_env() {
  [ -n "$VLLM_ENV_CACHE_GCS_URI" ] || return 1
  gcloud storage ls "$VLLM_ENV_CACHE_GCS_URI" >/dev/null 2>&1 || return 1
  local archive_path
  archive_path="$(mktemp /tmp/vllm_tpu_env.XXXXXX.tar.gz)"
  rm -rf "$VLLM_VENV_DIR"
  mkdir -p "$(dirname "$VLLM_VENV_DIR")"
  gcloud storage cp "$VLLM_ENV_CACHE_GCS_URI" "$archive_path" >/dev/null
  tar -xzf "$archive_path" -C "$(dirname "$VLLM_VENV_DIR")"
  rm -f "$archive_path"
  [ -x "$VLLM_VENV_DIR/bin/python" ]
}

ensure_pip_in_vllm_env() {
  "$VLLM_VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1 && return 0
  echo "[worker] repairing missing pip in $VLLM_VENV_DIR"
  if "$VLLM_VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
    "$VLLM_VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    return 0
  fi
  if [ -x "$MINIFORGE_DIR/bin/conda" ]; then
    "$MINIFORGE_DIR/bin/conda" install -y -p "$VLLM_VENV_DIR" pip setuptools wheel
    "$VLLM_VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    return 0
  fi
  echo "[worker] ERROR unable to repair pip in $VLLM_VENV_DIR" >&2
  return 1
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
        print(f"[worker] WARN missing TPU RPA kernel {path}")
        continue
    text = path.read_text()
    updated = old_implicit.sub(replacement, text)
    updated = old_product.sub(replacement, updated)
    updated = old_hardcoded.sub(replacement, updated)
    updated = old_dynamic_mult.sub(replacement, updated)
    if updated != text:
        path.write_text(updated)
        print(f"[worker] patched TPU RPA kernel {path}")
    elif replacement in text:
        print(f"[worker] TPU RPA kernel already patched {path}")
    else:
        print(f"[worker] WARN TPU RPA kernel patch pattern not found {path}")

if wrapper_path.exists():
    text = wrapper_path.read_text()
    updated = text.replace('"post_spmd_conservative"', '"POST_SPMD"')
    if updated != text:
        wrapper_path.write_text(updated)
        print(f"[worker] patched TPU collective matmul compiler options {wrapper_path}")
    elif '"POST_SPMD"' in text:
        print(f"[worker] TPU collective matmul compiler options already patched {wrapper_path}")
    else:
        print(f"[worker] WARN TPU collective matmul patch pattern not found {wrapper_path}")
else:
    print(f"[worker] WARN missing TPU vLLM model wrapper {wrapper_path}")
PY
}

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
    print(f"[worker] WARN local HF cache miss for {model}: {exc}", file=sys.stderr)
    raise SystemExit(0)
print(path)
PY
}

ensure_vllm_tpu_runtime_stack() {
  [ "${TMX_VLLM_ENSURE_JAX_STACK:-1}" = "0" ] && return 0
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
    echo "[worker] repairing TPU runtime stack current=[$current] expected=[$expected]"
    "$VLLM_VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall \
      --extra-index-url https://storage.googleapis.com/libtpu-wheels/index.html \
      --extra-index-url https://storage.googleapis.com/jax-releases/libtpu_releases.html \
      "jax==0.8.1" "jaxlib==0.8.1" "libtpu==0.0.31" "numpy==2.3.5"
  else
    echo "[worker] TPU runtime stack ok [$current]"
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
        print(f"[worker] WARN missing tpu_inference compat target {path}")
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
    print("[worker] patched transformers modeling_flax_utils compatibility: " + ", ".join(changed))
else:
    print("[worker] transformers modeling_flax_utils compatibility already ok")
PY
}

stop_existing_vllm

if [ ! -x "$VLLM_VENV_DIR/bin/vllm" ]; then
  rm -rf "$VLLM_VENV_DIR" 2>/dev/null || true
  if ! restore_cached_env; then
    if [ ! -d "$MINIFORGE_DIR" ]; then
      wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge_vllm.sh
      bash /tmp/miniforge_vllm.sh -b -p "$MINIFORGE_DIR"
    fi
    # shellcheck disable=SC1091
    source "$MINIFORGE_DIR/bin/activate"
    conda create -y -p "$VLLM_VENV_DIR" python=3.11 pip setuptools wheel
    conda activate "$VLLM_VENV_DIR"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install \
      --extra-index-url https://storage.googleapis.com/libtpu-wheels/index.html \
      --extra-index-url https://storage.googleapis.com/jax-releases/libtpu_releases.html \
      'vllm-tpu==0.13.3' datasets wandb transformers sentencepiece tiktoken
  fi
fi

ensure_pip_in_vllm_env
ensure_vllm_tpu_runtime_stack
patch_vllm_tpu_transformers_compat || echo "[worker] WARN transformers compatibility patch failed; continuing"
patch_vllm_tpu_rpa_kernel || echo "[worker] WARN TPU RPA patch failed; continuing"

model_arg="$MODEL"
cache_warn="/tmp/vllm_cached_model_tp${TP}.warn"
cached_model="$(resolve_cached_hf_model "$MODEL" 2>"$cache_warn" || true)"
if [ -n "$cached_model" ] && [ -d "$cached_model" ]; then
  model_arg="$cached_model"
  echo "[worker] using local HF snapshot for $MODEL -> $model_arg"
elif [ -s "$cache_warn" ]; then
  cat "$cache_warn"
fi

if [ -z "${VLLM_XLA_CACHE_PATH:-}" ] || [ "$VLLM_XLA_CACHE_PATH" = "$HOME/.cache/vllm/xla_cache" ]; then
  safe_cache_tag="$(printf '%s' "$VLLM_XLA_CACHE_TAG" | tr -c 'A-Za-z0-9_.-' '_')"
  VLLM_XLA_CACHE_PATH="$HOME/.cache/vllm/xla_cache_${safe_cache_tag}_tp${TP}"
fi

cat > /tmp/run_vllm.sh <<EOF
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
export TMX_VLLM_SUPERVISE="$TMX_VLLM_SUPERVISE"
export TMX_VLLM_RESTART_SLEEP_SECONDS="$TMX_VLLM_RESTART_SLEEP_SECONDS"
export TPU_WORKER_HOSTNAMES="\${TPU_WORKER_HOSTNAMES:-localhost}"
export MODEL_IMPL_TYPE="$MODEL_IMPL_TYPE"
if [ -d "$model_arg" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
export VLLM_XLA_CACHE_PATH="\${VLLM_XLA_CACHE_PATH:-$VLLM_XLA_CACHE_PATH}"
export XLA_PERSISTENT_CACHE_PATH="\${XLA_PERSISTENT_CACHE_PATH:-\$VLLM_XLA_CACHE_PATH}"
if [ -n "\${TMX_VLLM_SAFE_XLA_FLAGS:-}" ]; then
  export XLA_FLAGS="\$TMX_VLLM_SAFE_XLA_FLAGS"
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
if [ -n "$MODEL_IMPL_TYPE" ] && [ "$MODEL_IMPL_TYPE" != "auto" ]; then
  extra_args+=(--model-impl "$MODEL_IMPL_TYPE")
fi
model_lc="\$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
if [[ "\$model_lc" == *ministral-3* || "\$model_lc" == *ministral3* ]]; then
  # Ministral 3 is packaged as a Mistral3/Pixtral-style multimodal checkpoint.
  # The math pipeline is text-only; these flags keep vLLM on the verified TPU
  # language-model path and avoid multimodal chunk accounting.
  extra_args+=(--language-model-only --disable-chunked-mm-input)
fi
while true; do
  "$VLLM_VENV_DIR/bin/vllm" serve "$model_arg" \\
    --host 0.0.0.0 --port "$PORT" \\
    --tensor-parallel-size "$TP" \\
    --dtype bfloat16 \\
    --max-model-len "$MAX_MODEL_LEN" \\
    --max-num-seqs "$MAX_NUM_SEQS" \\
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \\
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \\
    --generation-config vllm \\
    --served-model-name "$MODEL" \\
    "\${extra_args[@]}"
  rc="\$?"
  echo "[worker] vLLM exited rc=\$rc supervise=\${TMX_VLLM_SUPERVISE:-1} at \$(date -Is)"
  if [ "\${TMX_VLLM_SUPERVISE:-1}" = "0" ]; then
    exit "\$rc"
  fi
  sleep "\${TMX_VLLM_RESTART_SLEEP_SECONDS:-5}"
done
EOF
chmod +x /tmp/run_vllm.sh
if ! tmux new-session -d -s vllm_server "bash /tmp/run_vllm.sh > \"$HOME/vllm_server.log\" 2>&1"; then
  tmux kill-server 2>/dev/null || true
  rm -rf "/tmp/tmux-$(id -u)" 2>/dev/null || true
  tmux new-session -d -s vllm_server "bash /tmp/run_vllm.sh > \"$HOME/vllm_server.log\" 2>&1"
fi
echo "[worker] bootstrap + tmux vllm_server launched"
