#!/usr/bin/env bash
set -euo pipefail

# TPU smoke suite: step-by-step checks to isolate slowness.
# Usage:
#   bash scripts/tpu_smoke_suite.sh
# Optional env vars:
#   MODEL_NAME=Qwen/Qwen2.5-Math-1.5B-Instruct
#   TPU_VISIBLE_DEVICES=0
#   RUN_GRPO=0   # skip optional GRPO step (default: 0)

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
TPU_VISIBLE_DEVICES="${TPU_VISIBLE_DEVICES:-}"
RUN_GRPO="${RUN_GRPO:-0}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export PJRT_DEVICE=TPU
export TOKENIZERS_PARALLELISM=false
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v4-8}"
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
if [ -n "$TPU_VISIBLE_DEVICES" ]; then
  export TPU_VISIBLE_DEVICES
fi

echo "== Step 1: /dev/accel* access =="
"$PYTHON_BIN" - <<'PY'
import os
ok = False
for i in range(4):
    p = f"/dev/accel{i}"
    try:
        fd = os.open(p, os.O_RDWR)
        os.close(fd)
        print(f"{p} OK")
        ok = True
    except Exception as e:
        print(f"{p} FAIL {e}")
if not ok:
    raise SystemExit("TPU device not openable.")
PY

echo "== Step 2: XLA matmul microbench =="
"$PYTHON_BIN" - <<'PY'
import time
import torch
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f"Using device: {device}")
sizes = [512, 1024, 2048]
for size in sizes:
    x_cpu = torch.randn(size, size)
    t0 = time.time()
    for _ in range(5):
        torch.matmul(x_cpu, x_cpu)
    cpu_time = time.time() - t0

    x_tpu = torch.randn(size, size).to(device)
    xm.mark_step()
    t0 = time.time()
    for _ in range(5):
        torch.matmul(x_tpu, x_tpu)
    xm.mark_step()
    tpu_time = time.time() - t0
    print(f"{size}x{size}: CPU={cpu_time:.3f}s TPU={tpu_time:.3f}s")
PY

echo "== Step 3: HF model load + forward (TPU) =="
"$PYTHON_BIN" - <<PY
import time
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tmx.utils import (
    patch_torch_isin_for_xla,
    patch_transformers_isin_for_xla,
    patch_transformers_attention_mask_for_xla,
    patch_transformers_logits_processor_for_xla,
    patch_transformers_stopping_criteria_for_xla,
)

patch_torch_isin_for_xla()
patch_transformers_isin_for_xla()
patch_transformers_attention_mask_for_xla()
patch_transformers_logits_processor_for_xla()
patch_transformers_stopping_criteria_for_xla()

device = xm.xla_device()
tok = AutoTokenizer.from_pretrained("${MODEL_NAME}")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("${MODEL_NAME}").to(device)
model.eval()

inputs = tok("Hello world!", return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attn = inputs["attention_mask"].to(device)

with torch.no_grad():
    xm.mark_step()
    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attn)
    xm.mark_step()
    print(f"Forward time: {time.time() - t0:.3f}s")
PY

echo "== Step 4: HF generate (TPU, cache on) =="
"$PYTHON_BIN" - <<PY
import time
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tmx.utils import (
    patch_torch_isin_for_xla,
    patch_transformers_isin_for_xla,
    patch_transformers_attention_mask_for_xla,
    patch_transformers_logits_processor_for_xla,
    patch_transformers_stopping_criteria_for_xla,
)

patch_torch_isin_for_xla()
patch_transformers_isin_for_xla()
patch_transformers_attention_mask_for_xla()
patch_transformers_logits_processor_for_xla()
patch_transformers_stopping_criteria_for_xla()

device = xm.xla_device()
tok = AutoTokenizer.from_pretrained("${MODEL_NAME}")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("${MODEL_NAME}").to(device)
model.eval()
model.config.use_cache = True
if getattr(model, "generation_config", None) is not None:
    model.generation_config.use_cache = True

inputs = tok("Hello world!", return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attn = inputs["attention_mask"].to(device)

with torch.no_grad():
    xm.mark_step()
    t0 = time.time()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=8,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    xm.mark_step()
    t1 = time.time()
    out2 = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=8,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    xm.mark_step()
    t2 = time.time()
    print(f"Generate time (1st): {t1 - t0:.3f}s")
    print(f"Generate time (2nd): {t2 - t1:.3f}s")
    print(tok.decode(out[0], skip_special_tokens=True))
PY

echo "== Step 5: xla_safe_generate (TPU) =="
"$PYTHON_BIN" - <<PY
import time
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer, AutoModelForCausalLM
from tmx.utils import (
    patch_torch_isin_for_xla,
    patch_transformers_isin_for_xla,
    patch_transformers_attention_mask_for_xla,
    patch_transformers_logits_processor_for_xla,
    patch_transformers_stopping_criteria_for_xla,
    xla_safe_generate,
)

patch_torch_isin_for_xla()
patch_transformers_isin_for_xla()
patch_transformers_attention_mask_for_xla()
patch_transformers_logits_processor_for_xla()
patch_transformers_stopping_criteria_for_xla()

device = xm.xla_device()
tok = AutoTokenizer.from_pretrained("${MODEL_NAME}")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("${MODEL_NAME}").to(device)
model.eval()

inputs = tok("Hello world!", return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attn = inputs["attention_mask"].to(device)

with torch.no_grad():
    xm.mark_step()
    t0 = time.time()
    out = xla_safe_generate(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=8,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    xm.mark_step()
    t1 = time.time()
    out2 = xla_safe_generate(
        model,
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=8,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    xm.mark_step()
    t2 = time.time()
    print(f"xla_safe_generate time (1st): {t1 - t0:.3f}s")
    print(f"xla_safe_generate time (2nd): {t2 - t1:.3f}s")
    print(tok.decode(out[0], skip_special_tokens=True))
PY

echo "== Step 6: HF generate (CPU baseline) =="
env -u PJRT_DEVICE USE_TORCH_XLA=0 CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" - <<PY
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("${MODEL_NAME}")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained("${MODEL_NAME}")
model.eval()

inputs = tok("Hello world!", return_tensors="pt")
input_ids = inputs["input_ids"]
attn = inputs["attention_mask"]

t0 = time.time()
_ = model.generate(
    input_ids=input_ids,
    attention_mask=attn,
    max_new_tokens=8,
    do_sample=True,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
)
t1 = time.time()
print(f"CPU generate time: {t1 - t0:.3f}s")
PY

if [ "$RUN_GRPO" = "1" ]; then
  echo "== Step 7: Minimal GRPO step (optional) =="
  "$PYTHON_BIN" main.py --device tpu \
    --model-name "$MODEL_NAME" \
    --max-train-examples 2 --max-eval-examples 2 \
    --max-prompt-len 64 --max-completion-len 32 \
    --grpo-batch-size 1 --grpo-generation-batch-size 2 --grpo-num-generations 2 \
    --grpo-grad-accum 1 --grpo-max-steps 1 \
    --lambda-novelty 0.1 --grpo-loss-type dr_grpo --novelty-metric mse \
    --no-wandb
fi
