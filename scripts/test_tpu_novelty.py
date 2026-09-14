#!/usr/bin/env python3
"""Test whether TPU hidden-state extraction actually hangs or works fine.

Run on a TPU VM:
  export PJRT_DEVICE=TPU
  export TPU_VISIBLE_CHIPS=0
  python scripts/test_tpu_novelty.py
"""
import os
import time
import torch

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_VISIBLE_CHIPS", "0")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
os.environ.setdefault("TPU_WORKER_HOSTNAMES", "localhost")

print("=== TPU Novelty Hidden-State Extraction Test ===")
print(f"PJRT_DEVICE={os.environ.get('PJRT_DEVICE')}")

# --- Step 1: Check XLA availability ---
print("\n[1] Importing torch_xla...")
t0 = time.monotonic()
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f"    XLA device: {device} ({time.monotonic()-t0:.1f}s)")

# --- Step 2: Load model ---
print("\n[2] Loading Qwen3-1.7B...")
t0 = time.monotonic()
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.eval()
print(f"    Model loaded on CPU ({time.monotonic()-t0:.1f}s)")
num_layers = model.config.num_hidden_layers
print(f"    num_hidden_layers={num_layers}, hidden_size={model.config.hidden_size}")

# --- Step 3: Move to TPU ---
print("\n[3] Moving model to TPU...")
t0 = time.monotonic()
model = model.to(device)
xm.mark_step()
print(f"    Model on TPU ({time.monotonic()-t0:.1f}s)")

# --- Step 4: Test basic forward pass (no hidden states) ---
print("\n[4] Basic forward pass (no hidden states)...")
texts = [
    "What is 2+2? The answer is 4.",
    "Solve: if x^2 = 9, then x = 3 or x = -3.",
] * 8  # 16 samples
enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=128, padding=True)
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)
t0 = time.monotonic()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
xm.mark_step()
print(f"    logits shape: {logits.shape} ({time.monotonic()-t0:.1f}s)")
del outputs, logits

# --- Step 5: Test with output_hidden_states=True (THE CRITICAL TEST) ---
print("\n[5] Forward pass WITH output_hidden_states=True...")
t0 = time.monotonic()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
xm.mark_step()
dt = time.monotonic() - t0
print(f"    Got {len(outputs.hidden_states)} hidden states ({dt:.1f}s)")
for i, hs in enumerate(outputs.hidden_states):
    print(f"      hidden_states[{i}]: {hs.shape} dtype={hs.dtype}")
    if i >= 3:
        print(f"      ... ({len(outputs.hidden_states) - 4} more)")
        break

# --- Step 6: Extract specific layers (multilayer) ---
print("\n[6] Multilayer extraction (layers at quarters)...")
selected = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
print(f"    Selected layers: {selected}")
t0 = time.monotonic()
features = {}
for li in selected:
    hidden = outputs.hidden_states[li + 1]  # +1 because index 0 is embeddings
    # Mean pool over non-pad tokens
    mask = attention_mask.float().unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    features[f"layer_{li}"] = pooled.detach()
    print(f"    layer_{li}: pooled shape {pooled.shape}")
xm.mark_step()
print(f"    Multilayer extraction done ({time.monotonic()-t0:.1f}s)")
del outputs

# --- Step 7: Test chunked extraction (like production code) ---
print("\n[7] Chunked multilayer extraction (batch_size=8, like production)...")
batch_size = 8
num_texts = input_ids.size(0)
all_features = {}
t0 = time.monotonic()
for chunk_start in range(0, num_texts, batch_size):
    chunk_end = min(num_texts, chunk_start + batch_size)
    chunk_ids = input_ids[chunk_start:chunk_end]
    chunk_mask = attention_mask[chunk_start:chunk_end]
    ct0 = time.monotonic()
    with torch.no_grad():
        chunk_out = model(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    xm.mark_step()
    cdt = time.monotonic() - ct0
    print(f"    chunk {chunk_start}:{chunk_end} forward: {cdt:.1f}s")
    for li in selected:
        li_clamped = max(0, min(len(chunk_out.hidden_states) - 2, li))
        key = f"layer_{li_clamped}"
        hidden = chunk_out.hidden_states[li_clamped + 1]
        mask = chunk_mask.float().unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        all_features.setdefault(key, []).append(pooled.detach())
    del chunk_out
    xm.mark_step()
total_dt = time.monotonic() - t0
print(f"    Total chunked extraction: {total_dt:.1f}s")

# Concatenate
for k in all_features:
    all_features[k] = torch.cat(all_features[k], dim=0)
    print(f"    {k}: {all_features[k].shape}")

# --- Step 8: Test RND-style forward on extracted features ---
print("\n[8] RND-style computation on TPU features...")
import torch.nn as nn
hidden_dim = 512
input_dim = model.config.hidden_size
target = nn.Sequential(
    nn.Linear(input_dim, hidden_dim), nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
).to(device).to(torch.bfloat16)
predictor = nn.Sequential(
    nn.Linear(input_dim, hidden_dim), nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
).to(device).to(torch.bfloat16)
for p in target.parameters():
    p.requires_grad = False
t0 = time.monotonic()
for key, feat in all_features.items():
    feat_dev = feat.to(device).to(torch.bfloat16)
    with torch.no_grad():
        t_out = target(feat_dev)
        p_out = predictor(feat_dev)
        novelty = ((t_out - p_out) ** 2).mean(dim=-1)
    xm.mark_step()
    print(f"    {key}: novelty scores shape={novelty.shape}, mean={novelty.float().mean().item():.6f}")
print(f"    RND done ({time.monotonic()-t0:.1f}s)")

# --- Step 9: Larger batch test (64 samples) ---
print("\n[9] Larger batch test (64 samples, batch_size=16)...")
texts_64 = texts * 4  # 64 samples
enc64 = tokenizer(texts_64, return_tensors="pt", truncation=True, max_length=512, padding=True)
ids64 = enc64["input_ids"].to(device)
mask64 = enc64["attention_mask"].to(device)
batch_size = 16
t0 = time.monotonic()
for chunk_start in range(0, 64, batch_size):
    chunk_end = min(64, chunk_start + batch_size)
    ct0 = time.monotonic()
    with torch.no_grad():
        out = model(
            input_ids=ids64[chunk_start:chunk_end],
            attention_mask=mask64[chunk_start:chunk_end],
            output_hidden_states=True,
            return_dict=True,
        )
    xm.mark_step()
    cdt = time.monotonic() - ct0
    print(f"    chunk {chunk_start}:{chunk_end}: {cdt:.1f}s ({len(out.hidden_states)} states)")
    del out
    xm.mark_step()
total_dt = time.monotonic() - t0
print(f"    Total 64-sample extraction: {total_dt:.1f}s")

print("\n=== ALL TESTS PASSED ===")
print("TPU hidden-state extraction works. The 'hang' was likely a bug, not a fundamental limitation.")
