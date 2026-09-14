"""Subprocess helper: create a tiny HF checkpoint and independent forward outputs.

Requires torch and a Transformers version supporting the requested architecture.
No network or model downloads. Used by test_expdis_jax_model_reference.py.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

root, architecture, dtype = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
root.mkdir(exist_ok=True)
torch.manual_seed(4)
config_type, model_type = Qwen3Config, Qwen3ForCausalLM
extra = {}
if architecture == "ministral3":
    from transformers import Ministral3Config, Ministral3ForCausalLM
    config_type, model_type = Ministral3Config, Ministral3ForCausalLM
    extra["rope_parameters"] = dict(rope_type="yarn", rope_theta=10000., factor=4.,
        original_max_position_embeddings=4, beta_fast=32., beta_slow=1.,
        llama_4_scaling_beta=.1)
cfg = config_type(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
    tie_word_embeddings=False, **extra)
cfg._attn_implementation = "eager"
model = model_type(cfg).eval()
model.save_pretrained(root)
if dtype == "bf16":
    model = model.to(torch.bfloat16)
ids = torch.tensor([[0, 0, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
mask = (ids != 0).long()
positions = (mask.cumsum(-1) - 1).clamp(min=0)
with torch.no_grad():
    out = model(ids, attention_mask=mask, position_ids=positions, output_hidden_states=True)
np.savez(root / "reference.npz", ids=ids.numpy(), mask=mask.numpy(),
    logits=out.logits.float().numpy(), layer1=out.hidden_states[2].float().numpy(),
    layer2=out.hidden_states[3].float().numpy())
