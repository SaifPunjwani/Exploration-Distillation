"""Numerical comparison with an independent Hugging Face forward pass.

Set EXPDIS_HF_REFERENCE_PYTHON to a Python with torch + transformers (5.x for
Ministral3). The subprocess isolates those optional dependencies from JAX.
"""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from expdis_jax.model import Qwen3Model
from expdis_jax.train import _make_qwen3_config
from expdis_jax.weights import hf_to_flax_params


@pytest.mark.parametrize("architecture", ["qwen3", "ministral3"])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_hugging_face_forward_and_intermediate_features(tmp_path, architecture, dtype):
    python = os.environ.get("EXPDIS_HF_REFERENCE_PYTHON", sys.executable)
    probe = subprocess.run([python, "-c", "from transformers import " +
        ("Ministral3ForCausalLM" if architecture == "ministral3" else "Qwen3ForCausalLM")],
        capture_output=True, text=True)
    if probe.returncode:
        pytest.skip("optional HF reference runtime does not support " + architecture)
    subprocess.run([python, str(Path(__file__).with_name("hf_forward_reference.py")),
                    str(tmp_path), architecture, dtype], check=True, capture_output=True, text=True)
    cfg = replace(_make_qwen3_config(str(tmp_path)),
                  dtype=jnp.float32 if dtype == "fp32" else jnp.bfloat16)
    params = hf_to_flax_params(str(tmp_path), cfg.num_hidden_layers, cfg.tie_word_embeddings,
        dtype=jnp.float32, use_qk_norm=cfg.use_qk_norm, hf_weight_prefix=cfg.hf_weight_prefix)["params"]
    ref = np.load(tmp_path / "reference.npz")
    model = Qwen3Model(cfg)
    output = np.asarray(model.apply({"params": params}, jnp.array(ref["ids"]),
                        jnp.array(ref["mask"])), dtype=np.float32)
    tolerance = 1e-6 if dtype == "fp32" else .004
    np.testing.assert_allclose(output[ref["mask"] > 0], ref["logits"][ref["mask"] > 0],
                               atol=tolerance, rtol=tolerance)
    pooled = model.apply({"params": params}, jnp.array(ref["ids"]), jnp.array(ref["mask"]),
                         return_pooled_layers=(2, 1))
    expected = np.stack([(ref[key] * ref["mask"][..., None]).sum(1) / ref["mask"].sum(1)[:, None]
                         for key in ("layer2", "layer1")], axis=1)
    np.testing.assert_allclose(pooled, expected, atol=tolerance, rtol=tolerance)
