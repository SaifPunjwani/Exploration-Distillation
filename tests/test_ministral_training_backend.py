import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

from tmx_jax.mesh import build_mesh
from tmx_jax.model import Qwen3Config, Qwen3Model, set_active_mesh
from tmx_jax.train import _make_qwen3_config
from tmx_jax.weights import flax_params_to_hf_tensors, hf_to_flax_params


def test_ministral3_config_uses_text_decoder_contract():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "mistral3",
                    "text_config": {
                        "hidden_size": 8,
                        "intermediate_size": 16,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "head_dim": 4,
                        "vocab_size": 32,
                        "tie_word_embeddings": True,
                        "rms_norm_eps": 1e-5,
                        "max_position_embeddings": 256,
                        "rope_parameters": {
                            "rope_type": "yarn",
                            "factor": 16.0,
                            "rope_theta": 1000000.0,
                            "original_max_position_embeddings": 16,
                        },
                    },
                },
                f,
            )
        cfg = _make_qwen3_config(d)

    assert cfg.architecture == "ministral3"
    assert cfg.hf_weight_prefix == "language_model.model"
    assert cfg.use_qk_norm is False
    assert cfg.rope_type == "yarn"
    assert cfg.rope_factor == 16.0


def test_ministral3_text_weight_mapping_omits_qk_norm():
    with tempfile.TemporaryDirectory() as d:
        tensors = {
            "language_model.model.embed_tokens.weight": np.zeros((32, 8), np.float32),
            "language_model.model.norm.weight": np.ones((8,), np.float32),
            "language_model.model.layers.0.input_layernorm.weight": np.ones((8,), np.float32),
            "language_model.model.layers.0.post_attention_layernorm.weight": np.ones((8,), np.float32),
            "language_model.model.layers.0.self_attn.q_proj.weight": np.zeros((8, 8), np.float32),
            "language_model.model.layers.0.self_attn.k_proj.weight": np.zeros((4, 8), np.float32),
            "language_model.model.layers.0.self_attn.v_proj.weight": np.zeros((4, 8), np.float32),
            "language_model.model.layers.0.self_attn.o_proj.weight": np.zeros((8, 8), np.float32),
            "language_model.model.layers.0.mlp.gate_proj.weight": np.zeros((16, 8), np.float32),
            "language_model.model.layers.0.mlp.up_proj.weight": np.zeros((16, 8), np.float32),
            "language_model.model.layers.0.mlp.down_proj.weight": np.zeros((8, 16), np.float32),
        }
        save_file(tensors, os.path.join(d, "model.safetensors"))
        params = hf_to_flax_params(
            d,
            1,
            True,
            use_qk_norm=False,
            hf_weight_prefix="language_model.model",
        )["params"]

    assert "q_norm" not in params["layers_0"]["self_attn"]
    exported = flax_params_to_hf_tensors(
        params,
        1,
        True,
        use_qk_norm=False,
        hf_weight_prefix="language_model.model",
    )
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in exported
    assert not any("q_norm" in key or "k_norm" in key for key in exported)


def test_ministral3_text_model_forward_shapes_on_reference_backend():
    set_active_mesh(build_mesh())
    cfg = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        tie_word_embeddings=True,
        use_qk_norm=False,
        architecture="ministral3",
        rope_type="yarn",
        rope_factor=16.0,
        rope_original_max_position_embeddings=16,
        max_position_embeddings=256,
    )
    model = Qwen3Model(cfg)
    input_ids = jnp.ones((2, 4), dtype=jnp.int32)
    attention_mask = jnp.ones((2, 4), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), input_ids, attention_mask)["params"]

    logits = model.apply({"params": params}, input_ids, attention_mask)
    hidden = model.apply({"params": params}, input_ids, attention_mask, return_hidden=True)
    pooled = model.apply({"params": params}, input_ids, attention_mask, return_pooled_layers=[0])

    assert logits.shape == (2, 4, 32)
    assert hidden.shape == (2, 4, 8)
    assert pooled.shape == (2, 1, 8)
