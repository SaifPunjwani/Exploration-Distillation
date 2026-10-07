"""Standalone text exports from the publisher's multimodal Ministral wrapper."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from safetensors import safe_open
from transformers import AutoConfig

from expdis_jax.weights import export_flax_params_to_hf_dir, hf_to_flax_params


def _tiny_params(*, tied, qk_norm=False):
    rng = np.random.default_rng(17)

    def linear(shape):
        return {"kernel": rng.normal(0, 0.02, shape).astype(np.float32)}

    params = {
        "embed_tokens": {"embedding": rng.normal(0, 0.02, (16, 8)).astype(np.float32)},
        "norm": {"weight": np.ones(8, np.float32)},
    }
    attn = {name: linear(shape) for name, shape in {
        "q_proj": (8, 8), "k_proj": (8, 4), "v_proj": (8, 4), "o_proj": (8, 8),
    }.items()}
    if qk_norm:
        attn.update({name: {"weight": np.ones(4, np.float32)} for name in ("q_norm", "k_norm")})
    params["layers_0"] = {
        "input_layernorm": {"weight": np.ones(8, np.float32)},
        "post_attention_layernorm": {"weight": np.ones(8, np.float32)},
        "self_attn": attn,
        "mlp": {name: linear(shape) for name, shape in {
            "gate_proj": (8, 16), "up_proj": (8, 16), "down_proj": (16, 8),
        }.items()},
    }
    if not tied:
        params["lm_head"] = linear((8, 16))
    return params


def _source_snapshot(tmp_path, *, tied, architecture="ministral3"):
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    text_config = {
        "model_type": architecture,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 16,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": tied,
    }
    if architecture == "ministral3":
        text_config["rope_parameters"] = {
            "rope_type": "yarn", "rope_theta": 10000., "factor": 4.,
            "original_max_position_embeddings": 16, "beta_fast": 32., "beta_slow": 1.,
            "mscale": 1., "mscale_all_dim": 1., "llama_4_scaling_beta": 0.1,
        }
        config = {
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "mistral3",
            "dtype": "float32",
            "bos_token_id": 1,
            "eos_token_id": [2, 3],
            "pad_token_id": 0,
            "text_config": text_config,
            "vision_config": {"model_type": "pixtral", "hidden_size": 8},
            "image_token_index": 10,
        }
    else:
        config = {**text_config, "architectures": ["Qwen3ForCausalLM"], "rope_theta": 10000.}
    (snapshot / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Distinct bytes ensure the exporter keeps the native instruct interface.
    assets = {
        "tokenizer.json": '{"version":"1.0","model":{"type":"WordLevel"}}\n',
        "tokenizer_config.json": '{"tokenizer_class":"TokenizersBackend"}\n',
        "special_tokens_map.json": '{"eos_token":"</s>"}\n',
        "generation_config.json": '{"eos_token_id":[2,3],"pad_token_id":0}\n',
        "chat_template.jinja": "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}\n",
        "tokenizer.model": "native-tokenizer-data\n",
    }
    for name, content in assets.items():
        (snapshot / name).write_text(content)
    return snapshot, config, assets


@pytest.mark.parametrize("tied", [True, False])
def test_ministral_export_is_standalone_text_and_roundtrips(tmp_path, tied):
    snapshot, original, assets = _source_snapshot(tmp_path, tied=tied)
    params = _tiny_params(tied=tied)
    exported = Path(export_flax_params_to_hf_dir(
        params, str(snapshot), 1, tied, str(tmp_path / "export"),
        save_dtype="float32", use_qk_norm=False, hf_weight_prefix="language_model.model",
    ))

    config = AutoConfig.from_pretrained(exported, local_files_only=True)
    assert config.model_type == "ministral3"
    assert config.architectures == ["Ministral3ForCausalLM"]
    assert not hasattr(config, "vision_config")
    assert not hasattr(config, "text_config")
    assert config.tie_word_embeddings is tied
    for key, value in original["text_config"]["rope_parameters"].items():
        assert config.rope_parameters[key] == value
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        assert getattr(config, key) == original[key]
    assert json.loads((exported / "config.json").read_text())["dtype"] == original["dtype"]
    for name in assets:
        assert (exported / name).read_bytes() == (snapshot / name).read_bytes()

    with safe_open(exported / "model.safetensors", framework="np") as handle:
        keys = set(handle.keys())
        assert "model.embed_tokens.weight" in keys
        assert not any("language_model" in key or "vision" in key for key in keys)
        assert ("lm_head.weight" in keys) is (not tied)
    restored = hf_to_flax_params(str(exported), 1, tied, use_qk_norm=False)["params"]
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        np.testing.assert_array_equal(restored["layers_0"]["self_attn"][name]["kernel"],
                                      params["layers_0"]["self_attn"][name]["kernel"])
    np.testing.assert_array_equal(restored["embed_tokens"]["embedding"], params["embed_tokens"]["embedding"])
    if not tied:
        np.testing.assert_array_equal(restored["lm_head"]["kernel"], params["lm_head"]["kernel"])


def test_qwen_export_preserves_original_metadata_bytes(tmp_path):
    snapshot, _, assets = _source_snapshot(tmp_path, tied=False, architecture="qwen3")
    exported = Path(export_flax_params_to_hf_dir(
        _tiny_params(tied=False, qk_norm=True), str(snapshot), 1, False,
        str(tmp_path / "export"), save_dtype="float32",
    ))
    for name in ("config.json", *assets):
        assert (exported / name).read_bytes() == (snapshot / name).read_bytes()
    with safe_open(exported / "model.safetensors", framework="np") as handle:
        assert "model.layers.0.self_attn.q_norm.weight" in handle.keys()
        assert "lm_head.weight" in handle.keys()


@pytest.mark.parametrize("tied", [True, False])
def test_ministral_export_loads_with_standard_hf_causal_lm(tmp_path, tied):
    python = os.environ.get("EXPDIS_HF_REFERENCE_PYTHON", sys.executable)
    probe = subprocess.run(
        [python, "-c", "import torch; from transformers import Ministral3ForCausalLM"],
        capture_output=True, text=True,
    )
    if probe.returncode:
        pytest.skip("optional torch + Transformers Ministral3 reference runtime unavailable")
    snapshot, _, _ = _source_snapshot(tmp_path, tied=tied)
    exported = export_flax_params_to_hf_dir(
        _tiny_params(tied=tied), str(snapshot), 1, tied, str(tmp_path / "export"),
        save_dtype="float32", use_qk_norm=False, hf_weight_prefix="language_model.model",
    )
    check = """
import sys
import torch
from transformers import AutoModelForCausalLM
model, info = AutoModelForCausalLM.from_pretrained(
    sys.argv[1], local_files_only=True, output_loading_info=True)
assert type(model).__name__ == 'Ministral3ForCausalLM'
assert not info['missing_keys'], info
assert not info['unexpected_keys'], info
assert not info['mismatched_keys'], info
assert not hasattr(model, 'vision_tower')
tied = sys.argv[2] == '1'
assert (model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()) == tied
with torch.no_grad():
    logits = model(torch.tensor([[1, 4, 5]])).logits
assert logits.shape == (1, 3, 16)
assert torch.isfinite(logits).all()
"""
    result = subprocess.run([python, "-c", check, exported, str(int(tied))],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
