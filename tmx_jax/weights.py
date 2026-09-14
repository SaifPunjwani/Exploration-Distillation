"""HuggingFace Qwen3 safetensors → Flax params mapping.

Qwen3 HF state-dict naming convention:
  model.embed_tokens.weight
  model.layers.<i>.self_attn.{q,k,v,o}_proj.weight
  model.layers.<i>.mlp.{gate,up,down}_proj.weight
  model.layers.<i>.{input_layernorm, post_attention_layernorm}.weight
  model.norm.weight
  lm_head.weight            (absent when tie_word_embeddings=true)

Mapping to our Flax tree (see model.py):
  Qwen3Model/embed_tokens/embedding          <- model.embed_tokens.weight
  Qwen3Model/layers_<i>/input_layernorm/weight
                                             <- model.layers.<i>.input_layernorm.weight
  Qwen3Model/layers_<i>/self_attn/{q,k,v,o}_proj/kernel
                                             <- model.layers.<i>.self_attn.{q,k,v,o}_proj.weight (transposed)
  Qwen3Model/layers_<i>/self_attn/{q,k}_norm/weight
                                             <- model.layers.<i>.self_attn.{q,k}_norm.weight
  Qwen3Model/layers_<i>/post_attention_layernorm/weight
                                             <- model.layers.<i>.post_attention_layernorm.weight
  Qwen3Model/layers_<i>/mlp/{gate,up,down}_proj/kernel
                                             <- model.layers.<i>.mlp.{gate,up,down}_proj.weight (transposed)
  Qwen3Model/norm/weight                     <- model.norm.weight
  Qwen3Model/lm_head/kernel                  <- lm_head.weight (transposed)
                                                 (skip if tied to embed)

Transpose: Flax `nn.Dense` kernel is (in, out); PyTorch `nn.Linear.weight` is (out, in).
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors_file


def _download(model_name: str, cache_dir: str | None = None) -> str:
    if os.path.isdir(model_name):
        return os.path.abspath(model_name)
    return snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.safetensors", "*.json", "*.tokenizer*", "*.model"],
        cache_dir=cache_dir,
    )


def _load_shards(snapshot_dir: str) -> Dict[str, np.ndarray]:
    """Load all tensors from every safetensors file in `snapshot_dir`."""
    # safetensors' NumPy backend needs ml_dtypes registered before reading
    # HuggingFace bf16 shards, otherwise Qwen3 base weights fail with
    # "data type 'bfloat16' not understood" in minimal TPU envs.
    import ml_dtypes  # noqa: F401

    tensors: Dict[str, np.ndarray] = {}
    for fname in sorted(os.listdir(snapshot_dir)):
        if not fname.endswith(".safetensors"):
            continue
        path = os.path.join(snapshot_dir, fname)
        with safe_open(path, framework="np") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    return tensors


def _model_prefix_from_state_dict(sd: Dict[str, np.ndarray]) -> str:
    if "model.embed_tokens.weight" in sd:
        return "model"
    if "language_model.model.embed_tokens.weight" in sd:
        return "language_model.model"
    raise KeyError(
        "Could not find a supported text model prefix. Expected either "
        "'model.embed_tokens.weight' or 'language_model.model.embed_tokens.weight'."
    )


def hf_to_flax_params(
    model_name: str,
    num_hidden_layers: int,
    tie_word_embeddings: bool,
    dtype=None,
    *,
    use_qk_norm: bool | None = None,
    hf_weight_prefix: str | None = None,
) -> Dict[str, Any]:
    """Download HF text weights and map to this repo's Flax params pytree.

    Returns a nested dict matching our Flax model (top key: "params").
    """
    snap = _download(model_name)
    sd = _load_shards(snap)
    prefix = (hf_weight_prefix or "").strip(".") or _model_prefix_from_state_dict(sd)
    if use_qk_norm is None:
        use_qk_norm = f"{prefix}.layers.0.self_attn.q_norm.weight" in sd

    def T(x):  # transpose (out,in) -> (in,out)
        return np.ascontiguousarray(x.T)

    params: Dict[str, Any] = {
        "embed_tokens": {"embedding": sd[f"{prefix}.embed_tokens.weight"]},
        "norm": {"weight": sd[f"{prefix}.norm.weight"]},
    }
    for i in range(num_hidden_layers):
        attn = {
            "q_proj": {"kernel": T(sd[f"{prefix}.layers.{i}.self_attn.q_proj.weight"])},
            "k_proj": {"kernel": T(sd[f"{prefix}.layers.{i}.self_attn.k_proj.weight"])},
            "v_proj": {"kernel": T(sd[f"{prefix}.layers.{i}.self_attn.v_proj.weight"])},
            "o_proj": {"kernel": T(sd[f"{prefix}.layers.{i}.self_attn.o_proj.weight"])},
        }
        if use_qk_norm:
            attn["q_norm"] = {"weight": sd[f"{prefix}.layers.{i}.self_attn.q_norm.weight"]}
            attn["k_norm"] = {"weight": sd[f"{prefix}.layers.{i}.self_attn.k_norm.weight"]}
        layer = {
            "input_layernorm": {"weight": sd[f"{prefix}.layers.{i}.input_layernorm.weight"]},
            "post_attention_layernorm": {"weight": sd[f"{prefix}.layers.{i}.post_attention_layernorm.weight"]},
            "self_attn": attn,
            "mlp": {
                "gate_proj": {"kernel": T(sd[f"{prefix}.layers.{i}.mlp.gate_proj.weight"])},
                "up_proj":   {"kernel": T(sd[f"{prefix}.layers.{i}.mlp.up_proj.weight"])},
                "down_proj": {"kernel": T(sd[f"{prefix}.layers.{i}.mlp.down_proj.weight"])},
            },
        }
        params[f"layers_{i}"] = layer

    lm_head_key = "lm_head.weight" if prefix == "model" else "language_model.lm_head.weight"
    if not tie_word_embeddings and lm_head_key in sd:
        params["lm_head"] = {"kernel": T(sd[lm_head_key])}

    if dtype is not None:
        # Keep arrays as numpy (host memory) so downstream can shard them
        # directly via jax.device_put(x, NamedSharding). If we cast to jnp here,
        # weights end up on chip 0 BEFORE sharding → OOM on small-HBM chips.
        np_dtype_map = {
            "float32": np.float32, "float16": np.float16, "bfloat16": np.float32,  # bf16 not in numpy; keep fp32
        }
        try:
            import jax.numpy as jnp
            target = dtype
            if target is jnp.bfloat16:
                # bf16 isn't a native numpy dtype; keep fp32 and cast at device_put time.
                target = np.float32
            elif hasattr(target, "dtype"):
                target = np.dtype(target)
            else:
                target = np.dtype(str(target))
        except Exception:
            target = np.float32

        def _cast(x):
            if isinstance(x, dict):
                return {k: _cast(v) for k, v in x.items()}
            return x.astype(target, copy=False)
        params = _cast(params)

    return {"params": params}


def flax_params_to_hf_tensors(
    params: Dict[str, Any],
    num_hidden_layers: int,
    tie_word_embeddings: bool,
    *,
    use_qk_norm: bool = True,
    hf_weight_prefix: str = "model",
) -> Dict[str, np.ndarray]:
    """Map Flax params pytree back to HF text tensor names.

    Output tensors are numpy arrays ready for safetensors serialization.
    """

    def _np(x):
        return np.asarray(x)

    def _pt_linear_weight(kernel):
        return np.ascontiguousarray(_np(kernel).T)

    prefix = (hf_weight_prefix or "model").strip(".")
    tensors: Dict[str, np.ndarray] = {
        f"{prefix}.embed_tokens.weight": np.ascontiguousarray(_np(params["embed_tokens"]["embedding"])),
        f"{prefix}.norm.weight": np.ascontiguousarray(_np(params["norm"]["weight"])),
    }

    for i in range(num_hidden_layers):
        layer = params[f"layers_{i}"]
        tensors[f"{prefix}.layers.{i}.input_layernorm.weight"] = np.ascontiguousarray(
            _np(layer["input_layernorm"]["weight"])
        )
        tensors[f"{prefix}.layers.{i}.post_attention_layernorm.weight"] = np.ascontiguousarray(
            _np(layer["post_attention_layernorm"]["weight"])
        )
        tensors[f"{prefix}.layers.{i}.self_attn.q_proj.weight"] = _pt_linear_weight(
            layer["self_attn"]["q_proj"]["kernel"]
        )
        tensors[f"{prefix}.layers.{i}.self_attn.k_proj.weight"] = _pt_linear_weight(
            layer["self_attn"]["k_proj"]["kernel"]
        )
        tensors[f"{prefix}.layers.{i}.self_attn.v_proj.weight"] = _pt_linear_weight(
            layer["self_attn"]["v_proj"]["kernel"]
        )
        tensors[f"{prefix}.layers.{i}.self_attn.o_proj.weight"] = _pt_linear_weight(
            layer["self_attn"]["o_proj"]["kernel"]
        )
        if use_qk_norm:
            tensors[f"{prefix}.layers.{i}.self_attn.q_norm.weight"] = np.ascontiguousarray(
                _np(layer["self_attn"]["q_norm"]["weight"])
            )
            tensors[f"{prefix}.layers.{i}.self_attn.k_norm.weight"] = np.ascontiguousarray(
                _np(layer["self_attn"]["k_norm"]["weight"])
            )
        tensors[f"{prefix}.layers.{i}.mlp.gate_proj.weight"] = _pt_linear_weight(
            layer["mlp"]["gate_proj"]["kernel"]
        )
        tensors[f"{prefix}.layers.{i}.mlp.up_proj.weight"] = _pt_linear_weight(
            layer["mlp"]["up_proj"]["kernel"]
        )
        tensors[f"{prefix}.layers.{i}.mlp.down_proj.weight"] = _pt_linear_weight(
            layer["mlp"]["down_proj"]["kernel"]
        )

    if not tie_word_embeddings and "lm_head" in params:
        lm_head_key = "lm_head.weight" if prefix == "model" else "language_model.lm_head.weight"
        tensors[lm_head_key] = _pt_linear_weight(params["lm_head"]["kernel"])

    return tensors


def export_flax_params_to_hf_dir(
    params: Dict[str, Any],
    model_name: str,
    num_hidden_layers: int,
    tie_word_embeddings: bool,
    output_dir: str,
    *,
    cache_dir: str | None = None,
    save_dtype: str | None = "bfloat16",
    use_qk_norm: bool = True,
    hf_weight_prefix: str = "model",
) -> str:
    """Export Flax params to an HF-style local model directory.

    The output contains config/tokenizer files copied from the original snapshot
    plus a single `model.safetensors` file with the current trained weights.
    """

    os.makedirs(output_dir, exist_ok=True)
    snap = _download(model_name, cache_dir=cache_dir)
    tensors = flax_params_to_hf_tensors(
        params,
        num_hidden_layers,
        tie_word_embeddings,
        use_qk_norm=use_qk_norm,
        hf_weight_prefix=hf_weight_prefix,
    )
    if save_dtype is not None:
        if save_dtype in ("bf16", "bfloat16"):
            import ml_dtypes
            target_dtype = ml_dtypes.bfloat16
        else:
            target_dtype = np.dtype(save_dtype)
        tensors = {
            k: np.ascontiguousarray(v.astype(target_dtype, copy=False))
            for k, v in tensors.items()
        }

    keep_files = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "vocab.txt",
    }
    keep_suffixes = (".model",)

    for name in os.listdir(snap):
        src = os.path.join(snap, name)
        dst = os.path.join(output_dir, name)
        if name in keep_files or name.endswith(keep_suffixes):
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    save_safetensors_file(
        tensors,
        os.path.join(output_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )
    return output_dir


def load_hf_config(model_name: str) -> Dict[str, Any]:
    """Read the HF config.json to grab real model dims."""
    import json
    if os.path.isdir(model_name):
        snap = os.path.abspath(model_name)
    else:
        snap = snapshot_download(repo_id=model_name, allow_patterns=["config.json"])
    with open(os.path.join(snap, "config.json")) as f:
        return json.load(f)
