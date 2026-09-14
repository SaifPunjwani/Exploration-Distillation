"""Qwen3-1.7B in Flax.

Qwen3 arch (per HF config for Qwen/Qwen3-1.7B):
  hidden_size = 2048
  intermediate_size = 6144 (for 1.7B; Qwen3-0.6B uses 3072)
  num_hidden_layers = 28
  num_attention_heads = 16
  num_key_value_heads = 8       # Qwen3-1.7B uses GQA with 8 kv-heads
  head_dim = 128
  rope_theta = 1_000_000         # NTK-aware, larger than LLaMA/Qwen2
  rms_norm_eps = 1e-6
  vocab_size = 151_936
  max_position_embeddings = 40960
  tie_word_embeddings = True    # Qwen3 ties lm_head to embed_tokens for 1.7B

Components:
  - RMSNorm (no bias)
  - Per-head q/k RMSNorm, then Rotary Position Embeddings on q, k
  - Grouped-Query Attention (num_q_heads=16, num_kv_heads=8 → ratio=2)
  - SwiGLU MLP: act_fn(gate(x)) * up(x) -> down(...)
  - No biases in linear layers

Weight loading:
  HF checkpoint is PyTorch by default. We convert to JAX arrays via
  `safetensors` + shape reshuffle. The Flax `params` pytree matches the
  `PARAM_SPECS` in mesh.py for sharded placement.

Status:
  M1 scaffold — logits parity vs PyTorch TBD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as _pallas_flash_attention, SegmentIds as _SegmentIds
from jax.experimental.shard_map import shard_map as _shard_map
from jax.sharding import PartitionSpec as _P
from .mesh import FSDP_AXIS as _FSDP_AXIS
from jax import Array
import flax.linen as nn


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151_936
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True
    use_qk_norm: bool = True
    hf_weight_prefix: str = "model"
    architecture: str = "qwen3"
    rope_type: str = "default"
    rope_factor: float = 1.0
    rope_attention_factor: float = 1.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_original_max_position_embeddings: int = 0
    rope_truncate: bool = True
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32   # keep params in fp32 for optimizer stability


def _yarn_inv_freq(
    head_dim: int,
    theta: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    original_max_position_embeddings: int,
    truncate: bool,
) -> Array:
    """YaRN inverse-frequency schedule matching Transformers' rope_utils."""

    pos_freqs = theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (float(factor) * pos_freqs)
    original_len = max(1, int(original_max_position_embeddings))

    def correction_dim(num_rotations: float) -> float:
        return (head_dim * math.log(original_len / (num_rotations * 2.0 * math.pi))) / (
            2.0 * math.log(theta)
        )

    low = correction_dim(float(beta_fast))
    high = correction_dim(float(beta_slow))
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0.0)
    high = min(high, float(head_dim - 1))
    if low == high:
        high += 0.001

    ramp = (jnp.arange(head_dim // 2, dtype=jnp.float32) - float(low)) / (float(high) - float(low))
    ramp = jnp.clip(ramp, 0.0, 1.0)
    extrapolation_factor = 1.0 - ramp
    return inv_freq_interpolation * (1.0 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor


def rotary_cos_sin(seq_len: int, cfg: Qwen3Config):
    head_dim = int(cfg.head_dim)
    theta = float(cfg.rope_theta)
    rope_type = (cfg.rope_type or "default").lower()
    if rope_type == "yarn":
        original_len = int(cfg.rope_original_max_position_embeddings or cfg.max_position_embeddings)
        inv_freq = _yarn_inv_freq(
            head_dim,
            theta,
            float(cfg.rope_factor or 1.0),
            float(cfg.rope_beta_fast or 32.0),
            float(cfg.rope_beta_slow or 1.0),
            original_len,
            bool(cfg.rope_truncate),
        )
        attention_factor = float(cfg.rope_attention_factor or 1.0)
    else:
        inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        attention_factor = 1.0
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)                    # (T, head_dim/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)    # (T, head_dim)
    return (
        (jnp.cos(emb) * attention_factor).astype(cfg.dtype),
        (jnp.sin(emb) * attention_factor).astype(cfg.dtype),
    )


def rotate_half(x: Array) -> Array:
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    # x:   (B, T, H, Dh)
    # cos: (T, Dh)
    return (x * cos[None, :, None, :]) + (rotate_half(x) * sin[None, :, None, :])


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        weight = self.param(
            "weight", nn.initializers.ones, (self.hidden_size,), self.param_dtype
        )
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        scale = jax.lax.rsqrt(var + self.eps).astype(self.dtype)
        return x.astype(self.dtype) * scale * weight.astype(self.dtype)


def repeat_kv(h: Array, n_rep: int) -> Array:
    """Expand kv heads for GQA: (B, T, Hkv, Dh) -> (B, T, Hkv*n_rep, Dh)."""
    if n_rep == 1:
        return h
    B, T, Hkv, Dh = h.shape
    h = jnp.broadcast_to(h[:, :, :, None, :], (B, T, Hkv, n_rep, Dh))
    return h.reshape(B, T, Hkv * n_rep, Dh)




# Active mesh for shard_map'd Pallas kernels. Training code sets this before building the model.
_ACTIVE_MESH = None

def set_active_mesh(mesh):
    global _ACTIVE_MESH
    _ACTIVE_MESH = mesh

class Qwen3Attention(nn.Module):
    cfg: Qwen3Config

    @nn.compact
    def __call__(self, x: Array, attn_mask: Array, cos: Array, sin: Array) -> Array:
        c = self.cfg
        B, T, _ = x.shape
        Hq, Hkv, Dh = c.num_attention_heads, c.num_key_value_heads, c.head_dim

        # q: (B, T, Hq*Dh); k,v: (B, T, Hkv*Dh)
        q = nn.Dense(Hq * Dh,  use_bias=False, dtype=c.dtype, param_dtype=c.param_dtype, name="q_proj")(x)
        k = nn.Dense(Hkv * Dh, use_bias=False, dtype=c.dtype, param_dtype=c.param_dtype, name="k_proj")(x)
        v = nn.Dense(Hkv * Dh, use_bias=False, dtype=c.dtype, param_dtype=c.param_dtype, name="v_proj")(x)

        q = q.reshape(B, T, Hq, Dh)
        k = k.reshape(B, T, Hkv, Dh)
        v = v.reshape(B, T, Hkv, Dh)

        if c.use_qk_norm:
            q = RMSNorm(Dh, c.rms_norm_eps, c.dtype, c.param_dtype, name="q_norm")(q)
            k = RMSNorm(Dh, c.rms_norm_eps, c.dtype, c.param_dtype, name="k_norm")(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: expand kv heads to match q heads
        n_rep = Hq // Hkv
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        if jax.default_backend() != "tpu":
            qf = q.astype(jnp.float32)
            kf = k.astype(jnp.float32)
            vf = v.astype(jnp.float32)
            scores = jnp.einsum("bthd,bshd->bhts", qf, kf) * (Dh ** -0.5)
            causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
            same_segment = (attn_mask[:, None, :, None] * attn_mask[:, None, None, :]).astype(bool)
            scores = jnp.where(causal & same_segment, scores, jnp.finfo(jnp.float32).min)
            probs = jax.nn.softmax(scores, axis=-1).astype(c.dtype)
            out = jnp.einsum("bhts,bshd->bthd", probs, vf.astype(c.dtype)).reshape(B, T, Hq * Dh)
            return nn.Dense(
                c.hidden_size,
                use_bias=False,
                dtype=c.dtype,
                param_dtype=c.param_dtype,
                name="o_proj",
            )(out)

        # Pallas TPU flash attention (Mosaic kernel) via shard_map: O(T) memory.
        # q,k,v are (B, T, H, Dh) -> transpose to (B, H, T, Dh) for Pallas layout.
        q_bhtd = q.transpose(0, 2, 1, 3)
        k_bhtd = k.transpose(0, 2, 1, 3)
        v_bhtd = v.transpose(0, 2, 1, 3)
        sid_i = attn_mask.astype(jnp.int32)
        _sm = float(1.0 / (Dh ** 0.5))

        def _flash_local(qq, kk, vv, s):
            return _pallas_flash_attention(
                qq, kk, vv,
                segment_ids=_SegmentIds(q=s, kv=s),
                causal=True,
                sm_scale=_sm,
            )

        assert _ACTIVE_MESH is not None, "Call set_active_mesh(mesh) before building Qwen3Model."
        out_bhtd = _shard_map(
            _flash_local,
            mesh=_ACTIVE_MESH,
            in_specs=(
                _P(None, None, None, None),
                _P(None, None, None, None),
                _P(None, None, None, None),
                _P(None, None),
            ),
            out_specs=_P(None, None, None, None),
            check_rep=False,
        )(q_bhtd, k_bhtd, v_bhtd, sid_i)
        # (B, H, T, Dh) -> (B, T, H*Dh)
        out = out_bhtd.transpose(0, 2, 1, 3).reshape(B, T, Hq * Dh)

        out = nn.Dense(c.hidden_size, use_bias=False, dtype=c.dtype, param_dtype=c.param_dtype, name="o_proj")(out)
        return out


class Qwen3MLP(nn.Module):
    cfg: Qwen3Config

    @nn.compact
    def __call__(self, x: Array) -> Array:
        c = self.cfg
        gate = nn.Dense(c.intermediate_size, use_bias=False, dtype=c.dtype, param_dtype=c.param_dtype, name="gate_proj")(x)
        up   = nn.Dense(c.intermediate_size, use_bias=False, dtype=c.dtype, param_dtype=c.param_dtype, name="up_proj")(x)
        hid  = jax.nn.silu(gate) * up
        return nn.Dense(c.hidden_size, use_bias=False, dtype=c.dtype, param_dtype=c.param_dtype, name="down_proj")(hid)


class Qwen3DecoderLayer(nn.Module):
    cfg: Qwen3Config

    @nn.compact
    def __call__(self, x: Array, attn_mask: Array, cos: Array, sin: Array) -> Array:
        c = self.cfg
        h = RMSNorm(c.hidden_size, c.rms_norm_eps, c.dtype, c.param_dtype, name="input_layernorm")(x)
        h = Qwen3Attention(c, name="self_attn")(h, attn_mask, cos, sin)
        x = x + h
        h = RMSNorm(c.hidden_size, c.rms_norm_eps, c.dtype, c.param_dtype, name="post_attention_layernorm")(x)
        h = Qwen3MLP(c, name="mlp")(h)
        return x + h


class Qwen3Model(nn.Module):
    cfg: Qwen3Config

    @nn.compact
    def __call__(
        self,
        input_ids: Array,
        attention_mask: Array,
        return_hidden: bool = False,
        return_pooled_layers: Optional[Sequence[int]] = None,
        layer_pool: str = "mean",
    ) -> Array:
        c = self.cfg
        B, T = input_ids.shape

        embed = nn.Embed(c.vocab_size, c.hidden_size, dtype=c.dtype, param_dtype=c.param_dtype, name="embed_tokens")
        x = embed(input_ids)

        cos, sin = rotary_cos_sin(T, c)

        selected_layers = tuple(int(i) for i in (return_pooled_layers or ()))
        selected_set = set(selected_layers)
        pooled_features = []

        def _pool_hidden(h: Array) -> Array:
            mask = attention_mask.astype(jnp.float32)
            if (layer_pool or "mean").lower() == "last_token":
                last_idx = (mask.sum(axis=1) - 1).astype(jnp.int32).clip(min=0)
                batch_idx = jnp.arange(h.shape[0])
                return h[batch_idx, last_idx].astype(jnp.float32)
            mask_e = mask[..., None]
            summed = (h.astype(jnp.float32) * mask_e).sum(axis=1)
            count = mask.sum(axis=1, keepdims=True).clip(min=1.0)
            return summed / count

        for i in range(c.num_hidden_layers):
            x = nn.remat(Qwen3DecoderLayer)(c, name=f"layers_{i}")(x, attention_mask, cos, sin)
            if selected_set and i in selected_set:
                pooled_features.append(_pool_hidden(x))

        x = RMSNorm(c.hidden_size, c.rms_norm_eps, c.dtype, c.param_dtype, name="norm")(x)
        if selected_set:
            # Shape: (B, num_selected_layers, hidden). The caller keeps the
            # selected layer order stable for per-layer RND bookkeeping.
            return jnp.stack(pooled_features, axis=1).astype(jnp.float32)
        if return_hidden:
            return x

        # lm_head: tied to embed_tokens for Qwen3-1.7B
        if c.tie_word_embeddings:
            logits = x @ embed.variables["params"]["embedding"].T
        else:
            logits = nn.Dense(c.vocab_size, use_bias=False, dtype=c.dtype, param_dtype=c.param_dtype, name="lm_head")(x)
        return logits


# ---------------------------------------------------------------------------
# HF weight loading
# ---------------------------------------------------------------------------

def load_hf_weights_to_flax(model: Qwen3Model, hf_path_or_repo: str, dtype=jnp.float32):
    """Load HF checkpoint (PyTorch state_dict) and map into Flax `params` pytree.

    Implementation stub. Steps:
      1. Use `huggingface_hub.snapshot_download` to get safetensors.
      2. Open each shard with `safetensors.torch.load_file`.
      3. Map PyTorch parameter names -> our Flax tree paths.
      4. Transpose weight matrices to match nn.Dense convention (Dense is
         (in, out) in Flax but Linear is (out, in) in PyTorch).
      5. Cast to `dtype` and return as a frozen dict.

    TODO: implement with HF safetensors + param-name map. For now, raise.
    """
    raise NotImplementedError(
        "load_hf_weights_to_flax: M1 TODO — use safetensors to import Qwen3-1.7B weights "
        "and map into nested Flax params. Required before forward-pass parity test."
    )
