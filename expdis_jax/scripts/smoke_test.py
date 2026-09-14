"""End-to-end smoke test:
- Build FSDP mesh
- Load Qwen3-1.7B HF weights → Flax pytree
- Apply FSDP shardings
- Forward pass on random input (B=2, T=128)
- Print logits shape + sample values
"""
import time
import jax
import jax.numpy as jnp
import numpy as np

from expdis_jax.mesh import build_mesh, shardings_for_params
from expdis_jax.model import Qwen3Config, Qwen3Model, set_active_mesh
from expdis_jax.weights import hf_to_flax_params, load_hf_config


def main():
    t0 = time.perf_counter()
    devs = jax.devices()
    print(f"devices={len(devs)} platform={devs[0].platform}", flush=True)
    mesh = build_mesh()
    set_active_mesh(mesh)
    print(f"mesh built ({time.perf_counter()-t0:.1f}s): {mesh}", flush=True)

    cfg_hf = load_hf_config("Qwen/Qwen3-1.7B")
    hs = cfg_hf["hidden_size"]
    nl = cfg_hf["num_hidden_layers"]
    nh = cfg_hf["num_attention_heads"]
    nkv = cfg_hf["num_key_value_heads"]
    vocab = cfg_hf["vocab_size"]
    tie = bool(cfg_hf.get("tie_word_embeddings", False))
    print(f"qwen3 config: hidden={hs} layers={nl} heads={nh} kv={nkv} vocab={vocab} tie={tie} ({time.perf_counter()-t0:.1f}s)", flush=True)

    params = hf_to_flax_params("Qwen/Qwen3-1.7B", nl, tie, dtype=jnp.float32)
    print(f"weights loaded ({time.perf_counter()-t0:.1f}s) top keys: {sorted(params['params'].keys())[:6]}", flush=True)

    m_cfg = Qwen3Config(
        hidden_size=int(hs),
        intermediate_size=int(cfg_hf["intermediate_size"]),
        num_hidden_layers=int(nl),
        num_attention_heads=int(nh),
        num_key_value_heads=int(nkv),
        head_dim=int(cfg_hf.get("head_dim", hs // nh)),
        rope_theta=float(cfg_hf.get("rope_theta", 1_000_000.0)),
        rms_norm_eps=float(cfg_hf.get("rms_norm_eps", 1e-6)),
        vocab_size=int(vocab),
        max_position_embeddings=int(cfg_hf.get("max_position_embeddings", 40960)),
        tie_word_embeddings=tie,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )
    model = Qwen3Model(m_cfg)

    # Apply FSDP sharding
    with mesh:
        param_sh = shardings_for_params(params["params"], mesh)
        sharded_params = jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), params["params"], param_sh)
        print(f"params sharded ({time.perf_counter()-t0:.1f}s)", flush=True)

        # Random inputs
        rng = np.random.default_rng(0)
        B, T = 2, 128
        input_ids = jnp.asarray(rng.integers(0, vocab, size=(B, T), dtype=np.int32))
        attn = jnp.ones((B, T), dtype=jnp.int32)
        print(f"forward starting ({time.perf_counter()-t0:.1f}s)", flush=True)

        fwd = jax.jit(lambda p, ids, mask: model.apply({"params": p}, ids, mask))
        logits = fwd(sharded_params, input_ids, attn)
        logits.block_until_ready()
        print(f"forward done ({time.perf_counter()-t0:.1f}s) logits shape={logits.shape} dtype={logits.dtype}", flush=True)
        print(f"logits sample row 0 head 5: {logits[0, 0, :5]}", flush=True)
        print("OK", flush=True)


if __name__ == "__main__":
    main()
