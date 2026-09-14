"""Smoke test: single DR-GRPO train step on random rollout.

Proper FSDP sharding of params AND opt_state via `jax.jit(init_fn, out_shardings=...)`.
Validates: grpo.py, mesh, optax gradient flow, chipwise memory split.
"""
import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P

from tmx_jax.mesh import FSDP_AXIS, build_mesh, named, shardings_for_params
from tmx_jax.model import Qwen3Config, Qwen3Model, set_active_mesh
from tmx_jax.weights import hf_to_flax_params, load_hf_config
from tmx_jax.grpo import GrpoConfig, compute_group_advantages, compute_per_token_logps, dr_grpo_loss


def main():
    t0 = time.perf_counter()
    mesh = build_mesh()
    set_active_mesh(mesh)
    print(f"mesh {mesh}", flush=True)

    cfg_hf = load_hf_config("Qwen/Qwen3-1.7B")
    m_cfg = Qwen3Config(
        hidden_size=int(cfg_hf["hidden_size"]),
        intermediate_size=int(cfg_hf["intermediate_size"]),
        num_hidden_layers=int(cfg_hf["num_hidden_layers"]),
        num_attention_heads=int(cfg_hf["num_attention_heads"]),
        num_key_value_heads=int(cfg_hf["num_key_value_heads"]),
        head_dim=int(cfg_hf.get("head_dim", cfg_hf["hidden_size"] // cfg_hf["num_attention_heads"])),
        rope_theta=float(cfg_hf.get("rope_theta", 1_000_000.0)),
        rms_norm_eps=float(cfg_hf.get("rms_norm_eps", 1e-6)),
        vocab_size=int(cfg_hf["vocab_size"]),
        max_position_embeddings=int(cfg_hf.get("max_position_embeddings", 40960)),
        tie_word_embeddings=bool(cfg_hf.get("tie_word_embeddings", False)),
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )
    model = Qwen3Model(m_cfg)

    with mesh:
        # --- Load weights on host, then shard to devices ---
        params_cpu = hf_to_flax_params(
            "Qwen/Qwen3-1.7B", m_cfg.num_hidden_layers, m_cfg.tie_word_embeddings, dtype=jnp.float32
        )["params"]
        param_sh = shardings_for_params(params_cpu, mesh)

        # Jit-initialize: shards params + opt state together so AdamW moments
        # sit on the same chips as their params.
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=5e-6, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.0),
        )

        def init_fn(raw_params):
            return train_state.TrainState.create(apply_fn=model.apply, params=raw_params, tx=tx)

        sharded_params = jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), params_cpu, param_sh)
        state_shape = jax.eval_shape(init_fn, sharded_params)

        def infer_leaf_sh(leaf):
            if leaf.ndim >= 2:
                return NamedSharding(mesh, P(FSDP_AXIS, *([None] * (leaf.ndim - 1))))
            return NamedSharding(mesh, P())
        state_sh = jax.tree_util.tree_map(infer_leaf_sh, state_shape)
        # TrainState is flax.struct.dataclass → .replace works; override params subtree.
        state_sh = state_sh.replace(params=param_sh)

        jit_init = jax.jit(init_fn, out_shardings=state_sh)
        state = jit_init(sharded_params)
        print(f"state sharded ({time.perf_counter()-t0:.1f}s)", flush=True)

        # Fake rollout: B=4 rows, P=128, T=128.
        # Total sequence length must be divisible by the Pallas flash-attention
        # KV block size (128), matching the real 2048+8192 contract.
        rng = np.random.default_rng(0)
        B, P_len, T, NG = 4, 128, 128, 4
        full_ids = rng.integers(1, m_cfg.vocab_size, size=(B, P_len + T), dtype=np.int32)
        full_attn = np.ones((B, P_len + T), dtype=np.int32)
        comp_mask = np.ones((B, T), dtype=np.int32)
        old_lp = np.zeros((B, T), dtype=np.float32)
        rewards = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        advantages = np.array(compute_group_advantages(jnp.asarray(rewards), num_generations=NG))

        batch_sh = {
            "full_input_ids": named(mesh, P(FSDP_AXIS, None)),
            "full_attention_mask": named(mesh, P(FSDP_AXIS, None)),
            "completion_mask": named(mesh, P(FSDP_AXIS, None)),
            "old_per_token_logps": named(mesh, P(FSDP_AXIS, None)),
            "advantages": named(mesh, P(FSDP_AXIS)),
        }
        batch = {
            "full_input_ids": jax.device_put(jnp.asarray(full_ids), batch_sh["full_input_ids"]),
            "full_attention_mask": jax.device_put(jnp.asarray(full_attn), batch_sh["full_attention_mask"]),
            "completion_mask": jax.device_put(jnp.asarray(comp_mask), batch_sh["completion_mask"]),
            "old_per_token_logps": jax.device_put(jnp.asarray(old_lp), batch_sh["old_per_token_logps"]),
            "advantages": jax.device_put(jnp.asarray(advantages), batch_sh["advantages"]),
        }
        grpo_cfg = GrpoConfig(clip_epsilon=0.2, max_completion_len=T, kl_beta=0.0)

        def train_step(state, batch):
            def loss_fn(params):
                logits = state.apply_fn({"params": params}, batch["full_input_ids"], batch["full_attention_mask"])
                shift_logits = logits[:, P_len - 1 : -1, :]
                comp_ids = batch["full_input_ids"][:, P_len:]
                loss, metrics = dr_grpo_loss(
                    shift_logits, comp_ids, batch["completion_mask"],
                    batch["old_per_token_logps"], batch["advantages"], grpo_cfg,
                )
                return loss, metrics
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, metrics

        step_jit = jax.jit(train_step, out_shardings=(state_sh, None))
        print(f"compiling first step ({time.perf_counter()-t0:.1f}s)...", flush=True)
        state, metrics = step_jit(state, batch)
        jax.block_until_ready(metrics["loss"])
        print(f"step 1 done ({time.perf_counter()-t0:.1f}s) loss={float(metrics['loss']):.4f} clip={float(metrics['clip_fraction']):.3f} kl={float(metrics['approx_kl']):.3e}", flush=True)

        t1 = time.perf_counter()
        state, metrics = step_jit(state, batch)
        jax.block_until_ready(metrics["loss"])
        print(f"step 2 done ({time.perf_counter()-t1:.1f}s) loss={float(metrics['loss']):.4f}", flush=True)
        print("OK", flush=True)


if __name__ == "__main__":
    main()
