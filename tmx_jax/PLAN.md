# JAX-TPU rewrite plan

## Goal
Clean JAX pipeline for Explorer + SFT + Actual GRPO on TPU (v4/v5e/v6e).
Bypass torch-xla bugs (SPMD NaN, Pallas FA NaN, libtpu pin, XLA compile time).

## Non-negotiable contract
- Qwen3-1.7B, DAPO-Math-17k, no filter
- group_size=8 (batch=1, grad_accum=2, num_gen=8)
- MAX_PROMPT=2048, MAX_COMPLETION=8192, MAX_TOTAL=10240
- DR-GRPO loss; multilayer novelty [7,14,21]; full_trace SFT
- Actual > Base (49.1%) on AIME24 avg@32 (32 rollouts × 32k completion)

## Milestones
- [x] M0: Scaffold directory, branch `jax-tpu`, README/PLAN
- [ ] M1: Qwen3 in Flax with HF weight loading (match PyTorch logits within 1e-4)
- [ ] M2: vLLM HTTP client port from `tmx/vllm_generate.py`
- [ ] M3: Single forward+backward on 1 sample, verify numerics
- [ ] M4: DR-GRPO loss in JAX (matches Torch reference)
- [ ] M5: On-device rescoring (eliminates stale vLLM logprobs)
- [ ] M6: Training loop with pjit mesh + optax AdamW
- [ ] M7: Checkpoint save/load (Orbax)
- [ ] M8: Multi-host coordination (v5e-64 / v6e-64)
- [ ] M9: Trajectory buffer → SFT distillation (AdamW)
- [ ] M10: Actual online GRPO (reuse Explorer path)
- [ ] M11: AIME24 benchmark eval (32 rollouts × 32k)
- [ ] M12: End-to-end smoke run (50 steps)
- [ ] M13: Full pipeline run matching contract

## Key design decisions
1. **Mesh**: ('data', 'model') mesh. For single-host TPU v4-8: (2, 4). For v5e-64: (16, 4). For v6e-64: (16, 4).
2. **Sharding**:
   - Params: sharded across 'model' axis for attn heads / ffn, replicated across 'data'
   - Activations: sharded 'data' dim (for GRPO batch=1 this is effectively replicated)
   - Optimizer state: matches param sharding (with `optax.chain` + `shard_map`)
3. **vLLM for generation only** (run on a separate TPU slice; HTTP interface identical to current)
4. **Rescoring**: always re-score old_logprobs on the training model via a jit'd forward pass. Kills stale-vLLM KL bug by design.
5. **No gradient checkpointing by default** (TPU HBM is 16-32GB/chip; sharding across mesh should suffice). Add optional `jax.remat` if needed per-layer.

## Dependencies
- JAX/Jaxlib with TPU support (PJRT)
- Flax (nnx for cleaner code) or linen
- Optax AdamW
- Orbax (checkpoints to GCS)
- HuggingFace Transformers (tokenizer + weight loading utilities)

## Risks
- Qwen3 isn't in HF Flax as of transformers 4.57 — need to port (similar to Qwen2 Flax if available, else from FlaxLlama arch)
- DR-GRPO must match PyTorch exactly (reference: `tmx/explorer.py::_train_explorer_from_rollout_groups`)
- Multi-host mesh configuration is subtle
