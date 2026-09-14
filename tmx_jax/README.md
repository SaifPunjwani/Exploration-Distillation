# tmx_jax — JAX/TPU pipeline for Two-Model Exploration

Native JAX implementation of the Explorer → SFT → Actual GRPO → AIME24 pipeline.
Designed for v4/v5e/v6e TPUs with FSDP sharding from day-1 (picodo-style).

Status: **scaffolding + core modules in place. Runnable for single-GPU smoke test once deps installed; multi-host TPU parity TBD.**

## Why a rewrite?
The torch-xla path accumulated workarounds:
- FSDP/SPMD NaN on v6e, Pallas Flash-Attn NaN, libtpu pinning quirks
- 20-30 min XLA compile on first step, stale vLLM logprobs causing KL blow-up

JAX's GSPMD + NamedSharding gives cleaner FSDP; Flax gives clean model code.

## Contract (from `memory/project_serious_run_contract.md`)
- Qwen/Qwen3-1.7B on DAPO-Math-17k (no filter)
- group_size=8: batch_size=1, grad_accum=2, num_generations=8
- MAX_PROMPT_LEN=2048, MAX_COMPLETION_LEN=8192, MAX_TOTAL_LEN=10240
- DR-GRPO loss, multilayer novelty [7,14,21], full_trace SFT
- Thinking mode (auto); reasoning_final response style
- Final eval: AIME24 avg@32 (32 rollouts × 32k completion), target Actual > Base=49.1%

## Quickstart (TPU)
```bash
# On TPU VM (v4-8 / v5e-64 / v6e-64):
bash tmx_jax/scripts/setup_jax_tpu.sh

# Start vLLM on a separate TPU slice (or same slice workers 1-15):
# (re-use existing tmx/scripts/run_vllm_server.sh — vLLM side unchanged)

# Launch training:
TMX_VLLM_SERVER_URLS="http://<vllm_ip>:8000/v1" \
LAMBDA_NOVELTY=0 \
RUN_NAME=jax_lambda0_$(date +%s) \
bash tmx_jax/scripts/launch_jax.sh
```

## Modules
- `mesh.py` — 1-D FSDP mesh over all devices; param/activation sharding helpers
- `model.py` — Qwen3-1.7B in Flax (RMSNorm + RoPE + GQA + SwiGLU)
- `weights.py` — HF safetensors → Flax params (handles PyTorch (out,in) → Flax (in,out) transpose)
- `grpo.py` — DR-GRPO loss (PPO-clip with max_completion_len normalization)
- `generate.py` — vLLM HTTP client, pure Python (no torch dependency)
- `data.py` — DAPO loader, Qwen3 chat-template rendering, reward computation
- `config.py` — TrainConfig + CLI parsing
- `train.py` — Explorer training loop (rollout → rescore → DR-GRPO update)
- `scripts/setup_jax_tpu.sh`, `scripts/launch_jax.sh`

## TODO (tracked in `PLAN.md`)
- M1 forward-pass parity vs HF PyTorch
- Novelty scorer (multilayer RND) in JAX
- SFT distillation module
- Actual online GRPO reuse
- Orbax checkpointing → GCS
- Multi-host mesh coordination
- AIME24 eval harness
