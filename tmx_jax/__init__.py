"""JAX-native TPU pipeline for two-model-exploration.

Clean reimplementation targeting TPU (v4/v5e/v6e) without torch-xla.
Built for the serious-run contract:
  - Qwen3-1.7B, DAPO-Math-17k, group_size=8 (batch=1 x grad_accum=2 x num_gen=8)
  - MAX_PROMPT_LEN=2048, MAX_COMPLETION_LEN=8192, MAX_TOTAL_LEN=10240
  - DR-GRPO loss, multilayer novelty (layers 7,14,21), full_trace SFT
  - Goal: Actual > Base (49.1%) on AIME24 avg@32

Design:
  - Flax port of Qwen3 (RMSNorm, RoPE, SwiGLU) with HF weight loading
  - pjit + shard_map for mesh parallelism (avoids torch-xla SPMD bugs)
  - vLLM HTTP client for generation (reuse existing tmx/vllm_generate.py)
  - On-device logprob rescoring each round (eliminates stale-vLLM KL blow-up)
  - Optax AdamW; Orbax for checkpoints
"""

__version__ = "0.1.0"
