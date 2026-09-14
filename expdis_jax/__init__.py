"""JAX-native TPU implementation of Exploration-Distillation.

Contract (see config.py and the paper's Appendix B):
  - Qwen3-1.7B on DAPO-Math-17k (also Qwen3-4B and text-only Ministral-3-3B)
  - group size 16, 4 prompts per update (64 rollouts/update), 1 update per rollout batch
  - MAX_PROMPT_LEN=2048, MAX_COMPLETION_LEN=32768, MAX_TOTAL_LEN=34816
  - Dr.GRPO loss (mean-centered advantages, fixed-length denominator), DAPO
    dynamic sampling and soft-overlong penalty (onset 26214, ramp 6554), KL 0
  - multilayer RND novelty on the Explorer only, credited on correct completions
  - QualityPool filter -> Main-model SFT -> correctness-only GRPO -> eval

Design:
  - Flax port of Qwen3 (RMSNorm, RoPE, GQA, SwiGLU) with HF weight loading
  - 1-D FSDP mesh with NamedSharding; jit with explicit shardings
  - vLLM HTTP client for generation; logprobs recomputed on the training policy
  - Optax AdamW; Orbax checkpoints
"""

__version__ = "0.1.0"
