"""
Exploration-Distillation original torch_xla pilot package (Scout + Central exploration RL).

Modules:
  - config: argument parsing and run config dataclass.
  - devices: device resolution and TPU-safe optimizer steps.
  - data: GSM8K processing and distillation dataset builder.
  - embeddings: helper to get LM embeddings for novelty.
  - rnd: Random Network Distillation module.
  - reward: correctness + novelty reward helpers.
  - explorer: GRPO training and trajectory generation.
  - distill: distillation training for the Actual model.
  - eval_plot: evaluation + plotting utilities.
  - novelty_eval: embedding-space novelty/diversity analysis.
  - pipeline: end-to-end runner.
"""

__all__ = [
    "config",
    "devices",
    "data",
    "embeddings",
    "rnd",
    "reward",
    "explorer",
    "distill",
    "eval_plot",
    "novelty_eval",
    "pipeline",
]
