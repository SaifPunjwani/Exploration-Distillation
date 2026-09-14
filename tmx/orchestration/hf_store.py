from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HfArtifactLayout:
    model_repo: str
    dataset_repo: str
    run_prefix: str

    def checkpoint_path(self, phase: str, step: int) -> str:
        return f"{self.run_prefix}/{phase}/checkpoints/step_{step:06d}"

    def rollout_shard_path(self, phase: str, step: int, shard: int) -> str:
        return f"{self.run_prefix}/{phase}/trajectories/step_{step:06d}/shard_{shard:05d}.jsonl"

    def config_path(self, name: str = "experiment_spec.json") -> str:
        return f"{self.run_prefix}/configs/{name}"

    def eval_path(self, name: str) -> str:
        return f"{self.run_prefix}/evals/{Path(name).name}"

    def plot_path(self, name: str) -> str:
        return f"{self.run_prefix}/plots/{Path(name).name}"
