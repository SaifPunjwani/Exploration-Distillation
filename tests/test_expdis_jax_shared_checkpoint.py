from pathlib import Path

import jax.numpy as jnp
import pytest

from expdis_jax.checkpointing import checkpoint_path, save_shared_checkpoint
from expdis_jax.config import TrainConfig


def test_shared_checkpoint_reuse_requires_a_committed_write(tmp_path):
    cfg = TrainConfig(output_dir=str(tmp_path / "local"), checkpoint_root=str(tmp_path / "shared"))
    path = checkpoint_path(cfg, "step_000001")
    Path(path).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="incomplete shared checkpoint"):
        save_shared_checkpoint(cfg, "step_000001", {"params": {"w": jnp.ones(2)}, "step": 1})
    Path(path).rmdir()
    assert save_shared_checkpoint(cfg, "step_000001", {"params": {"w": jnp.ones(2)}, "step": 1}) == path
    assert save_shared_checkpoint(cfg, "step_000001", {"params": {"w": jnp.ones(2)}, "step": 1}) == path
