from __future__ import annotations

import os

import pytest

from tmx.orchestration.optimization import (
    MeshShape,
    apply_runtime_env,
    compute_mesh_shape,
)
from tmx.orchestration.spec import TopologySpec


def _save_env(*keys: str) -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in keys}


def _restore_env(saved: dict[str, str | None]) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_apply_runtime_env_respects_user_overrides() -> None:
    saved = _save_env("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR")
    try:
        os.environ["XLA_FLAGS"] = "--user_override=1"
        os.environ["JAX_COMPILATION_CACHE_DIR"] = "/custom"
        delta = apply_runtime_env(aggressive=False)
        assert "XLA_FLAGS" not in delta
        assert "JAX_COMPILATION_CACHE_DIR" not in delta
        assert os.environ["XLA_FLAGS"] == "--user_override=1"
        assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "/custom"
    finally:
        _restore_env(saved)


def test_apply_runtime_env_writes_defaults_when_unset() -> None:
    saved = _save_env(
        "XLA_FLAGS",
        "JAX_COMPILATION_CACHE_DIR",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "TMX_APPLY_TPU_RUNTIME_ENV",
    )
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ["TMX_APPLY_TPU_RUNTIME_ENV"] = "1"
        delta = apply_runtime_env(aggressive=False)
        assert "XLA_FLAGS" in delta
        assert "JAX_COMPILATION_CACHE_DIR" in delta
        assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
        assert "async_collective_fusion" in os.environ["XLA_FLAGS"]
    finally:
        _restore_env(saved)


def test_apply_runtime_env_aggressive_adds_flags() -> None:
    saved = _save_env("XLA_FLAGS", "TMX_APPLY_TPU_RUNTIME_ENV")
    try:
        os.environ.pop("XLA_FLAGS", None)
        os.environ["TMX_APPLY_TPU_RUNTIME_ENV"] = "1"
        apply_runtime_env(aggressive=True)
        assert "latency_hiding_scheduler" in os.environ["XLA_FLAGS"]
    finally:
        _restore_env(saved)


def test_apply_runtime_env_skips_tpu_xla_flags_locally() -> None:
    saved = _save_env("XLA_FLAGS", "JAX_PLATFORMS", "TMX_APPLY_TPU_RUNTIME_ENV")
    try:
        for k in saved:
            os.environ.pop(k, None)
        delta = apply_runtime_env(aggressive=False)
        assert "XLA_FLAGS" not in delta
        assert "JAX_PLATFORMS" not in delta
    finally:
        _restore_env(saved)


def test_compute_mesh_shape_full_fsdp() -> None:
    topo = TopologySpec(
        total_workers=16,
        chips_per_worker=4,
        train_workers=tuple(range(4)),
        rollout_workers=tuple(range(4, 16)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    )
    mesh = compute_mesh_shape(topo, tensor_parallel=1)
    assert mesh == MeshShape(data=1, fsdp=16, tensor=1)
    assert mesh.total == 16


def test_compute_mesh_shape_with_tp() -> None:
    topo = TopologySpec(
        total_workers=64,
        chips_per_worker=4,
        train_workers=tuple(range(16)),
        rollout_workers=tuple(range(16, 64)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    )
    mesh = compute_mesh_shape(topo, tensor_parallel=4)
    assert mesh.fsdp == 16
    assert mesh.tensor == 4
    assert mesh.total == 64


def test_compute_mesh_shape_rejects_indivisible_tp() -> None:
    topo = TopologySpec(
        total_workers=8,
        chips_per_worker=4,
        train_workers=(0, 1, 2),
        rollout_workers=tuple(range(3, 8)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    )
    with pytest.raises(ValueError):
        compute_mesh_shape(topo, tensor_parallel=8)


def test_compute_mesh_shape_rejects_zero_tp() -> None:
    topo = TopologySpec(
        total_workers=4,
        chips_per_worker=4,
        train_workers=(0,),
        rollout_workers=(1, 2, 3),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    )
    with pytest.raises(ValueError):
        compute_mesh_shape(topo, tensor_parallel=0)
