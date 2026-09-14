from __future__ import annotations

from tmx.orchestration import topologies as base
from tmx.orchestration.topologies_extra import register_extras


def test_register_extras_adds_v6e_and_v5p_presets() -> None:
    register_extras()
    assert "v6e-64-uc1a" in base.NETWORK_PRESETS
    assert "trc2-v6e256-ew4b" in base.NETWORK_PRESETS
    assert "trc2-v5p128-ew4b" in base.NETWORK_PRESETS
    assert "v6e-64-4train-12rollout" in base.TOPOLOGY_PRESETS
    assert "v6e-256-16train-48rollout" in base.TOPOLOGY_PRESETS
    assert "v5p-128-8train-24rollout" in base.TOPOLOGY_PRESETS


def test_register_extras_is_idempotent() -> None:
    register_extras()
    nets_before = dict(base.NETWORK_PRESETS)
    tops_before = dict(base.TOPOLOGY_PRESETS)
    register_extras()
    assert nets_before == base.NETWORK_PRESETS
    assert tops_before == base.TOPOLOGY_PRESETS


def test_v6e_topology_split_is_disjoint() -> None:
    register_extras()
    topo = base.TOPOLOGY_PRESETS["v6e-256-16train-48rollout"]
    train = set(topo.train_workers)
    rollout = set(topo.rollout_workers)
    assert not (train & rollout)
    assert train | rollout == set(range(topo.total_workers))


def test_v6e_topology_chip_count() -> None:
    register_extras()
    topo = base.TOPOLOGY_PRESETS["v6e-256-16train-48rollout"]
    assert topo.train_chip_count == 64
    assert topo.rollout_chip_count == 192
