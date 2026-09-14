"""Additional TPU network/topology presets beyond the v5litepod-256 default.

This module extends ``topologies.NETWORK_PRESETS`` and ``TOPOLOGY_PRESETS``
without touching the codex-authored base file. Callers must invoke
``register_extras()`` to opt in.
"""

from __future__ import annotations

from .safety import TRC_PROJECT
from .spec import NetworkSpec, TopologySpec
from .topologies import NETWORK_PRESETS, TOPOLOGY_PRESETS


def _v6e_64_split() -> TopologySpec:
    return TopologySpec(
        total_workers=16,
        chips_per_worker=4,
        train_workers=tuple(range(4)),
        rollout_workers=tuple(range(4, 16)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    )


def _v6e_256_split() -> TopologySpec:
    return TopologySpec(
        total_workers=64,
        chips_per_worker=4,
        train_workers=tuple(range(16)),
        rollout_workers=tuple(range(16, 64)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    )


def _v5p_128_split() -> TopologySpec:
    return TopologySpec(
        total_workers=32,
        chips_per_worker=4,
        train_workers=tuple(range(8)),
        rollout_workers=tuple(range(8, 32)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    )


_EXTRA_NETWORKS: dict[str, NetworkSpec] = {
    "v6e-64-uc1a": NetworkSpec(
        project="",
        zone="us-central1-a",
        tpu_name="",
        accelerator_type="v6e-64",
    ),
    "trc2-v6e256-ew4b": NetworkSpec(
        project=TRC_PROJECT,
        zone="europe-west4-b",
        tpu_name="trc2-v6e256-ew4b-r1",
        accelerator_type="v6e-256",
    ),
    "trc2-v5p128-ew4b": NetworkSpec(
        project=TRC_PROJECT,
        zone="europe-west4-b",
        tpu_name="trc2-v5p128-ew4b-r1",
        accelerator_type="v5p-128",
    ),
}

_EXTRA_TOPOLOGIES: dict[str, TopologySpec] = {
    "v6e-64-4train-12rollout": _v6e_64_split(),
    "v6e-256-16train-48rollout": _v6e_256_split(),
    "v5p-128-8train-24rollout": _v5p_128_split(),
}


def register_extras() -> tuple[list[str], list[str]]:
    """Add extra presets to the orchestration registry.

    Idempotent. Returns the (networks_added, topologies_added) names so the
    caller can log what was actually registered.
    """
    nets: list[str] = []
    for name, spec in _EXTRA_NETWORKS.items():
        if name not in NETWORK_PRESETS:
            NETWORK_PRESETS[name] = spec
            nets.append(name)
    tops: list[str] = []
    for name, spec in _EXTRA_TOPOLOGIES.items():
        if name not in TOPOLOGY_PRESETS:
            TOPOLOGY_PRESETS[name] = spec
            tops.append(name)
    return nets, tops
