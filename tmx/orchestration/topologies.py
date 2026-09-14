from __future__ import annotations

from dataclasses import replace

from .safety import TRC_PROJECT
from .spec import NetworkSpec, TopologySpec, network_with_env_defaults


TOPOLOGY_PRESETS: dict[str, TopologySpec] = {
    "v5litepod-256-64train-192rollout": TopologySpec(
        total_workers=64,
        chips_per_worker=4,
        train_workers=tuple(range(16)),
        rollout_workers=tuple(range(16, 64)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    ),
    "v5litepod-256-16train-240rollout": TopologySpec(
        total_workers=64,
        chips_per_worker=4,
        train_workers=(0, 1, 2, 3),
        rollout_workers=tuple(range(4, 64)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    ),
    "v5litepod-256-4train-192rollout": TopologySpec(
        total_workers=64,
        chips_per_worker=4,
        train_workers=(0,),
        rollout_workers=tuple(range(16, 64)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    ),
    "v5litepod-64-16train-48rollout": TopologySpec(
        total_workers=16,
        chips_per_worker=4,
        train_workers=tuple(range(4)),
        rollout_workers=tuple(range(4, 16)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    ),
    "v5litepod-64-4train-60rollout": TopologySpec(
        total_workers=16,
        chips_per_worker=4,
        train_workers=(0,),
        rollout_workers=tuple(range(1, 16)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    ),
    "v6e-64-16train-48rollout": TopologySpec(
        total_workers=16,
        chips_per_worker=4,
        train_workers=tuple(range(4)),
        rollout_workers=tuple(range(4, 16)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    ),
    "v6e-64-4train-60rollout": TopologySpec(
        total_workers=16,
        chips_per_worker=4,
        train_workers=(0,),
        rollout_workers=tuple(range(1, 16)),
        train_chips=(0, 1, 2, 3),
        rollout_chips=(0, 1, 2, 3),
    ),
}


# Network presets carry the zone and accelerator class only. The GCP project
# comes from TMX_TRC_PROJECT (TRC presets) or TPU_PROJECT, and the TPU name
# from TPU_NAME or --tpu-name, when a spec is created; nothing personal is
# baked in. TRC presets keep the nanny-created TPU naming their guard expects.
NETWORK_PRESETS: dict[str, NetworkSpec] = {
    "trc2-v5litepod-256-ew4b": NetworkSpec(
        project=TRC_PROJECT,
        zone="europe-west4-b",
        tpu_name="trc2-v5lite256-ew4b-r1",
        accelerator_type="v5litepod-256",
        external_ips_required=True,
        create_tpu=False,
        allow_nat=False,
        allow_internal_ips=False,
    ),
    "trc2-v5litepod-256-uc1a": NetworkSpec(
        project=TRC_PROJECT,
        zone="us-central1-a",
        tpu_name="trc2-v5lite256-uc1a-r1",
        accelerator_type="v5litepod-256",
        external_ips_required=True,
        create_tpu=False,
        allow_nat=False,
        allow_internal_ips=False,
    ),
    "v5litepod-64-uc1a": NetworkSpec(
        project="",
        zone="us-central1-a",
        tpu_name="",
        accelerator_type="v5litepod-64",
        external_ips_required=True,
        create_tpu=False,
        allow_nat=False,
        allow_internal_ips=False,
    ),
    "v5litepod-64-ew4b": NetworkSpec(
        project="",
        zone="europe-west4-b",
        tpu_name="",
        accelerator_type="v5litepod-64",
        external_ips_required=True,
        create_tpu=False,
        allow_nat=False,
        allow_internal_ips=False,
    ),
    "v6e-64-ew4a": NetworkSpec(
        project="",
        zone="europe-west4-a",
        tpu_name="",
        accelerator_type="v6e-64",
        external_ips_required=True,
        create_tpu=False,
        allow_nat=False,
        allow_internal_ips=False,
    ),
    "v6e-64-use1d": NetworkSpec(
        project="",
        zone="us-east1-d",
        tpu_name="",
        accelerator_type="v6e-64",
        external_ips_required=True,
        create_tpu=False,
        allow_nat=False,
        allow_internal_ips=False,
    ),
}


def topology_preset(name: str) -> TopologySpec:
    try:
        return TOPOLOGY_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(TOPOLOGY_PRESETS))
        raise ValueError(f"unknown topology preset {name!r}; available: {available}") from exc


def network_preset(name: str, *, tpu_name: str | None = None) -> NetworkSpec:
    try:
        spec = NETWORK_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(NETWORK_PRESETS))
        raise ValueError(f"unknown network preset {name!r}; available: {available}") from exc
    if tpu_name:
        spec = replace(spec, tpu_name=tpu_name)
    return network_with_env_defaults(spec)
