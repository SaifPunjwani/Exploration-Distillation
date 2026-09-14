"""Tuned env-var bundles and mesh-shape calculators for the production pipeline.

Pure Python; safe to import before JAX is loaded. Workload entrypoints
should call ``apply_runtime_env()`` first, then import jax/maxtext/tunix so
that XLA picks up the flags.

Mirrors infra/xpk/optimization_env.sh — keep the two in sync.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

from .spec import TopologySpec

_XLA_FLAGS_BASE = (
    "--xla_tpu_enable_async_collective_fusion=true",
    "--xla_tpu_enable_async_all_gather=true",
    "--xla_tpu_enable_megacore_fusion=true",
)
_XLA_FLAGS_AGGRESSIVE = (
    "--xla_tpu_data_parallel_opt_different_sized_ops=true",
    "--xla_enable_async_collective_permute=true",
    "--xla_tpu_enable_latency_hiding_scheduler=true",
)


def _looks_like_tpu_runtime() -> bool:
    """Return True when TPU-only XLA flags are safe to install.

    The local macOS/conda test environment aborts if TPU-specific XLA flags
    are present before importing JAX. Remote TPU launchers export TPU_* vars,
    and callers can force this with TMX_APPLY_TPU_RUNTIME_ENV=1.
    """
    forced = os.environ.get("TMX_APPLY_TPU_RUNTIME_ENV")
    if forced is not None:
        return forced == "1"
    if platform.system().lower() == "darwin":
        return False
    return any(
        os.environ.get(key)
        for key in (
            "TPU_NAME",
            "TPU_ACCELERATOR_TYPE",
            "TPU_WORKER_ID",
            "TPU_HOST_BOUNDS",
            "LIBTPU_INIT_ARGS",
            "MEGASCALE_COORDINATOR_ADDRESS",
        )
    )


def apply_runtime_env(*, aggressive: bool | None = None) -> dict[str, str]:
    """Set tuned env vars in os.environ before any JAX import.

    Idempotent and override-respecting: anything already in os.environ wins.
    Returns the delta this call wrote, for logging.
    """
    delta: dict[str, str] = {}

    def _set(key: str, value: str) -> None:
        if key not in os.environ:
            os.environ[key] = value
            delta[key] = value

    _set("JAX_COMPILATION_CACHE_DIR", "/workspace/.xla_cache")
    _set("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    _set("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1.0")
    _set("TF_CPP_MIN_LOG_LEVEL", "2")
    _set("JAX_TRACEBACK_FILTERING", "off")
    _set("HF_HUB_ENABLE_HF_TRANSFER", "1")
    _set("TOKENIZERS_PARALLELISM", "false")

    if _looks_like_tpu_runtime():
        _set("JAX_PLATFORMS", "tpu,cpu")
        if aggressive is None:
            aggressive = os.environ.get("TMX_AGGRESSIVE_XLA") == "1"
        flags = list(_XLA_FLAGS_BASE)
        if aggressive:
            flags.extend(_XLA_FLAGS_AGGRESSIVE)
        if "XLA_FLAGS" not in os.environ:
            os.environ["XLA_FLAGS"] = " ".join(flags)
            delta["XLA_FLAGS"] = os.environ["XLA_FLAGS"]
    return delta


def maybe_init_distributed_jax() -> dict[str, object]:
    """Initialize JAX multi-host coordinator if launched in multi-process mode.

    Convention used by ``infra/xpk/launch_trc256_full_pod.sh`` and any other
    multi-host launcher: every worker exports::

        JAX_NUM_PROCESSES=<N>
        JAX_PROCESS_ID=<0..N-1>             # one per worker
        JAX_COORDINATOR_ADDRESS=<leader_ip:port>

    When N>1 we call ``jax.distributed.initialize`` once with explicit args.
    This must run BEFORE any ``import maxtext``/``import tunix`` because both
    libraries take device snapshots on import.

    Idempotent: ``jax.distributed.initialize`` itself rejects double-init,
    but we guard with a process-local flag so re-entrant calls are safe.
    """
    if globals().get("_TMX_DIST_INITIALIZED"):
        return {"multi_host": True, "already_initialized": True}
    num_processes = int(os.environ.get("JAX_NUM_PROCESSES", "1") or "1")
    if num_processes <= 1:
        return {"multi_host": False, "num_processes": num_processes}
    process_id = int(os.environ.get("JAX_PROCESS_ID", "0"))
    coord = os.environ.get("JAX_COORDINATOR_ADDRESS", "").strip()
    if not coord:
        raise RuntimeError(
            "JAX_NUM_PROCESSES > 1 requires JAX_COORDINATOR_ADDRESS=host:port"
        )
    import jax  # local import keeps optimization.py importable on CPU

    jax.distributed.initialize(
        coordinator_address=coord,
        num_processes=num_processes,
        process_id=process_id,
    )
    globals()["_TMX_DIST_INITIALIZED"] = True
    return {
        "multi_host": True,
        "num_processes": num_processes,
        "process_id": process_id,
        "coordinator": coord,
    }


@dataclass(frozen=True)
class MeshShape:
    """FSDP+TP mesh derived from a TopologySpec.

    For a 1.7B model, fsdp == train_chip_count and tensor == 1 is the right
    default. Tensor parallelism is only worth the all-reduce cost for >7B.
    """

    data: int
    fsdp: int
    tensor: int

    @property
    def total(self) -> int:
        return self.data * self.fsdp * self.tensor


def compute_mesh_shape(topology: TopologySpec, *, tensor_parallel: int = 1) -> MeshShape:
    chips = topology.train_chip_count
    if chips <= 0:
        raise ValueError("topology has no train chips")
    if tensor_parallel < 1:
        raise ValueError("tensor_parallel must be >= 1")
    if chips % tensor_parallel != 0:
        raise ValueError(
            f"train chip count {chips} not divisible by tensor_parallel {tensor_parallel}"
        )
    return MeshShape(data=1, fsdp=chips // tensor_parallel, tensor=tensor_parallel)
