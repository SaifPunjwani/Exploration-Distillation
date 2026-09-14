"""FSDP mesh + sharding helpers (picodo-style: single ``fsdp`` axis over all devices).

Design follows https://docs.jax.dev/en/latest/the-training-cookbook.html#fully-sharded-data-parallel-fsdp
and https://github.com/tingtang2/picodo.

Summary:
  * 1-D mesh over every visible TPU chip.
  * Parameters sharded along the ``fsdp`` axis (2D matrices -> shard first
    axis; 1D scalars -> replicate).
  * Activations: batch axis sharded on ``fsdp`` so micro-batches are
    distributed and then gathered only inside a layer.
  * Optimizer state sharded identically to params (standard FSDP).
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


FSDP_AXIS = "fsdp"


def _device_process_index(device: Any) -> int:
    value = getattr(device, "process_index", 0)
    return int(value() if callable(value) else value)


def build_mesh(axis_name: str = FSDP_AXIS) -> Mesh:
    """Create a 1-D mesh over *every* visible JAX device.

    Works on v4-8 (8 chips), v5e-64 (64 chips), v6e-64 (64 chips), multi-host
    slices, and single GPU. Use the resulting mesh as ``with mesh:`` or pass
    its spec to ``NamedSharding``.
    """
    all_devices = list(jax.devices())
    if jax.process_count() > 1:
        # host_local_array_to_global_array requires each host's local devices
        # to occupy a contiguous slice of the global mesh. TPU topology-aware
        # ordering can interleave hosts, so use process-major order for FSDP.
        devices = np.array(
            sorted(
                all_devices,
                key=lambda d: (_device_process_index(d), int(getattr(d, "id", 0))),
            ),
            dtype=object,
        )
    else:
        devices = mesh_utils.create_device_mesh((len(all_devices),))
    return Mesh(devices, axis_names=(axis_name,))


def named(mesh: Mesh, spec: P) -> NamedSharding:
    return NamedSharding(mesh, spec)


def shardings_dict(mesh: Mesh, fsdp_axis: str = FSDP_AXIS) -> Dict[str, NamedSharding]:
    """Named shardings for the common Flax param/activation shapes."""
    return {
        # Parameters (leading-axis FSDP)
        "embed":          named(mesh, P(fsdp_axis, None)),        # (vocab, hidden)
        "embed_hidden":   named(mesh, P(None, fsdp_axis)),        # fallback when vocab is not divisible by mesh
        "kernel_in":      named(mesh, P(fsdp_axis, None)),        # Dense (in,out): shard in
        "kernel_out":     named(mesh, P(None, fsdp_axis)),        # Dense (in,out): shard out (down/o_proj)
        "norm":           named(mesh, P()),                        # (hidden,) replicate
        # Activations
        "batch":          named(mesh, P(fsdp_axis, None)),         # (B, T) shard B
        "batch_seq_h":    named(mesh, P(fsdp_axis, None, None)),   # (B, T, H) shard B
        "batch_seq_heads_dh": named(mesh, P(fsdp_axis, None, None, None)),  # (B, T, Hq, Dh)
        "replicated":     named(mesh, P()),
    }


def shardings_for_params(params: Any, mesh: Mesh, fsdp_axis: str = FSDP_AXIS):
    """Return a pytree of NamedSharding matching `params` shapes.

    Rules:
      * ``embed_tokens.embedding``        -> shard axis 0 (vocab), unless the
        vocab dimension is not divisible by the mesh; then shard hidden dim.
      * ``self_attn.{q,k,v}_proj.kernel`` -> shard axis 0 (input-hidden)
      * ``self_attn.o_proj.kernel``       -> shard axis 1 (output-hidden)
      * ``mlp.gate_proj.kernel``          -> shard axis 0
      * ``mlp.up_proj.kernel``            -> shard axis 0
      * ``mlp.down_proj.kernel``          -> shard axis 1
      * ``*.weight`` (1-D norms)          -> replicate
    """
    sds = shardings_dict(mesh, fsdp_axis)

    def rule(path, leaf):
        name = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        if leaf.ndim == 1:
            return sds["norm"]
        if name.endswith("embedding"):
            if int(leaf.shape[0]) % int(mesh.size) == 0:
                return sds["embed"]
            if leaf.ndim >= 2 and int(leaf.shape[1]) % int(mesh.size) == 0:
                return sds["embed_hidden"]
            return sds["replicated"]
        if name.endswith("o_proj/kernel") or name.endswith("down_proj/kernel"):
            return sds["kernel_out"]
        if "kernel" in name:
            return sds["kernel_in"]
        # Fallback
        return sds["kernel_in"] if leaf.ndim == 2 else sds["replicated"]

    return jax.tree_util.tree_map_with_path(rule, params)
