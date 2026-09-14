"""Shared stage checkpoints for distributed policy training.

Every process participates in array serialization. A caller must explicitly
provide a filesystem path mounted at the same location on every host, or a
gs:// prefix, for multi-host handoffs. Local single-host runs need no setting.
"""
from __future__ import annotations

import os
import pickle

import jax
import numpy as np
from jax.experimental import multihost_utils


def checkpoint_path(cfg, name: str) -> str:
    root = str(getattr(cfg, "checkpoint_root", "") or "").rstrip("/")
    if not root:
        return os.path.abspath(os.path.join(cfg.output_dir, name))
    run_root = str(getattr(cfg, "pipeline_root_dir", "") or cfg.output_dir)
    relative = os.path.relpath(os.path.abspath(cfg.output_dir), os.path.abspath(run_root))
    if relative == ".." or relative.startswith("../"):
        raise ValueError("stage output directory is outside pipeline_root_dir")
    parts = [] if relative == "." else [relative]
    return root + "/" + "/".join([*parts, name])


def broadcast_source_object(value):
    """Broadcast trusted host metadata with variable structure from process 0."""
    if jax.process_count() == 1:
        return value
    source = jax.process_index() == 0
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL) if source else b""
    length = int(multihost_utils.broadcast_one_to_all(
        np.array(len(payload), dtype=np.int64), is_source=source))
    chunks = []
    for start in range(0, length, 4 << 20):
        size = min(4 << 20, length - start)
        buffer = np.frombuffer(payload[start:start + size], dtype=np.uint8).copy() if source else np.zeros(size, np.uint8)
        chunks.append(np.asarray(multihost_utils.broadcast_one_to_all(buffer, is_source=source)).tobytes())
    return pickle.loads(b"".join(chunks))


def save_shared_checkpoint(cfg, name, payload):
    """Write one complete global checkpoint, synchronously, on all processes."""
    import orbax.checkpoint as ocp
    from etils import epath

    path = checkpoint_path(cfg, name)
    if jax.process_count() > 1 and not cfg.checkpoint_root:
        raise ValueError("multi-host stage checkpoints require --checkpoint-root on shared storage")
    # A final checkpoint can coincide with the periodic checkpoint at this step.
    exists = broadcast_source_object(epath.Path(path).exists() if jax.process_index() == 0 else None)
    with ocp.PyTreeCheckpointer() as checkpointer:
        if exists:
            # Do not mistake an interrupted write for a usable stage handoff.
            metadata = checkpointer.metadata(path)
            if (metadata.item_metadata is None or
                    metadata.commit_timestamp_nsecs is None):
                raise RuntimeError(f"incomplete shared checkpoint: {path}")
        else:
            checkpointer.save(path, payload)
    return path
