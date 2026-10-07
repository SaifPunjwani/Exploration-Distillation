"""One local process of the distributed CPU pipeline integration test."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Limit CPU thread pools in small CI runners. Each process still owns its own
# JAX CPU device and participates in real Gloo collectives.
if hasattr(os, "sched_getaffinity"):
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[:2])
import jax
from jax._src.lib import xla_client

# macOS hostnames need not resolve in DNS. Bind this test's Gloo transport to
# loopback explicitly without modifying the machine's hosts configuration.
gloo_factory = xla_client._xla.make_gloo_tcp_collectives
xla_client._xla.make_gloo_tcp_collectives = lambda **kw: gloo_factory(**kw, hostname="127.0.0.1")
jax.distributed.initialize(coordinator_address=sys.argv[1], num_processes=2,
    process_id=int(sys.argv[2]), local_device_ids=[0], initialization_timeout=45,
    heartbeat_timeout_seconds=60)
import pytest
from jax.experimental import multihost_utils
from test_expdis_jax_pipeline_cpu import run_tiny_pipeline
from expdis_jax import train
from jax.sharding import PartitionSpec as P
import numpy as np
import time

# Deliberately let the peer request the next batch before publication. Values
# change on every call, so stale reads fail even if all model shapes coincide.
for generation in range(1, 4):
    if jax.process_index() == 0:
        time.sleep(0.15)
    batch = {"completion_mask": np.full((2, 1), generation, dtype=np.int32)}
    received = train._shard_batch_for_hosts(batch, {"completion_mask": P("fsdp", None)})
    np.testing.assert_array_equal(received["completion_mask"], [[generation]])
    multihost_utils.sync_global_devices(f"batch_transport_checked_{generation}")

root = Path(sys.argv[3])
host = root / f"host_{jax.process_index()}"
host.mkdir(exist_ok=True)
with pytest.MonkeyPatch.context() as patch:
    run_tiny_pipeline(patch, host, checkpoint_root=str(root / "shared"),
                      rounds=int(sys.argv[4]), explorers=int(sys.argv[5]))
multihost_utils.sync_global_devices("pipeline_test_complete")
print("DISTRIBUTED_PIPELINE_OK", jax.process_index(), flush=True)
jax.distributed.shutdown()
