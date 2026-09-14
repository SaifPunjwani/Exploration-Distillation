"""A real two-process CPU run with shared checkpoints and mocked rollout I/O."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("rounds,explorers", [(1, 1), (2, 2)])
def test_distributed_pipeline_and_round_handoffs(tmp_path, rounds, explorers):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{server.getsockname()[1]}"
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        batch_port = str(server.getsockname()[1])
    env = {k: v for k, v in os.environ.items() if k.lower() not in
           {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    env.update(JAX_PLATFORMS="cpu", JAX_CPU_COLLECTIVES_IMPLEMENTATION="gloo",
        XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.update(EXPDIS_BATCH_SHARD_HTTP_PORT=batch_port,
               EXPDIS_BATCH_SHARD_SOURCE_HOST="127.0.0.1")
    script = str(Path(__file__).with_name("multihost_pipeline_worker.py"))
    processes, logs = [], []
    try:
        for rank in range(2):
            log = tmp_path / f"rank_{rank}.log"
            logs.append(log)
            with log.open("w") as handle:
                processes.append(subprocess.Popen([sys.executable, script, address, str(rank),
                    str(tmp_path), str(rounds), str(explorers)], env=env, stdout=handle,
                    stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 180
        for process in processes:
            process.wait(timeout=max(1, deadline - time.monotonic()))
        for process, log in zip(processes, logs):
            output = log.read_text()
            assert process.returncode == 0, output[-12000:]
            assert "DISTRIBUTED_PIPELINE_OK" in output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
