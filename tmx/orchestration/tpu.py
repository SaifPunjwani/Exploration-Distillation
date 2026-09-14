from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

from .spec import ExperimentSpec, NetworkSpec


@dataclass(frozen=True)
class TpuStatus:
    project: str
    zone: str
    name: str
    state: str
    health: str
    accelerator_type: str
    external_ips_enabled: bool | None
    worker_count: int
    ready: bool
    raw: dict[str, Any]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw", None)
        return payload


def _gcloud_json(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip())
    return json.loads(proc.stdout or "{}")


def _external_ips_enabled(raw: dict[str, Any]) -> bool | None:
    network_config = raw.get("networkConfig") or {}
    explicit = network_config.get("enableExternalIps")
    if explicit is not None:
        return bool(explicit)

    endpoints = raw.get("networkEndpoints") or []
    if not endpoints:
        return None

    access_configs = [endpoint.get("accessConfig") for endpoint in endpoints]
    if all(value is not None for value in access_configs):
        return True
    if any(value is not None for value in access_configs):
        return None
    return False


def describe_tpu(network: NetworkSpec) -> TpuStatus:
    raw = _gcloud_json(
        [
            "gcloud",
            f"--project={network.project}",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            network.tpu_name,
            f"--zone={network.zone}",
            "--format=json",
        ]
    )
    state = str(raw.get("state") or "UNKNOWN")
    health = str(raw.get("health") or "UNKNOWN")
    accelerator_type = str(raw.get("acceleratorType") or "UNKNOWN")
    endpoints = raw.get("networkEndpoints") or []
    external_ips = _external_ips_enabled(raw)
    ready = (
        state == "READY"
        and health == "HEALTHY"
        and accelerator_type == network.accelerator_type
        and external_ips is not False
    )
    return TpuStatus(
        project=network.project,
        zone=network.zone,
        name=network.tpu_name,
        state=state,
        health=health,
        accelerator_type=accelerator_type,
        external_ips_enabled=external_ips,
        worker_count=len(endpoints),
        ready=ready,
        error="",
        raw=raw,
    )


def describe_queued_resource(network: NetworkSpec, *, original_error: Exception | str = "") -> TpuStatus:
    raw = _gcloud_json(
        [
            "gcloud",
            f"--project={network.project}",
            "alpha",
            "compute",
            "tpus",
            "queued-resources",
            "describe",
            network.tpu_name,
            f"--zone={network.zone}",
            "--format=json",
        ]
    )
    node_specs = ((raw.get("tpu") or {}).get("nodeSpec") or [])
    node = (node_specs[0].get("node") if node_specs else {}) or {}
    state = str((raw.get("state") or {}).get("state") or "UNKNOWN")
    accelerator_type = str(node.get("acceleratorType") or network.accelerator_type)
    external_ips = None
    network_config = node.get("networkConfig") or {}
    if "enableExternalIps" in network_config:
        external_ips = bool(network_config.get("enableExternalIps"))
    return TpuStatus(
        project=network.project,
        zone=network.zone,
        name=network.tpu_name,
        state=f"QUEUED_{state}",
        health="QUEUED",
        accelerator_type=accelerator_type,
        external_ips_enabled=external_ips,
        worker_count=0,
        ready=False,
        error=str(original_error),
        raw=raw,
    )


def unavailable_tpu_status(network: NetworkSpec, error: Exception | str) -> TpuStatus:
    return TpuStatus(
        project=network.project,
        zone=network.zone,
        name=network.tpu_name,
        state="UNAVAILABLE",
        health="UNKNOWN",
        accelerator_type=network.accelerator_type,
        external_ips_enabled=None,
        worker_count=0,
        ready=False,
        error=str(error),
        raw={},
    )


def production_status(spec: ExperimentSpec) -> dict[str, Any]:
    validation = spec.validate()
    try:
        tpu = describe_tpu(spec.network)
    except Exception as exc:
        try:
            tpu = describe_queued_resource(spec.network, original_error=exc)
        except Exception:
            tpu = unavailable_tpu_status(spec.network, exc)
    return {
        "ok": validation.ok and tpu.ready,
        "spec": spec.name,
        "validation": {
            "ok": validation.ok,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        },
        "tpu": tpu.to_dict(),
        "topology": {
            "total_workers": spec.topology.total_workers,
            "chips_per_worker": spec.topology.chips_per_worker,
            "train_workers": list(spec.topology.train_workers),
            "rollout_workers": list(spec.topology.rollout_workers),
            "train_chip_count": spec.topology.train_chip_count,
            "rollout_chip_count": spec.topology.rollout_chip_count,
            "total_chip_count": spec.topology.total_workers * spec.topology.chips_per_worker,
        },
        "storage": {
            "backend": "hf",
            "allow_gcs": spec.storage.allow_gcs,
            "hf_model_repo": spec.storage.hf_model_repo,
            "hf_dataset_repo": spec.storage.hf_dataset_repo,
            "artifact_prefix": spec.storage.artifact_prefix or spec.name,
        },
    }


def wait_until_ready(
    spec: ExperimentSpec,
    *,
    timeout_seconds: int,
    poll_seconds: int,
) -> TpuStatus:
    deadline = time.time() + float(timeout_seconds)
    last: TpuStatus | None = None
    while time.time() < deadline:
        try:
            last = describe_tpu(spec.network)
        except Exception as exc:
            try:
                last = describe_queued_resource(spec.network, original_error=exc)
            except Exception:
                last = unavailable_tpu_status(spec.network, exc)
        if last.ready:
            return last
        print(
            "[tpu-wait] "
            f"{last.project}/{last.zone}/{last.name} "
            f"state={last.state} health={last.health} "
            f"accel={last.accelerator_type} external_ips={last.external_ips_enabled} "
            f"workers={last.worker_count}"
            + (f" error={last.error}" if last.error else ""),
            flush=True,
        )
        time.sleep(float(poll_seconds))
    if last is None:
        last = describe_tpu(spec.network)
    raise TimeoutError(
        f"TPU did not become READY/HEALTHY before timeout: {last.to_dict()}"
    )
