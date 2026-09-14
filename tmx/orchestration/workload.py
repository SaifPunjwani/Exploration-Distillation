from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import subprocess
import sys

from .spec import ExperimentSpec


def _load_spec_from_env() -> ExperimentSpec:
    raw_b64 = os.environ.get("TMX_EXPERIMENT_SPEC_JSON_B64", "").strip()
    raw_json = os.environ.get("TMX_EXPERIMENT_SPEC_JSON", "").strip()
    if raw_b64:
        data = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
    elif raw_json:
        data = json.loads(raw_json)
    else:
        raise RuntimeError("TMX_EXPERIMENT_SPEC_JSON_B64 or TMX_EXPERIMENT_SPEC_JSON is required")
    from .spec import (
        AlgorithmSpec,
        BackendSpec,
        LoggingSpec,
        NetworkSpec,
        ResumeSpec,
        StorageSpec,
        TopologySpec,
    )

    return ExperimentSpec(
        name=data["name"],
        network=NetworkSpec(**data.get("network", {})),
        topology=TopologySpec(
            **{
                **data.get("topology", {}),
                "train_workers": tuple(data.get("topology", {}).get("train_workers", ())),
                "rollout_workers": tuple(data.get("topology", {}).get("rollout_workers", ())),
                "train_chips": tuple(data.get("topology", {}).get("train_chips", ())),
                "rollout_chips": tuple(data.get("topology", {}).get("rollout_chips", ())),
            }
        ),
        storage=StorageSpec(**data.get("storage", {})),
        logging=LoggingSpec(**data.get("logging", {})),
        resume=ResumeSpec(**data.get("resume", {})),
        algorithm=AlgorithmSpec(**data.get("algorithm", {})),
        backend=BackendSpec(**data.get("backend", {})),
    )


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"required module {name!r} is not installed in this container")


def _maxtext_command(spec: ExperimentSpec) -> list[str]:
    backend = spec.backend
    entrypoint = os.environ.get("TMX_MAXTEXT_ENTRYPOINT", backend.maxtext_entrypoint)
    model_name = os.environ.get("TMX_MAXTEXT_MODEL_NAME", backend.maxtext_model_name or spec.algorithm.model_name)
    load_path = os.environ.get("TMX_MAXTEXT_LOAD_PARAMETERS_PATH", backend.maxtext_load_parameters_path or spec.resume.init_checkpoint)
    output_dir = os.environ.get("TMX_MAXTEXT_BASE_OUTPUT_DIRECTORY", backend.maxtext_base_output_directory)
    if not load_path:
        raise RuntimeError("TMX_MAXTEXT_LOAD_PARAMETERS_PATH or resume.init_checkpoint is required")

    cmd = [
        sys.executable,
        "-m",
        entrypoint,
        f"model_name={model_name}",
        f"load_parameters_path={load_path}",
        f"run_name={spec.name}",
        f"base_output_directory={output_dir}",
        f"hf_access_token={os.environ.get('HF_TOKEN', '')}",
    ]
    loss_algo = os.environ.get("TMX_MAXTEXT_LOSS_ALGO", backend.maxtext_loss_algo).strip()
    if loss_algo and loss_algo != "grpo":
        cmd.append(f"loss_algo={loss_algo}")
    return cmd


def _print_payload(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def describe() -> int:
    spec = _load_spec_from_env()
    result = spec.validate()
    _print_payload(
        {
            "status": "ok" if result.ok else "invalid",
            "run_name": spec.name,
            "engine": os.environ.get("TMX_BACKEND_ENGINE", spec.backend.engine),
            "rollout_engine": os.environ.get("TMX_ROLLOUT_ENGINE", spec.backend.rollout_engine),
            "artifact_backend": os.environ.get("TMX_ARTIFACT_BACKEND", ""),
            "allow_gcs": spec.storage.allow_gcs,
            "train_chip_count": spec.topology.train_chip_count,
            "rollout_chip_count": spec.topology.rollout_chip_count,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
        }
    )
    return 0 if result.ok else 2


def run() -> int:
    spec = _load_spec_from_env()
    result = spec.validate()
    if not result.ok:
        _print_payload({"status": "invalid", "errors": list(result.errors)})
        return 2
    engine = os.environ.get("TMX_BACKEND_ENGINE", spec.backend.engine)
    if engine != "maxtext_tunix":
        _print_payload({"status": "unsupported", "engine": engine})
        return 3
    if os.environ.get("TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK", "") != "1" and not spec.backend.experimental_adapter_ack:
        _print_payload(
            {
                "status": "blocked",
                "reason": "MaxText/Tunix adapter requires TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1 after container smoke test",
            }
        )
        return 4
    try:
        _require_module("maxtext")
        _require_module("tunix")
    except RuntimeError as exc:
        _print_payload({"status": "missing_dependency", "error": str(exc)})
        return 5
    cmd = _maxtext_command(spec)
    payload = {
        "status": "starting",
        "engine": "maxtext_tunix",
        "command": cmd,
        "reward_plugin": os.environ.get("TMX_REWARD_PLUGIN", spec.backend.reward_plugin),
        "rollout_engine": os.environ.get("TMX_ROLLOUT_ENGINE", spec.backend.rollout_engine),
    }
    _print_payload(payload)
    if os.environ.get("TMX_DRY_RUN_WORKLOAD", "0") == "1":
        return 0
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="TMX production container workload adapter")
    parser.add_argument("action", nargs="?", default="describe", choices=("describe", "run"))
    args = parser.parse_args()
    if args.action == "describe":
        return describe()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
