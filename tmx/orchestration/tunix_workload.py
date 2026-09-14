"""Tunix CLI dispatch — alternate container/host entrypoint.

Reads ExperimentSpec from env (same ``TMX_EXPERIMENT_SPEC_JSON_B64``
contract as ``workload.py``), applies XLA/JAX tuning, then dispatches to
``python -m tunix.cli.grpo_main`` with overrides built from the spec.

Why this exists alongside ``workload.py``: that module shells to
``maxtext.trainers.post_train.rl.train_rl`` directly, and MaxText hard-codes
GSM8K rewards at ``train_rl.py:466-470``. The Tunix CLI is the only
upstream entrypoint that supports plugin rewards (``reward_functions=[...]``),
which is how we get TMX correctness + incorrectness_penalty into the run.

Runs as a process on the TPU VM directly (no XPK required). For TRC2 the
host already has external IPs and miniforge envs; we install Tunix into
that env via ``infra/xpk/bootstrap_tunix_env.sh``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys

from .optimization import apply_runtime_env
from .spec import (
    AlgorithmSpec,
    BackendSpec,
    ExperimentSpec,
    LoggingSpec,
    NetworkSpec,
    ResumeSpec,
    StorageSpec,
    TopologySpec,
)
from .tunix_config import emit_command


def _load_spec_from_env() -> ExperimentSpec:
    raw_b64 = os.environ.get("TMX_EXPERIMENT_SPEC_JSON_B64", "").strip()
    raw_json = os.environ.get("TMX_EXPERIMENT_SPEC_JSON", "").strip()
    if raw_b64:
        data = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
    elif raw_json:
        data = json.loads(raw_json)
    else:
        raise RuntimeError(
            "TMX_EXPERIMENT_SPEC_JSON_B64 or TMX_EXPERIMENT_SPEC_JSON is required"
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


def _print_payload(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def describe() -> int:
    spec = _load_spec_from_env()
    result = spec.validate()
    if float(spec.algorithm.lambda_novelty) > 0.0:
        cmd = [sys.executable, "-m", "tmx.orchestration.tunix_main", "run"]
        engine = "tunix_maxtext_programmatic_dapo_drgrpo"
        warnings = [
            *list(result.warnings),
            "Tunix novelty path requires TPU smoke gates for hidden-state feature extraction",
        ]
    else:
        cmd = emit_command(spec)
        engine = "tunix_cli"
        warnings = list(result.warnings)
    _print_payload(
        {
            "status": "ok" if result.ok else "invalid",
            "engine": engine,
            "run_name": spec.name,
            "command": cmd,
            "errors": list(result.errors),
            "warnings": warnings,
        }
    )
    return 0 if result.ok else 2


def run() -> int:
    spec = _load_spec_from_env()
    result = spec.validate()
    if not result.ok:
        _print_payload({"status": "invalid", "errors": list(result.errors)})
        return 2

    if (
        os.environ.get("TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK", "") != "1"
        and not spec.backend.experimental_adapter_ack
    ):
        _print_payload(
            {
                "status": "blocked",
                "reason": (
                    "Tunix CLI dispatch requires "
                    "TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1 after smoke test"
                ),
            }
        )
        return 4

    delta = apply_runtime_env()

    # Dispatch:
    #  - λ > 0 → programmatic Tunix/MaxText DAPO path (novelty learner adapter)
    #  - λ = 0 → Tunix CLI (correctness-only, simpler)
    use_programmatic = float(spec.algorithm.lambda_novelty) > 0.0
    if use_programmatic:
        os.environ.setdefault("TMX_LAMBDA_NOVELTY", str(spec.algorithm.lambda_novelty))
        os.environ.setdefault(
            "TMX_INCORRECT_NOVELTY_SCALE", str(spec.algorithm.incorrect_novelty_scale)
        )
        cmd = [sys.executable, "-m", "tmx.orchestration.tunix_main", "run"]
        engine = "tunix_maxtext_programmatic_dapo_drgrpo"
    else:
        cmd = emit_command(spec)
        engine = "tunix_cli"

    _print_payload(
        {
            "status": "starting",
            "engine": engine,
            "lambda_novelty": spec.algorithm.lambda_novelty,
            "command": cmd,
            "env_delta": delta,
        }
    )

    if os.environ.get("TMX_DRY_RUN_WORKLOAD", "0") == "1":
        return 0
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Tunix CLI dispatch entry point for TMX")
    parser.add_argument("action", nargs="?", default="describe", choices=("describe", "run"))
    args = parser.parse_args()
    return describe() if args.action == "describe" else run()


if __name__ == "__main__":
    raise SystemExit(main())
