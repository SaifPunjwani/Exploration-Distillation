from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .backends import build_submission_plan
from .preflight import run_preflight
from .safety import (
    assert_trc_external_ip_policy,
    find_forbidden_network_tokens,
    text_allows_forbidden_network_tokens,
)
from .spec import BackendSpec, ExperimentSpec, ResumeSpec, load_experiment_spec, missing_tpu_coordinates
from .stages import STAGE_PRESETS, stage_preset
from .topologies import NETWORK_PRESETS, TOPOLOGY_PRESETS, network_preset, topology_preset
from .tpu import production_status, wait_until_ready


PRODUCTION_DOCTOR_PATHS = [
    "configs",
    "tmx/orchestration",
    "tmx/artifacts.py",
    "scripts/README.md",
    "scripts/tmx_submit.py",
    "scripts/tmx_paper.py",
    "scripts/build_maxtext_tunix_image.sh",
    "scripts/trc2_v5lite256_guard.sh",
    "scripts/launch_trc2_v5lite256_tmx_jax_resume.sh",
    "scripts/restart_dapo_drgrpo_twoslice.sh",
    "scripts/tpu_target_lib.sh",
    "scripts/bootstrap_vllm_tpu_env.sh",
    "scripts/reload_vllm_with_model.sh",
    "scripts/reload_vllm_slice_with_model.sh",
    "docker/maxtext-tunix/Dockerfile",
]


def _print_plan(plan) -> None:
    print(json.dumps({
        "backend": plan.backend,
        "executable": plan.executable,
        "blockers": list(plan.blockers),
        "warnings": list(plan.warnings),
        "env": plan.env,
        "commands": list(plan.commands),
    }, indent=2, sort_keys=True))


def _require_tpu_coordinates(spec: ExperimentSpec) -> int:
    missing = missing_tpu_coordinates(spec.network)
    if missing:
        print(
            "ERROR: " + ", ".join(missing)
            + " must be set in the spec or via TPU_PROJECT / TPU_ZONE / TPU_NAME before contacting a TPU",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    assert_trc_external_ip_policy(sys.argv[1:])
    spec = load_experiment_spec(args.spec)
    result = spec.validate()
    print(json.dumps({
        "ok": result.ok,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
        "spec": spec.to_dict() if args.show_spec else None,
    }, indent=2, sort_keys=True))
    return 0 if result.ok else 2


def cmd_plan(args: argparse.Namespace) -> int:
    assert_trc_external_ip_policy(sys.argv[1:])
    spec = load_experiment_spec(args.spec)
    result = spec.validate()
    if not result.ok:
        print("spec validation failed:", file=sys.stderr)
        for error in result.errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    plan = build_submission_plan(spec, backend=args.backend)
    _print_plan(plan)
    return 0 if plan.executable or args.allow_blocked else 3


def cmd_submit(args: argparse.Namespace) -> int:
    assert_trc_external_ip_policy(sys.argv[1:])
    spec = load_experiment_spec(args.spec)
    result = spec.validate()
    if not result.ok:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    plan = build_submission_plan(spec, backend=args.backend)
    if not args.quiet_plan:
        _print_plan(plan)
    if not plan.executable:
        return 3
    if args.dry_run:
        return 0
    if _require_tpu_coordinates(spec):
        return 2
    if args.wait_ready:
        try:
            wait_until_ready(
                spec,
                timeout_seconds=int(args.wait_timeout_seconds),
                poll_seconds=int(args.poll_seconds),
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 6
    for command in plan.commands:
        subprocess.run(command, shell=True, check=True)
    return 0


def cmd_list_presets(_args: argparse.Namespace) -> int:
    print(json.dumps({
        "networks": sorted(NETWORK_PRESETS),
        "topologies": sorted(TOPOLOGY_PRESETS),
        "stages": sorted(STAGE_PRESETS),
        "backends": ["tunix-cli-direct", "xpk-gke-maxtext-tunix", "raw-tpu-tmx-jax"],
    }, indent=2, sort_keys=True))
    return 0


def cmd_init_spec(args: argparse.Namespace) -> int:
    assert_trc_external_ip_policy(sys.argv[1:])
    resume = ResumeSpec(
        fresh_run=not bool(args.resume_source_run or args.init_checkpoint or args.existing_trajectories),
        source_run=args.resume_source_run or "",
        source_step=args.resume_source_step,
        init_checkpoint=args.init_checkpoint or "",
        existing_trajectories=args.existing_trajectories or "",
    )
    network = network_preset(args.network, tpu_name=args.tpu_name)
    spec = ExperimentSpec(
        name=args.name,
        network=network,
        topology=topology_preset(args.topology),
        algorithm=stage_preset(args.stage),
        resume=resume,
        backend=BackendSpec(xpk_tpu_type=network.accelerator_type),
    )
    result = spec.validate()
    if not result.ok:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    roots = [Path(p) for p in args.paths]
    scanned = 0
    hits: list[dict[str, object]] = []
    for root in roots:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in files:
            if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".pdf", ".safetensors"}:
                continue
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            scanned += 1
            if text_allows_forbidden_network_tokens(text):
                continue
            found = find_forbidden_network_tokens([text])
            if found:
                hits.append({"path": str(path), "tokens": list(found)})
    payload = {"ok": not hits, "scanned_files": scanned, "hits": hits}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not hits or args.allow_hits else 4


def cmd_status(args: argparse.Namespace) -> int:
    assert_trc_external_ip_policy(sys.argv[1:])
    spec = load_experiment_spec(args.spec)
    if _require_tpu_coordinates(spec):
        return 2
    payload = production_status(spec)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] or args.allow_not_ready else 5


def cmd_wait_ready(args: argparse.Namespace) -> int:
    assert_trc_external_ip_policy(sys.argv[1:])
    spec = load_experiment_spec(args.spec)
    result = spec.validate()
    if not result.ok:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if _require_tpu_coordinates(spec):
        return 2
    try:
        status = wait_until_ready(
            spec,
            timeout_seconds=int(args.timeout_seconds),
            poll_seconds=int(args.poll_seconds),
        )
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 6
    print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    assert_trc_external_ip_policy(sys.argv[1:])
    spec = load_experiment_spec(args.spec)
    if args.require_tpu_ready and _require_tpu_coordinates(spec):
        return 2
    payload = run_preflight(spec, require_tpu_ready=args.require_tpu_ready)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] or args.allow_not_ready else 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TMX production experiment orchestration")
    sub = parser.add_subparsers(required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--spec", required=True)
    validate.add_argument("--show-spec", action="store_true")
    validate.set_defaults(func=cmd_validate)

    plan = sub.add_parser("plan")
    plan.add_argument("--spec", required=True)
    plan.add_argument("--backend", default=None)
    plan.add_argument("--allow-blocked", action="store_true")
    plan.set_defaults(func=cmd_plan)

    submit = sub.add_parser("submit")
    submit.add_argument("--spec", required=True)
    submit.add_argument("--backend", default=None)
    submit.add_argument("--dry-run", action="store_true", default=True)
    submit.add_argument("--execute", dest="dry_run", action="store_false")
    submit.add_argument("--wait-ready", action="store_true")
    submit.add_argument("--wait-timeout-seconds", type=int, default=7200)
    submit.add_argument("--poll-seconds", type=int, default=60)
    submit.add_argument("--quiet-plan", action="store_true")
    submit.set_defaults(func=cmd_submit)

    presets = sub.add_parser("list-presets")
    presets.set_defaults(func=cmd_list_presets)

    init_spec = sub.add_parser("init-spec")
    init_spec.add_argument("--name", required=True)
    init_spec.add_argument("--network", default="trc2-v5litepod-256-uc1a")
    init_spec.add_argument("--topology", default="v5litepod-256-64train-192rollout")
    init_spec.add_argument("--stage", default="explorer_novelty075_16k")
    init_spec.add_argument("--tpu-name", default=None)
    init_spec.add_argument("--resume-source-run", default="")
    init_spec.add_argument("--resume-source-step", default="latest")
    init_spec.add_argument("--init-checkpoint", default="")
    init_spec.add_argument("--existing-trajectories", default="")
    init_spec.add_argument("--output", required=True)
    init_spec.set_defaults(func=cmd_init_spec)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("paths", nargs="*", default=PRODUCTION_DOCTOR_PATHS)
    doctor.add_argument("--allow-hits", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    status = sub.add_parser("status")
    status.add_argument("--spec", required=True)
    status.add_argument("--allow-not-ready", action="store_true")
    status.set_defaults(func=cmd_status)

    wait_ready = sub.add_parser("wait-ready")
    wait_ready.add_argument("--spec", required=True)
    wait_ready.add_argument("--timeout-seconds", type=int, default=7200)
    wait_ready.add_argument("--poll-seconds", type=int, default=60)
    wait_ready.set_defaults(func=cmd_wait_ready)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--spec", required=True)
    preflight.add_argument("--require-tpu-ready", action="store_true")
    preflight.add_argument("--allow-not-ready", action="store_true")
    preflight.set_defaults(func=cmd_preflight)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
