from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tmx_gpu import common, fresh_c8_sft, grpo_gpu, pipeline_gpu
from tmx_gpu.tests.test_sft_checkpoint_selection import fresh_training_evidence


def _runtime(seed: int) -> dict:
    runtime = copy.deepcopy(
        fresh_training_evidence()["scout"]["runtime_contract"])
    runtime["seed"] = seed
    return runtime


def _stable_code_artifact(digest: str = "component-code-digest") -> dict:
    normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": {
            "artifact_ref": "entity/project/code:v29",
            "artifact_digest": digest,
        },
        "files": [{
            "relative_path": "tmx_gpu/pipeline_gpu.py",
            "sha256": "d" * 64,
            "size_bytes": 10,
        }],
    }
    return {
        **normalized,
        "file_manifest_sha256": common.canonical_json_sha256(normalized),
    }


def _component_evidence(seed: int) -> dict:
    return fresh_c8_sft.signed_payload(
        fresh_c8_sft.MULTISCOUT_COMPONENT_EVIDENCE_VERSION,
        {
            "profile": fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
            "component_seed": seed,
            "code_artifact": _stable_code_artifact(),
            "trajectory": {
                "row_count": 12800,
                "steps": 200,
                "rows_per_step": 64,
                "file_sha256": "a" * 64,
                "semantic_digest": {
                    "version": fresh_c8_sft.TRAJECTORY_SEMANTIC_DIGEST_VERSION,
                    "row_count": 12800,
                    "sha256": "b" * 64,
                },
            },
            "scout": {
                "phase": "scout",
                "checkpoint_kind": "deployable_model",
                "step": 200,
                "model": fresh_c8_sft.MODEL_INIT,
                "config_tier": "contract",
                "rl_validity_mode": "c8",
                "lambda_novelty": 0.5,
                "runtime_contract": _runtime(seed),
                "checkpoint_file_manifest": {"sha256": "e" * 64},
            },
        },
    )


def _resign(value: dict) -> dict:
    unsigned = copy.deepcopy(value)
    unsigned.pop("evidence_sha256", None)
    unsigned["evidence_sha256"] = common.canonical_json_sha256(unsigned)
    return unsigned


def _selected_rollout() -> dict:
    return {
        "problem_id": "problem-1",
        "prompt_text": "Solve the problem.",
        "completion_text": "Reasoning. \\boxed{1}",
        "ground_truth": "1",
        "correctness_reward": 1.0,
        "novelty_reward": 0.25,
        "blended_reward": 1.125,
        "is_correct": True,
        "finish_reason": "stop",
        "completion_token_length": 8,
        "clipped": False,
        "terminated": True,
        "valid_answer": True,
    }


def _skip_scout_args(tmp_path: Path, trajectory: Path):
    return pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", str(tmp_path / "pooled-run"),
        "--run-name", "pooled-run",
        "--skip-scout",
        "--trajectories", str(trajectory),
    ])


def _write_complete_component_trajectory(
    path: Path, *, profile_value=fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
    seed_value=1, omit_profile: bool = False, omit_seed: bool = False,
) -> None:
    with path.open("w") as handle:
        for step in range(1, 201):
            for sample in range(64):
                row = {
                    "step": step,
                    "problem_id": f"p-{step}-{sample}",
                    "prompt_text": "problem",
                    "completion_text": "reason \\boxed{1}",
                }
                if not omit_profile:
                    row["fresh_c8_evidence_profile"] = profile_value
                if not omit_seed:
                    row["component_seed"] = seed_value
                handle.write(json.dumps(row) + "\n")


def _write_complete_grpo_trajectory(path: Path, provenance: dict) -> None:
    with path.open("w") as handle:
        for step in range(1, 201):
            for sample in range(64):
                rollout = _selected_rollout()
                rollout["problem_id"] = f"p-{step}-{sample}"
                record = grpo_gpu.trajectory_record(
                    rollout, step, **provenance)
                handle.write(json.dumps(record) + "\n")


def _code_manifest_path(tmp_path: Path, name: str = "code") -> tuple[Path, Path]:
    root = tmp_path / name
    pipeline = root / "tmx_gpu/pipeline_gpu.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("# exact immutable pipeline code\n")
    normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": {
            "artifact_ref": "entity/project/code:v29",
            "artifact_digest": f"{name}-digest",
        },
        "files": [{
            "relative_path": "tmx_gpu/pipeline_gpu.py",
            "sha256": common.file_sha256(str(pipeline)),
            "size_bytes": pipeline.stat().st_size,
        }],
    }
    manifest = {
        **normalized,
        "download_root": str(root.resolve()),
        "file_manifest_sha256": common.canonical_json_sha256(normalized),
    }
    path = tmp_path / f"{name}_manifest.json"
    path.write_text(json.dumps(manifest))
    return path, pipeline


def _component_args(tmp_path: Path, seed: int, code: Path, *extra: str):
    return pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", str(tmp_path / "run"),
        "--run-name", f"component-{seed}",
        "--fresh-c8-multiscout-component",
        "--scout-only",
        "--seed", str(seed),
        "--ckpt-artifact-every", "25",
        "--vllm-max-model-len", "36864",
        "--code-source-manifest", str(code),
        *extra,
    ])


def _canonical_args(tmp_path: Path, *manifest_args: str):
    return pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", str(tmp_path / "canonical-run"),
        "--run-name", "canonical",
        "--fresh-c8-sft-gate",
        "--stop-after-sft",
        "--sft-save-steps", "125,250,500,1000",
        "--vllm-max-model-len", "36864",
        *manifest_args,
    ])


@pytest.mark.parametrize("seed", [1, 2])
def test_component_contract_accepts_only_preregistered_seeds(
    tmp_path, monkeypatch, seed,
):
    code, executing = _code_manifest_path(tmp_path)
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing))
    pipeline_gpu.validate_args(_component_args(tmp_path, seed, code))


@pytest.mark.parametrize("seed", [0, 3])
def test_component_contract_rejects_other_seeds(tmp_path, seed):
    code, _ = _code_manifest_path(tmp_path)
    with pytest.raises(ValueError, match="MultiScout component contract drift"):
        pipeline_gpu.validate_args(_component_args(tmp_path, seed, code))


@pytest.mark.parametrize(
    "mutation",
    ["not_scout_only", "skip_scout", "trajectory", "stop_after_sft",
     "sft_route"],
)
def test_component_profile_is_strictly_scout_only_and_disjoint(
    tmp_path, mutation,
):
    code, _ = _code_manifest_path(tmp_path)
    args = _component_args(tmp_path, 1, code)
    if mutation == "not_scout_only":
        args.scout_only = False
    elif mutation == "skip_scout":
        args.skip_scout = True
        args.trajectories = [str(tmp_path / "trajectory.jsonl")]
    elif mutation == "trajectory":
        args.trajectories = [str(tmp_path / "trajectory.jsonl")]
    elif mutation == "stop_after_sft":
        args.scout_only = False
        args.stop_after_sft = True
    else:
        args.fresh_c8_sft_gate = True
    with pytest.raises(ValueError):
        pipeline_gpu.validate_args(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scientific_variant", "posthoc-variant"),
        ("save_every", 50),
        ("keep_checkpoints", 2),
        ("ckpt_artifact_every", 100),
        ("probe_every", 0),
        ("vllm_max_model_len", 18432),
        ("max_consecutive_gen_failures", 4),
        ("gen_concurrency", 4),
    ],
)
def test_component_rejects_instrumentation_or_variant_drift(
    tmp_path, field, value,
):
    code, _ = _code_manifest_path(tmp_path)
    args = _component_args(tmp_path, 1, code)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="MultiScout component contract drift"):
        pipeline_gpu.validate_args(args)


def test_canonical_seed_zero_contract_remains_disjoint(tmp_path):
    code, _ = _code_manifest_path(tmp_path)
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", str(tmp_path / "run"), "--run-name", "canonical",
        "--fresh-c8-sft-gate", "--stop-after-sft",
        "--sft-save-steps", "125,250,500,1000",
        "--seed", "1", "--code-source-manifest", str(code),
    ])
    with pytest.raises(ValueError, match="fresh-c8 gated SFT contract drift"):
        pipeline_gpu.validate_args(args)


def test_canonical_rejects_valid_manifest_for_nonexecuting_root(
    tmp_path, monkeypatch,
):
    good, executing = _code_manifest_path(tmp_path, "executing-code")
    unrelated, _ = _code_manifest_path(tmp_path, "unrelated-code")
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing))
    with pytest.raises(ValueError, match="actual executing immutable code root"):
        pipeline_gpu.validate_args(_canonical_args(
            tmp_path, "--code-source-manifest", str(unrelated)))
    pipeline_gpu.validate_args(_canonical_args(
        tmp_path, "--code-source-manifest", str(good)))


def test_canonical_trusted_manifest_wins_only_when_appended_last(
    tmp_path, monkeypatch,
):
    good, executing = _code_manifest_path(tmp_path, "executing-code")
    unrelated, _ = _code_manifest_path(tmp_path, "unrelated-code")
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing))
    protected = _canonical_args(
        tmp_path,
        "--code-source-manifest", str(unrelated),
        "--code-source-manifest", str(good),
    )
    assert protected.code_source_manifest == str(good)
    pipeline_gpu.validate_args(protected)
    substituted = _canonical_args(
        tmp_path,
        "--code-source-manifest", str(good),
        "--code-source-manifest", str(unrelated),
    )
    with pytest.raises(ValueError, match="actual executing immutable code root"):
        pipeline_gpu.validate_args(substituted)


@pytest.mark.parametrize("seed", [1, 2])
def test_component_completion_evidence_accepts_exact_runtime(seed):
    evidence = _component_evidence(seed)
    validated = \
        fresh_c8_sft.validate_multiscout_component_completion_evidence(evidence)
    assert validated["component_seed"] == seed
    with pytest.raises(ValueError, match="version mismatch"):
        fresh_c8_sft.validate_scout_completion_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("seed", 2, "runtime seed"),
        ("optimizer", "adafactor", "runtime optimizer"),
        ("adam_beta1", 0.8, "runtime adam_beta1"),
        ("adam_beta2", 0.9, "runtime adam_beta2"),
        ("adam_eps", 1e-6, "runtime adam_eps"),
        ("weight_decay", 0.1, "runtime weight_decay"),
        ("grad_clip", 0.5, "runtime grad_clip"),
        ("clip_epsilon_low", 0.1, "runtime clip_epsilon_low"),
        ("clip_epsilon_high", 0.3, "runtime clip_epsilon_high"),
        ("parameter_dtype", "bfloat16", "runtime parameter_dtype"),
        ("expected_steps", 199, "runtime expected_steps"),
        ("resolved_max_prompt_len", 4096,
         "runtime resolved_max_prompt_len"),
        ("resolved_max_completion_len", 32768,
         "runtime resolved_max_completion_len"),
        ("resolved_vllm_max_model_len", 18432,
         "runtime resolved_vllm_max_model_len"),
        ("soft_overlong_expected_len", 12000,
         "runtime soft_overlong_expected_len"),
        ("soft_overlong_cache_len", 4000,
         "runtime soft_overlong_cache_len"),
        ("loss_normalization", "token_mean", "runtime loss_normalization"),
        ("advantage_standardization", True,
         "runtime advantage_standardization"),
        ("kl_beta", 0.1, "runtime kl_beta"),
        ("training_prompt", {"prompt_protocol": "old_system_message"},
         "runtime training_prompt"),
        ("max_train_examples", 19999, "runtime max_train_examples"),
        ("max_consecutive_generation_failures", 4,
         "runtime max_consecutive_generation_failures"),
        ("generation_timeout_seconds", 60.0,
         "runtime generation_timeout_seconds"),
        ("generation_concurrency_per_server", 4,
         "runtime generation_concurrency_per_server"),
        ("chunk_tokens", 128, "runtime chunk_tokens"),
    ],
)
def test_component_completion_evidence_rejects_runtime_drift(
    field, value, match,
):
    evidence = _component_evidence(1)
    evidence["scout"]["runtime_contract"][field] = value
    with pytest.raises(ValueError, match=match):
        fresh_c8_sft.validate_multiscout_component_completion_evidence(
            _resign(evidence))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_component_completion_evidence_requires_exact_runtime_key_set(mutation):
    evidence = _component_evidence(1)
    runtime = evidence["scout"]["runtime_contract"]
    if mutation == "missing":
        runtime.pop("adam_beta1")
    else:
        runtime["unregistered_scientific_field"] = 1
    with pytest.raises(ValueError, match="runtime key set mismatch"):
        fresh_c8_sft.validate_multiscout_component_completion_evidence(
            _resign(evidence))


def test_component_completion_evidence_rejects_unsigned_or_invalid_seed():
    evidence = _component_evidence(1)
    evidence["component_seed"] = 2
    with pytest.raises(ValueError, match="SHA256 is missing or invalid"):
        fresh_c8_sft.validate_multiscout_component_completion_evidence(evidence)
    evidence = _component_evidence(1)
    evidence["component_seed"] = 0
    with pytest.raises(ValueError, match="seed must be exactly 1 or 2"):
        fresh_c8_sft.validate_multiscout_component_completion_evidence(
            _resign(evidence))


@pytest.mark.parametrize("invalid_seed", [True, 1.0])
def test_component_completion_evidence_rejects_noninteger_seed_type(
    invalid_seed,
):
    evidence = _component_evidence(1)
    evidence["component_seed"] = invalid_seed
    with pytest.raises(ValueError, match="seed must be exactly 1 or 2"):
        fresh_c8_sft.validate_multiscout_component_completion_evidence(
            _resign(evidence))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("seed", True), ("seed", 1.0), ("round", True), ("round", 1.0)],
)
def test_component_runtime_rejects_bool_or_float_integer_fields(
    field, invalid_value,
):
    evidence = _component_evidence(1)
    evidence["scout"]["runtime_contract"][field] = invalid_value
    with pytest.raises(ValueError, match=f"runtime {field}"):
        fresh_c8_sft.validate_multiscout_component_completion_evidence(
            _resign(evidence))


@pytest.mark.parametrize("invalid_seed", [True, 1.0])
def test_component_runtime_helper_rejects_noninteger_expected_seed(invalid_seed):
    with pytest.raises(ValueError, match="seed must be exactly 1 or 2"):
        fresh_c8_sft.validate_multiscout_component_runtime_contract(
            _runtime(1), expected_seed=invalid_seed)


def test_component_rejects_manifest_for_nonexecuting_code_root(tmp_path):
    code, _ = _code_manifest_path(tmp_path, "unrelated-code")
    with pytest.raises(ValueError, match="actual executing immutable code root"):
        pipeline_gpu.validate_args(_component_args(tmp_path, 1, code))


def test_component_prepare_rejects_substituted_grpo_manifest(
    tmp_path, monkeypatch,
):
    code, executing = _code_manifest_path(tmp_path, "executing-code")
    other, _ = _code_manifest_path(tmp_path, "other-code")
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing))
    args = _component_args(tmp_path, 1, code)
    pipeline_gpu.validate_args(args)
    gargs = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model,
        output_dir=str(tmp_path / "run/round_01/scout"), lam=0.5,
        steps=200, round_idx=1, global_step_offset=0,
    )
    gargs.code_source_manifest = str(other)
    with pytest.raises(RuntimeError, match="lacks the current code manifest"):
        pipeline_gpu._prepare_fresh_c8_scout_code_binding(args, gargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adam_beta1", 0.8),
        ("adam_beta2", 0.9),
        ("adam_eps", 1e-6),
        ("weight_decay", 0.1),
        ("grad_clip", 0.5),
        ("clip_epsilon_low", 0.1),
        ("clip_epsilon_high", 0.3),
    ],
)
def test_component_runtime_drift_fails_preupdate(
    tmp_path, monkeypatch, field, value,
):
    code, executing = _code_manifest_path(tmp_path)
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing))
    args = _component_args(tmp_path, 1, code)
    pipeline_gpu.validate_args(args)
    gargs = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model,
        output_dir=str(tmp_path / "run/round_01/scout"), lam=0.5,
        steps=200, round_idx=1, global_step_offset=0,
    )
    setattr(gargs, field, value)
    with pytest.raises(ValueError, match=f"runtime {field}"):
        pipeline_gpu._prepare_fresh_c8_scout_code_binding(args, gargs)


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--adam-beta1", "0.8"),
        ("--adam-beta2", "0.9"),
        ("--adam-eps", "1e-6"),
        ("--weight-decay", "0.1"),
        ("--grad-clip", "0.5"),
        ("--clip-epsilon-low", "0.1"),
        ("--clip-epsilon-high", "0.3"),
    ],
)
def test_component_extra_args_cannot_expose_grpo_optimizer_drift(
    tmp_path, flag, value,
):
    code, _ = _code_manifest_path(tmp_path)
    with pytest.raises(SystemExit):
        _component_args(tmp_path, 1, code, flag, value)


def test_component_resume_rejects_cross_profile_and_cross_seed(
    tmp_path, monkeypatch,
):
    code_path, executing = _code_manifest_path(tmp_path)
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing))
    code_manifest = json.loads(code_path.read_text())
    code_binding = common.stable_artifact_download_identity(
        common.validate_artifact_download_manifest(code_manifest))
    scout_final = tmp_path / "run/round_01/scout/final"
    scout_final.mkdir(parents=True)
    (scout_final / "config.json").write_text("{}")
    (scout_final / "model.safetensors").write_bytes(b"scout")
    model_manifest = common.directory_file_manifest(
        str(scout_final), exclude_relative_paths=("trainer_state.json",))
    common.atomic_write_json(scout_final / "trainer_state.json", {
        "phase": "scout",
        "checkpoint_kind": "deployable_model",
        "step": 200,
        "model": fresh_c8_sft.MODEL_INIT,
        "config_tier": "contract",
        "rl_validity_mode": "c8",
        "lambda": 0.5,
        "code_artifact_binding": code_binding,
        "runtime_contract": _runtime(1),
        "checkpoint_file_manifest": model_manifest,
    })
    trajectory = scout_final.parent / "explorer_trajectories.jsonl"
    with trajectory.open("w") as handle:
        for step in range(1, 201):
            for sample in range(64):
                handle.write(json.dumps({
                    "step": step,
                    "problem_id": f"p-{step}-{sample}",
                    "prompt_text": "problem",
                    "completion_text": "reason \\boxed{1}",
                    "fresh_c8_evidence_profile":
                        fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
                    "component_seed": 1,
                }) + "\n")
    args = _component_args(tmp_path, 1, code_path)
    evidence, evidence_path = pipeline_gpu._record_fresh_c8_scout_completion(
        args, scout_final=str(scout_final), trajectory_path=str(trajectory),
        profile=fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE)
    stage = {
        "final": str(scout_final),
        "traj_paths": [str(trajectory)],
        "scout_completion_evidence_path": evidence_path,
        "scout_completion_evidence_sha256": evidence["evidence_sha256"],
    }
    pipeline_gpu._validate_persisted_fresh_c8_scout_completion(
        args, stage, profile=fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE)
    with pytest.raises(ValueError, match="version mismatch"):
        pipeline_gpu._validate_persisted_fresh_c8_scout_completion(
            args, stage, profile=fresh_c8_sft.PROFILE)
    args.seed = 2
    with pytest.raises(RuntimeError, match="evidence seed differs"):
        pipeline_gpu._validate_persisted_fresh_c8_scout_completion(
            args, stage, profile=fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE)


def test_launcher_keeps_component_manifest_after_extra_args_and_has_no_pool():
    path = Path(__file__).resolve().parents[1] / "launch/expdis_job.sh"
    launcher = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert "FRESH_C8_MULTISCOUT_COMPONENT" in launcher
    assert "--fresh-c8-multiscout-component" in launcher
    assert "--code-source-manifest \"$CODE_SOURCE_MANIFEST\"" in launcher
    invocation = launcher[launcher.index("python3 \"$CODE_ROOT/tmx_gpu/pipeline_gpu.py\""):]
    assert invocation.index('"${EXTRA_ARGV[@]}"') < invocation.index(
        '"${FRESH_C8_SFT_GATE_ARGS[@]}"')
    assert invocation.index('"${EXTRA_ARGV[@]}"') < invocation.index(
        '"${FRESH_C8_MULTISCOUT_COMPONENT_ARGS[@]}"')
    assert "FRESH_C8_MULTISCOUT_POOL" not in launcher
    assert "wandb_multiscout_trajectory_sources_v2" not in launcher
    assert "reject_multiscout_component_from_generic_pool" in launcher
    assert '"component_seed": int(os.environ["SEED"])' in launcher


@pytest.mark.parametrize(
    "abbreviation", ["--fresh-c8-m", "--fresh-c8-multiscout"],
)
def test_pipeline_parser_rejects_component_profile_abbreviations(
    tmp_path, abbreviation,
):
    with pytest.raises(SystemExit):
        pipeline_gpu.build_arg_parser().parse_args([
            "--output-dir", str(tmp_path / "run"),
            "--run-name", "abbreviation-attack",
            abbreviation,
        ])


@pytest.mark.parametrize(
    ("extra_args", "expected_status"),
    [
        ("--fresh-c8-multiscout-component", 1),
        ("--fresh-c8-sft-gate", 1),
        ("--probe-every 25", 1),
        ("--vllm-port-base 9000", 0),
    ],
)
def test_launcher_executes_protected_extra_args_guard(
    extra_args, expected_status,
):
    path = Path(__file__).resolve().parents[1] / "launch/expdis_job.sh"
    launcher = path.read_text()
    start = launcher.index("reject_protected_extra_args() {")
    call = launcher.index("\nparse_extra_args\n", start)
    function_source = launcher[start:call]
    tokens = extra_args.split()
    script = "\n".join([
        "set -euo pipefail",
        function_source,
        'reject_protected_extra_args "$@"',
    ])
    result = subprocess.run(
        ["bash", "-c", script, "guard-test", *tokens],
        env=os.environ,
        check=False,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (expected_status == 0)


@pytest.mark.parametrize(
    "flag",
    [
        "--rounds", "--seed", "--model", "--run-name",
        "--scouts-per-round", "--code-source-manifest",
        "--expdis-stabilization-mode", "--novelty-normalization",
        "--c9-execution-profile", "--stabilization-cell",
        "--scientific-variant", "--policy", "--scout-steps",
        "--central-steps", "--prompts-per-step", "--num-generations",
        "--explorers-per-round", "--explorer-steps", "--main-steps",
        "--explorer-lr", "--main-lr", "--explorer-only", "--skip-explorer",
        "--dataset", "--max-train-examples", "--probe-every",
        "--sft-grad-accum", "--completion-budget",
    ],
)
@pytest.mark.parametrize("spelling", ["separated", "equals"])
def test_launcher_rejects_every_provenance_override_spelling(flag, spelling):
    path = Path(__file__).resolve().parents[1] / "launch/expdis_job.sh"
    launcher = path.read_text()
    start = launcher.index("reject_protected_extra_args() {")
    call = launcher.index("\nparse_extra_args\n", start)
    function_source = launcher[start:call]
    argv = [flag, "attacker-value"] if spelling == "separated" else [
        f"{flag}=attacker-value"]
    result = subprocess.run(
        ["bash", "-c", "\n".join([
            "set -euo pipefail",
            function_source,
            'reject_protected_extra_args "$@"',
        ]), "guard-test", *argv],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert f"protected option {flag}" in result.stdout


def test_launcher_guard_covers_all_nonoperational_pipeline_options():
    path = Path(__file__).resolve().parents[1] / "launch/expdis_job.sh"
    launcher = path.read_text()
    start = launcher.index("reject_protected_extra_args() {")
    call = launcher.index("\nparse_extra_args\n", start)
    guard_source = launcher[start:call]
    operational_passthrough = {
        "--help", "--vllm-port-base", "--vllm-gpu-mem-util",
        "--vllm-max-num-seqs", "--gen-concurrency",
    }
    parser_flags = {
        option
        for action in pipeline_gpu.build_arg_parser()._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    missing = sorted(parser_flags - operational_passthrough - {
        flag for flag in parser_flags if flag in guard_source})
    assert missing == []


def test_launcher_shell_lexes_extra_args_without_eval(tmp_path):
    path = Path(__file__).resolve().parents[1] / "launch/expdis_job.sh"
    launcher = path.read_text()
    start = launcher.index("EXTRA_ARGV=()")
    end = launcher.index("\n# Every option already supplied", start)
    parser_source = launcher[start:end]
    sentinel = tmp_path / "must-not-exist"
    raw = f'--vllm-port-base 9000 "$(touch {sentinel})"'
    result = subprocess.run(
        ["bash", "-c", "\n".join([
            "set -euo pipefail",
            parser_source,
            "parse_extra_args",
            "printf '<%s>\\n' \"${EXTRA_ARGV[@]}\"",
        ])],
        env={**os.environ, "EXTRA_ARGS": raw},
        check=True,
        capture_output=True,
        text=True,
    )
    assert not sentinel.exists()
    assert result.stdout.splitlines() == [
        "<--vllm-port-base>", "<9000>", f"<$(touch {sentinel})>"]


@pytest.mark.parametrize(
    ("component_env", "single_env", "seed", "expected"),
    [
        ("1", "0", "2", {
            "fresh_c8_evidence_profile":
                fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
            "component_seed": 2,
        }),
        ("0", "1", "0", {
            "fresh_c8_evidence_profile": fresh_c8_sft.PROFILE,
            "seed": 0,
        }),
    ],
)
def test_joblogs_artifact_executes_profile_metadata_logic(
    monkeypatch, component_env, single_env, seed, expected,
):
    path = Path(__file__).resolve().parents[1] / "launch/expdis_job.sh"
    launcher = path.read_text()
    marker = 'python3 - "$status" <<\'PY\' || true\n'
    start = launcher.index(marker) + len(marker)
    end = launcher.index("\nPY\n}", start)
    body = launcher[start:end]
    captured = {}

    class FakeArtifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata

        def add_file(self, _path, name):
            captured.setdefault("files", []).append(name)

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def log_artifact(self, artifact):
            captured["artifact"] = artifact

        def finish(self):
            return None

    fake_wandb = SimpleNamespace(
        Artifact=FakeArtifact, init=lambda **_kwargs: FakeRun())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("RUN_NAME", "metadata-test")
    monkeypatch.setenv("FRESH_C8_MULTISCOUT_COMPONENT", component_env)
    monkeypatch.setenv("FRESH_C8_SFT_GATE", single_env)
    monkeypatch.setenv("SEED", seed)
    monkeypatch.setattr(sys, "argv", ["joblogs", "0"])
    exec(compile(body, "expdis_job.sh:upload_logs", "exec"), {})
    assert captured["artifact"].type == "logs"
    assert captured["artifact"].metadata == expected


def test_generic_pool_rejects_component_evidence_or_partial_metadata():
    with pytest.raises(ValueError, match="strict pooled consumer"):
        fresh_c8_sft.reject_multiscout_component_from_generic_pool(
            artifact_metadata={
                "fresh_c8_evidence_profile":
                    fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
                "component_seed": 1,
            },
            completion_evidence=[],
        )
    with pytest.raises(ValueError, match="generic trajectory pooling"):
        fresh_c8_sft.reject_multiscout_component_from_generic_pool(
            artifact_metadata={},
            completion_evidence=[_component_evidence(1)],
        )
    fresh_c8_sft.reject_multiscout_component_from_generic_pool(
        artifact_metadata={},
        completion_evidence=[{
            "version": fresh_c8_sft.SCOUT_COMPLETION_EVIDENCE_VERSION,
            "profile": fresh_c8_sft.PROFILE,
        }],
    )


def test_generic_pool_rejects_reserved_component_seed_without_profile():
    with pytest.raises(ValueError, match="reserved component_seed metadata"):
        fresh_c8_sft.reject_multiscout_component_from_generic_pool(
            artifact_metadata={"component_seed": 1},
            completion_evidence=[],
        )
    with pytest.raises(ValueError, match="generic trajectory pooling"):
        fresh_c8_sft.reject_multiscout_component_from_generic_pool(
            artifact_metadata={},
            completion_evidence=[{"component_seed": 1}],
        )
    # ``seed`` is a long-standing generic/canonical field.  It is not the
    # component profile's reserved identity key and remains backward
    # compatible, including alongside the canonical fresh-c8 profile.
    fresh_c8_sft.reject_multiscout_component_from_generic_pool(
        artifact_metadata={"seed": 1}, completion_evidence=[])
    fresh_c8_sft.reject_multiscout_component_from_generic_pool(
        artifact_metadata={
            "fresh_c8_evidence_profile": fresh_c8_sft.PROFILE,
            "seed": 0,
        },
        completion_evidence=[],
    )


def test_official_component_trajectory_rows_carry_copy_stable_provenance():
    record = grpo_gpu.trajectory_record(
        _selected_rollout(), 1,
        fresh_c8_evidence_profile=
            fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
        component_seed=2,
    )
    assert record["fresh_c8_evidence_profile"] == \
        fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE
    assert record["component_seed"] == 2
    canonical = grpo_gpu.trajectory_record(_selected_rollout(), 1)
    assert "fresh_c8_evidence_profile" not in canonical
    assert "component_seed" not in canonical


def test_actual_canonical_grpo_path_does_not_reserve_component_seed(
    tmp_path, monkeypatch,
):
    code, executing = _code_manifest_path(tmp_path)
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing))
    args = _canonical_args(
        tmp_path, "--code-source-manifest", str(code))
    pipeline_gpu.validate_args(args)
    gargs = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model,
        output_dir=str(tmp_path / "canonical-scout"), lam=0.5,
        steps=200, round_idx=1, global_step_offset=0,
    )
    provenance = grpo_gpu._fresh_c8_component_artifact_metadata(gargs)
    assert provenance == {
        "fresh_c8_evidence_profile": fresh_c8_sft.PROFILE,
        "seed": 0,
    }
    row = grpo_gpu.trajectory_record(_selected_rollout(), 1, **provenance)
    assert "component_seed" not in row
    assert row["seed"] == 0
    fresh_c8_sft.reject_multiscout_component_from_generic_pool(
        artifact_metadata=row, completion_evidence=[])


def test_actual_grpo_provenance_rows_pass_matching_completion_facts(
    tmp_path, monkeypatch,
):
    code, executing = _code_manifest_path(tmp_path)
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing))

    canonical_args = _canonical_args(
        tmp_path, "--code-source-manifest", str(code))
    pipeline_gpu.validate_args(canonical_args)
    canonical_gargs = pipeline_gpu._grpo_args(
        canonical_args, phase="scout", model=canonical_args.model,
        output_dir=str(tmp_path / "canonical-scout"), lam=0.5,
        steps=200, round_idx=1, global_step_offset=0,
    )
    canonical_provenance = \
        grpo_gpu._fresh_c8_component_artifact_metadata(canonical_gargs)
    canonical_path = tmp_path / "canonical.jsonl"
    _write_complete_grpo_trajectory(canonical_path, canonical_provenance)
    canonical_facts = pipeline_gpu._fresh_c8_trajectory_completion_facts(
        str(canonical_path), profile=fresh_c8_sft.PROFILE)
    assert canonical_facts["row_count"] == 12800

    component_args = _component_args(tmp_path, 1, code)
    pipeline_gpu.validate_args(component_args)
    component_gargs = pipeline_gpu._grpo_args(
        component_args, phase="scout", model=component_args.model,
        output_dir=str(tmp_path / "component-scout"), lam=0.5,
        steps=200, round_idx=1, global_step_offset=0,
    )
    component_provenance = \
        grpo_gpu._fresh_c8_component_artifact_metadata(component_gargs)
    component_path = tmp_path / "component.jsonl"
    _write_complete_grpo_trajectory(component_path, component_provenance)
    component_facts = pipeline_gpu._fresh_c8_trajectory_completion_facts(
        str(component_path),
        profile=fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
        component_seed=1,
    )
    assert component_facts["row_count"] == 12800


@pytest.mark.parametrize(
    "case",
    ["missing_profile", "missing_seed", "wrong_profile", "wrong_seed",
     "bool_seed", "float_seed"],
)
def test_component_completion_facts_require_every_exact_row_marker(
    tmp_path, case,
):
    trajectory = tmp_path / f"{case}.jsonl"
    kwargs = {}
    if case == "missing_profile":
        kwargs["omit_profile"] = True
    elif case == "missing_seed":
        kwargs["omit_seed"] = True
    elif case == "wrong_profile":
        kwargs["profile_value"] = fresh_c8_sft.PROFILE
    elif case == "wrong_seed":
        kwargs["seed_value"] = 2
    elif case == "bool_seed":
        kwargs["seed_value"] = True
    else:
        kwargs["seed_value"] = 1.0
    _write_complete_component_trajectory(trajectory, **kwargs)
    with pytest.raises(ValueError, match="row provenance mismatch"):
        pipeline_gpu._fresh_c8_trajectory_completion_facts(
            str(trajectory),
            profile=fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
            component_seed=1,
        )


def test_component_completion_facts_accept_exact_row_markers(tmp_path):
    trajectory = tmp_path / "valid-component.jsonl"
    _write_complete_component_trajectory(trajectory)
    facts = pipeline_gpu._fresh_c8_trajectory_completion_facts(
        str(trajectory),
        profile=fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
        component_seed=1,
    )
    assert facts["row_count"] == 12800
    assert facts["steps"] == 200


def test_direct_local_skip_scout_rejects_component_completion_sibling(
    tmp_path,
):
    trajectory = tmp_path / "scout" / "explorer_trajectories.jsonl"
    trajectory.parent.mkdir()
    trajectory.write_text(json.dumps(
        grpo_gpu.trajectory_record(_selected_rollout(), 1)) + "\n")
    (trajectory.parent / "fresh_c8_scout_completion_evidence.json").write_text(
        json.dumps(_component_evidence(1)))
    with pytest.raises(ValueError, match="generic trajectory pooling"):
        pipeline_gpu.validate_args(_skip_scout_args(tmp_path, trajectory))


def test_copied_official_component_jsonl_rejects_without_sibling(tmp_path):
    official = grpo_gpu.trajectory_record(
        _selected_rollout(), 1,
        fresh_c8_evidence_profile=
            fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
        component_seed=1,
    )
    copied = tmp_path / "copied-explorer-trajectories.jsonl"
    copied.write_text(json.dumps(official) + "\n")
    with pytest.raises(ValueError, match="strict pooled consumer"):
        pipeline_gpu.validate_args(_skip_scout_args(tmp_path, copied))


def test_copied_partial_component_seed_row_rejects_without_profile(tmp_path):
    row = grpo_gpu.trajectory_record(_selected_rollout(), 1)
    row["component_seed"] = 1
    copied = tmp_path / "partial-component-row.jsonl"
    copied.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="reserved component_seed metadata"):
        pipeline_gpu.validate_args(_skip_scout_args(tmp_path, copied))


def test_partial_component_seed_sibling_rejects_without_profile(tmp_path):
    trajectory = tmp_path / "partial-sibling" / "explorer_trajectories.jsonl"
    trajectory.parent.mkdir()
    trajectory.write_text(json.dumps(
        grpo_gpu.trajectory_record(_selected_rollout(), 1)) + "\n")
    (trajectory.parent / "fresh_c8_scout_completion_evidence.json").write_text(
        json.dumps({"component_seed": 2}))
    with pytest.raises(ValueError, match="generic trajectory pooling"):
        pipeline_gpu.validate_args(_skip_scout_args(tmp_path, trajectory))


@pytest.mark.parametrize("with_canonical_evidence", [False, True])
def test_direct_local_skip_scout_allows_legacy_and_canonical_inputs(
    tmp_path, with_canonical_evidence,
):
    trajectory = tmp_path / "canonical" / "explorer_trajectories.jsonl"
    trajectory.parent.mkdir()
    trajectory.write_text(json.dumps(
        grpo_gpu.trajectory_record(_selected_rollout(), 1)) + "\n")
    if with_canonical_evidence:
        evidence = {
            "version": fresh_c8_sft.SCOUT_COMPLETION_EVIDENCE_VERSION,
            "profile": fresh_c8_sft.PROFILE,
        }
        (trajectory.parent /
         "fresh_c8_scout_completion_evidence.json").write_text(
            json.dumps(evidence))
    pipeline_gpu.validate_args(_skip_scout_args(tmp_path, trajectory))


def test_pipeline_results_artifact_uses_component_seed_metadata(
    tmp_path, monkeypatch,
):
    output_dir = tmp_path / "pipeline-output"
    output_dir.mkdir()
    (output_dir / "metrics.jsonl").write_text("{}\n")
    captured = {}

    class FakeArtifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata

        def add_file(self, _path, name):
            captured.setdefault("files", []).append(name)

    class FakeRun:
        def log_artifact(self, artifact):
            captured["artifact"] = artifact
            return SimpleNamespace(wait=lambda: captured.setdefault("waited", True))

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Artifact=FakeArtifact))
    args = SimpleNamespace(
        run_name="component-2",
        output_dir=str(output_dir),
        seed=2,
        fresh_c8_sft_gate=False,
        fresh_c8_multiscout_component=True,
    )
    pipeline_gpu._push_results_artifact(FakeRun(), args, "scout")
    metadata = captured["artifact"].metadata
    assert metadata["fresh_c8_evidence_profile"] == \
        fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE
    assert metadata["component_seed"] == 2
    assert "seed" not in metadata
    assert captured["waited"] is True


def test_component_checkpoint_artifact_uses_component_seed_metadata(
    tmp_path, monkeypatch,
):
    output_dir = tmp_path / "checkpoint-output"
    args = grpo_gpu.build_arg_parser().parse_args([
        "--output-dir", str(output_dir),
        "--run-name", "component-1",
        "--phase", "scout",
        "--rl-validity-mode", "c8",
        "--ckpt-artifact-every", "1",
        "--keep-checkpoints", "1",
        "--seed", "1",
    ])
    args.fresh_c8_evidence_profile = \
        fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE
    args._policy_attention_implementation = "sdpa"
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())
    captured = {}

    def fake_save(_model, _tokenizer, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "config.json").write_text("{}")

    class Logged:
        def wait(self):
            return None

    def fake_log_dir_artifact(
        _run, *, name, path, type_, metadata,
    ):
        captured.update({
            "name": name, "path": path, "type": type_,
            "metadata": metadata,
        })
        return Logged()

    monkeypatch.setattr(grpo_gpu, "save_model_training_dtype", fake_save)
    monkeypatch.setattr(common, "log_dir_artifact", fake_log_dir_artifact)
    monkeypatch.setattr(
        grpo_gpu, "_log_checkpoint_progress_artifact",
        lambda *_args, **_kwargs: None,
    )
    grpo_gpu._save_checkpoint(
        None, None, None, optimizer, args, 1, 0.5,
        wandb_run=SimpleNamespace())
    assert captured["type"] == "checkpoint"
    assert captured["metadata"]["fresh_c8_evidence_profile"] == \
        fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE
    assert captured["metadata"]["component_seed"] == 1


def test_shell_results_and_final_model_reuse_component_metadata():
    path = Path(__file__).resolve().parents[1] / "launch/expdis_job.sh"
    launcher = path.read_text()
    assert 'type="expdis-results", metadata=artifact_metadata)' in launcher
    # The publication model extends the component/campaign metadata with the
    # immutable byte-manifest digest instead of passing the base dictionary
    # directly.  Assert the data flow rather than the old source spelling.
    assert "final_model_metadata = {" in launcher
    assert "**artifact_metadata" in launcher
    assert 'type="model", metadata=final_model_metadata)' in launcher
    assert '"model_file_manifest_sha256": final_model_manifest["sha256"]' in \
        launcher
    assert 'type="logs", metadata=artifact_metadata)' in launcher
    assert launcher.count("metadata=artifact_metadata") >= 2


def test_grpo_final_artifact_and_pipeline_runtime_recheck_are_wired():
    grpo_source = Path(grpo_gpu.__file__).read_text()
    final_block = grpo_source[
        grpo_source.index("logged_final = common.log_dir_artifact"):
        grpo_source.index("# Leave a caller-provided pool", grpo_source.index(
            "logged_final = common.log_dir_artifact"))
    ]
    # Metadata is assembled immediately before upload so the identical object
    # can also be embedded in the signed artifact-binding sidecar.
    metadata_block = grpo_source[
        grpo_source.rindex(
            "final_artifact_metadata = {", 0,
            grpo_source.index("logged_final = common.log_dir_artifact")):
        grpo_source.index("# Leave a caller-provided pool", grpo_source.index(
            "logged_final = common.log_dir_artifact"))
    ]
    assert "**_fresh_c8_component_artifact_metadata(args)" in metadata_block
    assert "metadata=final_artifact_metadata" in final_block
    assert '"artifact_metadata": final_artifact_metadata' in final_block

    pipeline_source = Path(pipeline_gpu.__file__).read_text()
    # Definition + validate_args preflight + run_round defense-in-depth check.
    assert pipeline_source.count(
        "_reject_multiscout_component_local_trajectory_inputs(") == 3


def test_component_progress_artifact_carries_executable_rejection_metadata(
    tmp_path, monkeypatch,
):
    output_dir = tmp_path / "scout"
    checkpoint_dir = output_dir / "step_000025"
    checkpoint_dir.mkdir(parents=True)
    for path in (
        output_dir / "metrics.jsonl",
        output_dir / "explorer_trajectories.jsonl",
        output_dir / "rollouts_all.jsonl",
        checkpoint_dir / "trainer_state.json",
    ):
        path.write_text("{}\n")

    captured = {}

    class FakeArtifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    class Logged:
        def wait(self):
            return None

    class FakeRun:
        def log_artifact(self, artifact, aliases):
            captured["artifact"] = artifact
            captured["aliases"] = aliases
            return Logged()

    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Artifact=FakeArtifact))
    args = SimpleNamespace(
        run_name="component-1",
        phase="scout",
        round=1,
        model=fresh_c8_sft.MODEL_INIT,
        scientific_variant="",
        seed=1,
        fresh_c8_evidence_profile=fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE,
        output_dir=str(output_dir),
        metrics_jsonl=str(output_dir / "metrics.jsonl"),
        code_source_manifest="",
        _code_artifact_binding=None,
    )
    grpo_gpu._log_checkpoint_progress_artifact(
        FakeRun(), args=args, step=25, checkpoint_dir=str(checkpoint_dir))
    metadata = captured["artifact"].metadata
    assert metadata["fresh_c8_evidence_profile"] == \
        fresh_c8_sft.MULTISCOUT_COMPONENT_PROFILE
    assert metadata["component_seed"] == 1
    with pytest.raises(ValueError, match="strict pooled consumer"):
        fresh_c8_sft.reject_multiscout_component_from_generic_pool(
            artifact_metadata=metadata,
            completion_evidence=[],
        )


def test_no_new_multiscout_pool_consumer_is_exposed():
    assert not hasattr(common, "MULTISCOUT_TRAJECTORY_SOURCE_DIGEST_VERSION")
    assert not hasattr(pipeline_gpu, "_validated_multiscout_trajectory_sources")
