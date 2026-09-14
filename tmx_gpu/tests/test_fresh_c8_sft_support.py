from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tmx_gpu import common, fresh_c8_sft, grpo_gpu, pipeline_gpu, sft_gate_gpu
from tmx_gpu.tests.test_sft_checkpoint_selection import (
    fresh_training_evidence,
    make_gate_bundle,
)


def test_expdis_launcher_keeps_runtime_files_outside_immutable_code_artifact():
    launcher = (Path(__file__).resolve().parents[1] / "launch/expdis_job.sh").read_text()
    assert "export PYTHONDONTWRITEBYTECODE=1" in launcher
    assert 'export WANDB_DIR="${WANDB_DIR:-$OUT_DIR/wandb}"' in launcher
    assert 'mkdir -p "$WORK" "$OUT_DIR" "$WANDB_DIR"' in launcher
    assert 'export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in launcher
    assert 'cd "$WORK"\n# shellcheck disable=SC2086\npython3 "$CODE_ROOT/tmx_gpu/pipeline_gpu.py"' in launcher


def test_original_paper_dapo_launcher_pins_central_rate_contract():
    path = Path(__file__).resolve().parents[1] / "launch/dapo_job.sh"
    launcher = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert 'STEPS="${STEPS:-100}"' in launcher
    assert 'LR="${LR:-0.000001}"' in launcher
    trainer = launcher[launcher.index('python3 "$CODE_ROOT/tmx_gpu/grpo_gpu.py"'):]
    for exact in (
        '--scientific-variant original_paper_dapo_100_central_lr',
        '--phase central',
        '--lambda-novelty 0',
        '--prompts-per-step 4',
        '--num-generations 16',
        '--max-prompt-len 2048',
        '--max-completion-len 16384',
        '--completion-budget 0',
        '--loss-denominator-len 16384',
        '--soft-overlong-expected-len 13107',
        '--soft-overlong-cache-len 3277',
        '--temperature 1.0',
        '--top-p 0.95',
        '--top-k 20',
        '--updates-per-rollout 1',
        '--clip-epsilon-low 0.2',
        '--clip-epsilon-high 0.28',
        '--rl-validity-mode c8',
        '--dynamic-max-attempts 64',
        '--optimizer adamw',
        '--adam-beta1 0.9',
        '--adam-beta2 0.95',
        '--adam-eps 1e-8',
        '--weight-decay 0',
        '--grad-clip 1.0',
        '--incorrect-novelty-scale 0.0',
        '--no-mask-truncated',
        '--dataset dapo_math_17k',
        '--max-train-examples 20000',
        '--param-dtype float32',
        '--vllm-max-model-len 36864',
        '--gen-timeout 3600',
        '--gen-concurrency 8',
        '--max-consecutive-gen-failures 3',
        '--chunk-tokens 256',
        '--save-every 25',
        '--ckpt-artifact-every 25',
        '--probe-every 25',
        '--code-source-manifest "$CODE_MANIFEST"',
        '"${RESUME_ARGS[@]}"',
    ):
        assert exact in trainer
        assert trainer.count(exact) == 1

    # Canonical evidence is bound before trainer entry, and successful
    # training cannot return zero when the final durable upload fails.
    assert launcher.index('trap upload_evidence EXIT') < launcher.index(
        '# Download to a digest-scoped root')
    assert launcher.index('common.validate_artifact_download_manifest') < \
        launcher.index('pre-update-1 code binding published') < \
        launcher.index('python3 "$CODE_ROOT/tmx_gpu/grpo_gpu.py"')
    assert 'logged.wait()' in launcher
    assert 'if [ "$status" -eq 0 ] && [ "$upload_status" -ne 0 ]' in launcher
    assert 'trap - EXIT' in launcher

    # Cross-pod resume requires two immutable sources: optimizer-complete
    # checkpoint bytes and the paired append-only metrics/trajectory evidence.
    assert 'RESUME_CHECKPOINT_ARTIFACT must be immutable name:vN' in launcher
    assert 'RESUME_RESULTS_ARTIFACT must be immutable name:vN' in launcher
    assert 'exact checkpoint and results artifacts are required together' in launcher
    assert 'resume artifact membership mismatch' in launcher
    assert 'resume results artifact membership mismatch' in launcher
    assert 'resume checkpoint code binding differs from current exact code' in launcher
    assert 'state_paths' in launcher and 'len(state_paths) != 1' in launcher
    assert 'python3 -m tmx_gpu.dapo_evidence' in launcher
    assert '--pair-record "$RESUME_PAIR_RECORD"' in launcher
    assert 'rm -rf "$PROVENANCE_DIR"' in launcher

    # All runtime output lives outside the immutable code artifact root.
    assert 'export PYTHONDONTWRITEBYTECODE=1' in launcher
    assert 'export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in launcher
    assert 'cd "$WORK"' in launcher
    assert 'cd "$CODE_ROOT"' not in launcher


def _artifact_manifest(root, ref="entity/project/code:v26", digest="code-digest"):
    files = [{
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": common.file_sha256(str(path)),
        "size_bytes": path.stat().st_size,
    } for path in sorted(item for item in root.rglob("*") if item.is_file())]
    normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": {"artifact_ref": ref, "artifact_digest": digest},
        "files": files,
    }
    return {
        **normalized,
        "download_root": str(root.resolve()),
        "file_manifest_sha256": common.canonical_json_sha256(normalized),
    }


def _scout_runtime():
    return copy.deepcopy(fresh_training_evidence()["scout"]["runtime_contract"])


def _fresh_builder_fixture(tmp_path, monkeypatch):
    code_root = tmp_path / "code"
    executing_pipeline = code_root / "tmx_gpu/pipeline_gpu.py"
    executing_pipeline.parent.mkdir(parents=True)
    executing_pipeline.write_text("# exact code\n")
    monkeypatch.setattr(pipeline_gpu, "__file__", str(executing_pipeline))
    code_manifest = _artifact_manifest(code_root)
    code_path = tmp_path / "code_manifest.json"
    code_path.write_text(json.dumps(code_manifest))
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
        "runtime_contract": _scout_runtime(),
        "checkpoint_file_manifest": model_manifest,
    })
    trajectory_path = scout_final.parent / "explorer_trajectories.jsonl"
    with trajectory_path.open("w") as handle:
        for step in range(1, 201):
            for sample in range(64):
                handle.write(json.dumps({
                    "step": step,
                    "problem_id": f"p-{step}-{sample}",
                    "prompt_text": f"problem {step}-{sample}",
                    "completion_text": "reason \\boxed{1}",
                }) + "\n")
    accepted_path = tmp_path / "run/round_01/accepted.jsonl"
    with accepted_path.open("w") as handle:
        for index in range(500):
            handle.write(json.dumps({
                "problem_id": f"accepted-{index}",
                "prompt_text": f"accepted problem {index}",
                "completion_text": "reason \\boxed{1}",
            }) + "\n")
    funnel = {"raw": 12800, "correct": 7000, "clean_correct": 4000,
              "accepted": 500}
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", str(tmp_path / "run"),
        "--run-name", "fresh",
        "--lambda-novelty", "0.5",
        "--fresh-c8-sft-gate",
        "--stop-after-sft",
        "--sft-save-steps", "125,250,500,1000",
        "--code-source-manifest", str(code_path),
        "--vllm-max-model-len", "36864",
        "--probe-every", "0",
    ])
    state = {"stages": {"r01/scout": {
        "final": str(scout_final),
        "traj_paths": [str(trajectory_path)],
    }}}
    completion, completion_path = pipeline_gpu._record_fresh_c8_scout_completion(
        args, scout_final=str(scout_final), trajectory_path=str(trajectory_path))
    state["stages"]["r01/scout"].update({
        "scout_completion_evidence_path": completion_path,
        "scout_completion_evidence_sha256": completion["evidence_sha256"],
    })
    return args, state, trajectory_path, accepted_path, funnel


def test_fresh_c8_training_evidence_builder_proves_exact_contract(
        tmp_path, monkeypatch):
    args, state, trajectory, accepted, funnel = _fresh_builder_fixture(
        tmp_path, monkeypatch)
    pipeline_gpu.validate_args(args)
    assert args.dynamic_max_attempts == 0
    assert pipeline_gpu._resolved_dynamic_max_attempts(args) == 64
    gargs = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model,
        output_dir=str(tmp_path / "positive-scout"), lam=0.5,
        steps=200, round_idx=1, global_step_offset=0)
    assert gargs.dynamic_max_attempts == 64
    assert gargs.optimizer == "adamw"
    evidence = pipeline_gpu._build_fresh_c8_sft_training_evidence(
        args, state, round_idx=1, traj_paths=[str(trajectory)],
        accepted_path=str(accepted), funnel=funnel)
    assert fresh_c8_sft.validate_training_evidence(evidence) == evidence
    assert evidence["trajectories"]["row_count"] == 12800
    assert evidence["accepted_library"]["unique_problem_count"] == 500
    assert evidence["code_source"]["artifact_digest"] == "code-digest"
    assert evidence["scout_completion_evidence"]["trajectory"][
        "semantic_digest"]["row_count"] == 12800


def test_completed_fresh_c8_scout_resume_rejects_trajectory_byte_drift(
        tmp_path, monkeypatch):
    args, state, trajectory, _accepted, _funnel = _fresh_builder_fixture(
        tmp_path, monkeypatch)
    stage = state["stages"]["r01/scout"]
    persisted = pipeline_gpu._validate_persisted_fresh_c8_scout_completion(
        args, stage)
    rows = [json.loads(line) for line in trajectory.read_text().splitlines()]
    rows[0]["completion_text"] = "semantically changed \\boxed{1}"
    trajectory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="trajectory bytes/semantics drifted"):
        pipeline_gpu._validate_persisted_fresh_c8_scout_completion(args, stage)
    assert persisted["trajectory"]["row_count"] == 12800


def test_completed_fresh_c8_scout_resume_rejects_code_byte_drift(
        tmp_path, monkeypatch):
    args, state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    manifest = json.loads(open(args.code_source_manifest, encoding="utf-8").read())
    code_file = Path(manifest["download_root"]) / "tmx_gpu/pipeline_gpu.py"
    code_file.write_text("# changed after Scout completion\n")
    with pytest.raises(ValueError, match="artifact.*SHA|SHA mismatch"):
        pipeline_gpu._validate_persisted_fresh_c8_scout_completion(
            args, state["stages"]["r01/scout"])


def test_completed_fresh_c8_scout_resume_requires_persisted_evidence(
        tmp_path, monkeypatch):
    args, state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    stage = state["stages"]["r01/scout"]
    Path(stage["scout_completion_evidence_path"]).unlink()
    with pytest.raises(RuntimeError, match="lacks.*completion-time evidence"):
        pipeline_gpu._validate_persisted_fresh_c8_scout_completion(args, stage)


def test_fresh_c8_training_evidence_fails_on_trajectory_or_coverage_drift(
        tmp_path, monkeypatch):
    args, state, trajectory, accepted, funnel = _fresh_builder_fixture(
        tmp_path, monkeypatch)
    lines = trajectory.read_text().splitlines()
    trajectory.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises((ValueError, RuntimeError), match="64 rows|drifted"):
        pipeline_gpu._build_fresh_c8_sft_training_evidence(
            args, state, round_idx=1, traj_paths=[str(trajectory)],
            accepted_path=str(accepted), funnel=funnel)

    args, state, trajectory, accepted, funnel = _fresh_builder_fixture(
        tmp_path / "duplicate", monkeypatch)
    rows = [json.loads(line) for line in accepted.read_text().splitlines()]
    rows[-1]["prompt_text"] = rows[0]["prompt_text"]
    accepted.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="500 unique"):
        pipeline_gpu._build_fresh_c8_sft_training_evidence(
            args, state, round_idx=1, traj_paths=[str(trajectory)],
            accepted_path=str(accepted), funnel=funnel)


def test_fresh_c8_pre_gate_sft_is_only_canonical_pending(
        monkeypatch, tmp_path):
    args, state, trajectory, accepted, funnel = _fresh_builder_fixture(
        tmp_path, monkeypatch)
    state["stages"]["r01/harvest"] = {"traj_paths": [str(trajectory)]}
    state["stages"]["r01/filter"] = {
        "accepted": str(accepted),
        "funnel": funnel,
        "stage_mode": "trained_pipeline_filter",
        "accepted_library_digest": common.semantic_sft_jsonl_digest(str(accepted)),
    }

    def fake_sft(sargs):
        with open(sargs.sft_training_evidence_manifest, encoding="utf-8") as handle:
            evidence = json.load(handle)
        final = Path(sargs.output_dir) / "actual_sft_final"
        final.mkdir(parents=True)
        (final / "config.json").write_text("{}")
        (final / "model.safetensors").write_bytes(b"fake-sft")
        model_manifest = common.directory_file_manifest(str(final))
        input_identity = pipeline_gpu._sft_input_identity(
            sargs.accepted, epochs=int(sargs.epochs), seed=int(sargs.seed))
        completed = fresh_c8_sft.build_completed_run_evidence(
            optimizer_steps=1000, micro_steps=1000,
            saved_steps=list(fresh_c8_sft.PREREGISTERED_STEPS),
            training_evidence_sha256=evidence["evidence_sha256"])
        common.atomic_write_json(final / "sft_provenance.json", {
            **input_identity,
            "fresh_c8_training_evidence": evidence,
            "sft_runtime_contract": fresh_c8_sft.SFT_RUNTIME_CONTRACT,
            "completed_run_evidence": completed,
            "optimizer_steps": 1000,
            "micro_steps": 1000,
            "checkpoint_file_manifest": model_manifest,
        })
        return str(final)

    monkeypatch.setattr(pipeline_gpu.sft_gpu, "run_sft", fake_sft)
    summary = pipeline_gpu.run_round(
        args, SimpleNamespace(), None, [], 1, 0.5,
        fresh_c8_sft.MODEL_INIT, None,
        common.MetricsLogger(str(tmp_path / "run/metrics.jsonl")), state)
    stage = state["stages"]["r01/sft"]
    assert stage["canonical_pending_gate"] is True
    assert "canonical" not in stage
    assert summary["canonical_pending_gate"] is True
    assert "canonical" not in summary


@pytest.mark.parametrize(
    "extra,match", [
        (["--sft-grad-accum", "8"], "sft_grad_accum"),
        (["--sft-save-steps", "125,250,500"], "sft_save_steps"),
        (["--completion-budget", "32768"], "completion_budget"),
        (["--policy", "naive_pool"], "policy"),
        (["--seed", "1"], "seed"),
        (["--dynamic-max-attempts", "1"], "dynamic_max_attempts"),
        (["--optimizer", "adafactor"], "optimizer"),
    ],
)
def test_fresh_c8_training_route_rejects_contract_drift(tmp_path, extra, match):
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "x").write_text("x")
    manifest = tmp_path / "code.json"
    manifest.write_text(json.dumps(_artifact_manifest(code_root)))
    argv = [
        "--output-dir", str(tmp_path / "run"), "--run-name", "fresh",
        "--lambda-novelty", "0.5", "--fresh-c8-sft-gate",
        "--stop-after-sft", "--sft-save-steps", "125,250,500,1000",
        "--code-source-manifest", str(manifest), "--probe-every", "0",
        *extra,
    ]
    args = pipeline_gpu.build_arg_parser().parse_args(argv)
    with pytest.raises(ValueError, match=match):
        pipeline_gpu.validate_args(args)


@pytest.mark.parametrize("extra,match", [
    (["--dynamic-max-attempts", "1"], "dynamic_max_attempts"),
    (["--optimizer", "adafactor"], "optimizer"),
])
def test_fresh_scout_sampling_or_optimizer_drift_stops_before_run_grpo(
        tmp_path, monkeypatch, extra, match):
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "x").write_text("x")
    manifest = tmp_path / "code.json"
    manifest.write_text(json.dumps(_artifact_manifest(code_root)))
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", str(tmp_path / "run"), "--run-name", "fresh",
        "--lambda-novelty", "0.5", "--fresh-c8-sft-gate",
        "--stop-after-sft", "--sft-save-steps", "125,250,500,1000",
        "--code-source-manifest", str(manifest), "--probe-every", "0",
        *extra,
    ])
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_grpo must not be entered")

    monkeypatch.setattr(pipeline_gpu.grpo_gpu, "run_grpo", forbidden_run)
    with pytest.raises(ValueError, match=match):
        pipeline_gpu.validate_args(args)
    assert called is False


def _rewrite_code_manifest(args, drift):
    path = Path(args.code_source_manifest)
    manifest = json.loads(path.read_text())
    if drift == "version":
        manifest["source"]["artifact_ref"] = "entity/project/code:v27"
    elif drift == "latest":
        manifest["source"]["artifact_ref"] = "entity/project/code:latest"
    elif drift == "digest":
        manifest["source"]["artifact_digest"] = "different-code-digest"
    elif drift == "manifest":
        member = Path(manifest["download_root"]) / manifest["files"][0][
            "relative_path"]
        member.write_text("# different immutable artifact bytes\n")
        manifest["files"][0]["sha256"] = common.file_sha256(str(member))
        manifest["files"][0]["size_bytes"] = member.stat().st_size
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(drift)
    normalized = {
        "version": manifest["version"],
        "source": manifest["source"],
        "files": manifest["files"],
    }
    manifest["file_manifest_sha256"] = common.canonical_json_sha256(normalized)
    path.write_text(json.dumps(manifest))


def _precompletion_gargs(args, *, binding):
    scout_dir = Path(args.output_dir) / "round_01/scout"
    checkpoint = scout_dir / "step_000025"
    checkpoint.mkdir(parents=True, exist_ok=True)
    state = {"step": 25}
    if binding is not None:
        state["code_artifact_binding"] = binding
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    args.resume = True
    gargs = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model, output_dir=str(scout_dir),
        lam=0.5, steps=200, round_idx=1, global_step_offset=0)
    assert gargs.resume_from == str(checkpoint)
    return gargs


def test_fresh_c8_code_artifact_requires_exact_version(tmp_path, monkeypatch):
    args, _state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    # Exact :vN is accepted and independently re-hashed.
    pipeline_gpu.validate_args(args)
    assert pipeline_gpu._validated_code_source_manifest(args)["source"][
        "artifact_ref"].endswith(":v26")
    _rewrite_code_manifest(args, "latest")
    with pytest.raises(ValueError, match="immutable.*:vN"):
        pipeline_gpu.validate_args(args)


def test_fresh_c8_same_code_identity_resume_succeeds(tmp_path, monkeypatch):
    args, _state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    binding = pipeline_gpu._validated_code_source_manifest(args)
    gargs = _precompletion_gargs(args, binding=binding)
    assert pipeline_gpu._prepare_fresh_c8_scout_code_binding(args, gargs) == binding
    sidecar = Path(gargs.output_dir) / "fresh_c8_code_artifact_binding.json"
    assert json.loads(sidecar.read_text()) == binding
    assert grpo_gpu.load_code_artifact_binding(gargs) == binding
    grpo_gpu.validate_code_bound_local_checkpoints(gargs, binding)
    recorded = json.loads((Path(gargs.resume_from) / "trainer_state.json").read_text())
    grpo_gpu.check_resume_compat(
        recorded, phase="scout", model=args.model, config_tier="contract",
        code_artifact_binding_current=binding)


def test_code_binding_allows_only_lineage_free_original_dapo_central(
        tmp_path, monkeypatch):
    args, _state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    gargs = _precompletion_gargs(
        args, binding=pipeline_gpu._validated_code_source_manifest(args))
    gargs.phase = "central"
    gargs.scientific_variant = grpo_gpu.ORIGINAL_PAPER_DAPO_VARIANT
    gargs.external_lineage_json = ""
    assert grpo_gpu.load_code_artifact_binding(gargs)["source"][
        "artifact_ref"].endswith(":v26")
    gargs.scientific_variant = "other-central"
    with pytest.raises(ValueError, match="DAPO Central-route"):
        grpo_gpu.load_code_artifact_binding(gargs)
    gargs.scientific_variant = grpo_gpu.ORIGINAL_PAPER_DAPO_VARIANT
    gargs.external_lineage_json = str(tmp_path / "external.json")
    with pytest.raises(ValueError, match="DAPO Central-route"):
        grpo_gpu.load_code_artifact_binding(gargs)


@pytest.mark.parametrize("drift", ["version", "digest", "manifest"])
def test_fresh_c8_code_identity_drift_fails_before_run_grpo(
        tmp_path, monkeypatch, drift):
    args, _state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    original = pipeline_gpu._validated_code_source_manifest(args)
    gargs = _precompletion_gargs(args, binding=original)
    _rewrite_code_manifest(args, drift)
    with pytest.raises(RuntimeError, match="binding mismatch before run_grpo"):
        pipeline_gpu._prepare_fresh_c8_scout_code_binding(args, gargs)


def test_fresh_c8_checkpoint_without_code_binding_fails_closed(
        tmp_path, monkeypatch):
    args, _state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    gargs = _precompletion_gargs(args, binding=None)
    with pytest.raises(RuntimeError, match="checkpoint exists.*binding is missing"):
        pipeline_gpu._prepare_fresh_c8_scout_code_binding(args, gargs)


def test_fresh_c8_drift_blocks_run_round_before_run_grpo(
        tmp_path, monkeypatch):
    args, _old_state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    original = pipeline_gpu._validated_code_source_manifest(args)
    _precompletion_gargs(args, binding=original)
    _rewrite_code_manifest(args, "digest")
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_grpo must not be entered")

    monkeypatch.setattr(pipeline_gpu.grpo_gpu, "run_grpo", forbidden_run)
    with pytest.raises(RuntimeError, match="binding mismatch before run_grpo"):
        pipeline_gpu.run_round(
            args, SimpleNamespace(), None, [], 1, 0.5,
            fresh_c8_sft.MODEL_INIT, None,
            common.MetricsLogger(str(Path(args.output_dir) / "metrics.jsonl")),
            {"stages": {}})
    assert called is False


def test_fresh_c8_step_checkpoint_embeds_code_binding(
        tmp_path, monkeypatch):
    args, _state, _trajectory, _accepted, _funnel = \
        _fresh_builder_fixture(tmp_path, monkeypatch)
    binding = pipeline_gpu._validated_code_source_manifest(args)
    gargs = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model,
        output_dir=str(tmp_path / "checkpointed-scout"), lam=0.5,
        steps=200, round_idx=1, global_step_offset=0)
    gargs._code_artifact_binding = binding
    gargs._policy_attention_implementation = "sdpa"
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())

    def fake_save(_model, _tokenizer, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "config.json").write_text("{}")

    monkeypatch.setattr(grpo_gpu, "save_model_training_dtype", fake_save)
    checkpoint = grpo_gpu._save_checkpoint(
        None, None, None, optimizer, gargs, 1, 0.5)
    state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
    assert state["code_artifact_binding"] == binding
    assert state["policy_attention_implementation"] == "sdpa"


def _signed_lineage(step: int, runtime: dict):
    canonical = step == 1000
    lineage = {
        "lineage_contract_version": fresh_c8_sft.LINEAGE_CONTRACT_VERSION,
        "canonical": canonical,
        "scientific_variant": (
            "" if canonical else fresh_c8_sft.EARLY_SELECTION_VARIANT),
        "selected_optimizer_step": step,
        "central_runtime_contract": runtime,
    }
    lineage["lineage_sha256"] = common.canonical_json_sha256(lineage)
    return lineage


@pytest.mark.parametrize("step,variant", [
    (1000, ""), (125, fresh_c8_sft.EARLY_SELECTION_VARIANT),
    (250, fresh_c8_sft.EARLY_SELECTION_VARIANT),
    (500, fresh_c8_sft.EARLY_SELECTION_VARIANT),
])
def test_grpo_accepts_only_correct_fresh_c8_canonical_variant_pair(
        tmp_path, step, variant):
    runtime = {"expected_steps": 100}
    lineage = _signed_lineage(step, runtime)
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(lineage))
    args = SimpleNamespace(
        external_lineage_json=str(path), phase="central",
        scientific_variant=variant)
    assert grpo_gpu.load_external_lineage(
        args, expected_runtime_contract=runtime) == lineage
    args.scientific_variant = "wrong"
    with pytest.raises(ValueError, match="scientific variant"):
        grpo_gpu.load_external_lineage(args, expected_runtime_contract=runtime)


def test_grpo_rejects_fresh_c8_canonical_flag_forgery(tmp_path):
    runtime = {"expected_steps": 100}
    lineage = _signed_lineage(500, runtime)
    lineage["canonical"] = True
    unsigned = dict(lineage)
    unsigned.pop("lineage_sha256")
    lineage["lineage_sha256"] = common.canonical_json_sha256(unsigned)
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(lineage))
    args = SimpleNamespace(
        external_lineage_json=str(path), phase="central",
        scientific_variant=fresh_c8_sft.EARLY_SELECTION_VARIANT)
    with pytest.raises(ValueError, match="canonical flag"):
        grpo_gpu.load_external_lineage(args, expected_runtime_contract=runtime)


def _entry(root, ref, digest, download_manifest_path, **extra):
    manifest = _artifact_manifest(root, ref, digest)
    download_manifest_path.write_text(json.dumps(manifest))
    return {
        "source": manifest["source"],
        "download_manifest": str(download_manifest_path),
        **extra,
    }, manifest


def _fresh_external_fixture(tmp_path, monkeypatch, selected_step):
    args0, state, trajectory, accepted, funnel = _fresh_builder_fixture(
        tmp_path / "training", monkeypatch)
    training = pipeline_gpu._build_fresh_c8_sft_training_evidence(
        args0, state, round_idx=1, traj_paths=[str(trajectory)],
        accepted_path=str(accepted), funnel=funnel)

    code_manifest = json.loads(
        open(args0.code_source_manifest, encoding="utf-8").read())
    code_entry = {
        "source": code_manifest["source"],
        "download_manifest": args0.code_source_manifest,
    }

    trajectory_root = tmp_path / "trajectory_artifact"
    trajectory_target = trajectory_root / "round_01/scout/explorer_trajectories.jsonl"
    trajectory_target.parent.mkdir(parents=True)
    shutil.copyfile(trajectory, trajectory_target)
    trajectory_manifest = {
        "version": common.TRAJECTORY_SOURCE_DIGEST_VERSION,
        "artifacts": [{
            "artifact_ref": "entity/project/fresh-results:v1",
            "artifact_digest": "fresh-results-digest",
            "download_root": str(trajectory_root.resolve()),
            "files": [{
                "relative_path": "round_01/scout/explorer_trajectories.jsonl",
                "local_path": str(trajectory_target.resolve()),
                "sha256": common.file_sha256(str(trajectory_target)),
            }],
        }],
    }
    trajectory_manifest_path = tmp_path / "trajectory_sources.json"
    trajectory_manifest_path.write_text(json.dumps(trajectory_manifest))

    accepted_root = tmp_path / "accepted_artifact"
    accepted_root.mkdir()
    accepted_target = accepted_root / "round_01/accepted.jsonl"
    accepted_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(accepted, accepted_target)
    accepted_entry, accepted_manifest = _entry(
        accepted_root, "entity/project/fresh-results:v1", "fresh-results-digest",
        tmp_path / "accepted-download.json",
        accepted_path=str(accepted_target.resolve()))
    accepted_sources = {
        "version": "wandb_accepted_sources_v2",
        "sources": [{
            **accepted_entry["source"],
            "file_manifest_sha256": accepted_manifest["file_manifest_sha256"],
        }],
    }
    accepted_sources["sha256"] = common.canonical_json_sha256(accepted_sources)

    selected_root = tmp_path / "selected_model"
    selected_root.mkdir()
    (selected_root / "config.json").write_text("{}")
    (selected_root / "model.safetensors").write_bytes(b"fresh selected")
    selected_metadata = {
        "checkpoint_kind": "post_optimizer_step_hf_model",
        "optimizer_step": selected_step,
        **copy.deepcopy(fresh_c8_sft.CHECKPOINT_CONTRACT),
        "accepted_library_digest": training["accepted_library"]["semantic_digest"],
        "fresh_c8_training_evidence": training,
        "completed_run_evidence": fresh_c8_sft.build_completed_run_evidence(
            optimizer_steps=1000, micro_steps=1000,
            saved_steps=list(fresh_c8_sft.PREREGISTERED_STEPS),
            training_evidence_sha256=training["evidence_sha256"]),
    }
    (selected_root / "sft_checkpoint.json").write_text(json.dumps(selected_metadata))
    model_entry, model_manifest = _entry(
        selected_root, f"entity/project/fresh-step-{selected_step}:v0",
        f"model-digest-{selected_step}", tmp_path / "model-download.json",
        model_dir=str(selected_root.resolve()))

    gate_entries = []
    bundles = {}
    gate_sources = {}
    for step in fresh_c8_sft.PREREGISTERED_STEPS:
        passed = step <= selected_step
        bundle = make_gate_bundle(
            tmp_path / f"gate-{step}", step, passed=passed,
            accepted_sources=accepted_sources,
            model_file_manifest=model_manifest if step == selected_step else None,
            selection_profile=fresh_c8_sft.PROFILE,
            training_evidence=training)
        ref = f"entity/project/fresh-gate-{step}:v0"
        digest = f"gate-digest-{step}"
        gate_entry, _ = _entry(
            bundle, ref, digest, tmp_path / f"gate-{step}-download.json",
            optimizer_step=step, bundle_dir=str(bundle.resolve()))
        gate_entries.append(gate_entry)
        bundles[step] = str(bundle)
        gate_sources[step] = f"{ref}#digest={digest}"
    selection = pipeline_gpu.sft_selector.build_selection_manifest_from_bundles(
        bundles, gate_sources, fresh_c8_sft.PROFILE,
        accepted_jsonl=str(accepted_target),
        accepted_artifact_identity=pipeline_gpu._stable_artifact_identity(
            accepted_manifest))
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    selection_path = selection_root / "selection_manifest.json"
    selection_path.write_text(json.dumps(selection))
    selection_entry, _ = _entry(
        selection_root, "entity/project/fresh-selection:v0", "selection-digest",
        tmp_path / "selection-download.json",
        manifest_path=str(selection_path.resolve()))
    index = {
        "version": fresh_c8_sft.EXTERNAL_EVIDENCE_INDEX_VERSION,
        "code": code_entry,
        "model": model_entry,
        "accepted": accepted_entry,
        "gates": gate_entries,
        "selection": selection_entry,
    }
    index_path = tmp_path / "evidence-index.json"
    index_path.write_text(json.dumps(index))
    run_dir = tmp_path / "continuation"
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", str(run_dir), "--run-name", "fresh-continuation",
        "--skip-scout", "--trajectories", str(trajectory_target),
        "--trajectory-source-manifest", str(trajectory_manifest_path),
        "--sft-model", str(selected_root),
        "--sft-evidence-index", str(index_path),
        "--min-accepted", "500", "--max-accepted", "500",
        "--central-steps", "100", "--probe-every", "0",
    ])
    pipeline_gpu.validate_args(args)
    run_dir.mkdir()
    accepted_copy = run_dir / "accepted.jsonl"
    shutil.copyfile(accepted_target, accepted_copy)
    return args, accepted_copy, training


@pytest.mark.parametrize("selected_step,canonical,variant", [
    (1000, True, ""),
    (500, False, fresh_c8_sft.EARLY_SELECTION_VARIANT),
])
def test_fresh_external_continuation_binds_all_evidence_and_labels_selection(
        tmp_path, monkeypatch, selected_step, canonical, variant):
    args, accepted, training = _fresh_external_fixture(
        tmp_path, monkeypatch, selected_step)
    lineage = pipeline_gpu._validate_external_sft_evidence(args, str(accepted))
    assert lineage["canonical"] is canonical
    assert lineage["scientific_variant"] == variant
    assert lineage["selected_optimizer_step"] == selected_step
    assert lineage["fresh_c8_training_evidence"] == training
    assert lineage["central_runtime_contract"]["expected_steps"] == 100
    assert args.dynamic_max_attempts == 0
    assert lineage["central_runtime_contract"]["dynamic_max_attempts"] == 64
    assert lineage["central_runtime_contract"]["scientific_variant"] == variant
    assert args.scientific_variant == variant


def test_fresh_external_continuation_rejects_noncontract_dynamic_attempts(
        tmp_path, monkeypatch):
    args, accepted, _training = _fresh_external_fixture(
        tmp_path, monkeypatch, 1000)
    args.dynamic_max_attempts = 1
    with pytest.raises(ValueError, match="dynamic_max_attempts=64"):
        pipeline_gpu._validate_external_sft_evidence(args, str(accepted))


def test_fresh_external_continuation_rejects_trajectory_byte_drift(
        tmp_path, monkeypatch):
    args, accepted, _training = _fresh_external_fixture(
        tmp_path, monkeypatch, 1000)
    with open(args.trajectories[0], "a", encoding="utf-8") as handle:
        handle.write("{}\n")
    # The outer trajectory-source verifier catches the changed artifact byte
    # before the semantic fresh-c8 lineage comparison can be reached.
    with pytest.raises(ValueError, match="trajectory source SHA mismatch"):
        pipeline_gpu._validate_external_sft_evidence(args, str(accepted))


def test_fresh_external_continuation_recomputes_heldout_disjointness(
        tmp_path, monkeypatch):
    args, accepted, _training = _fresh_external_fixture(
        tmp_path, monkeypatch, 1000)
    index = json.loads(Path(args.sft_evidence_index).read_text())
    gate_entry = index["gates"][0]
    bundle = Path(gate_entry["bundle_dir"])
    heldout_path = bundle / "heldout_selection.jsonl"
    rows = [json.loads(line) for line in heldout_path.read_text().splitlines()]
    rows[0]["problem_id"] = "accepted-0"
    rows[0]["selection_digest"] = sft_gate_gpu._sha256(
        f"20260713\0accepted-0\0{rows[0]['problem']}")
    heldout_path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))
    aggregate_path = bundle / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["selection"]["heldout_selection_sha256"] = \
        common.canonical_json_sha256([{
            "problem_id": row["problem_id"],
            "selection_digest": row["selection_digest"],
            "training_prompt_sha256": row["training_prompt_sha256"],
        } for row in rows])
    aggregate_path.write_text(json.dumps(aggregate))
    gate_entry["download_manifest"] = str(Path(
        gate_entry["download_manifest"]))
    refreshed = _artifact_manifest(
        bundle, gate_entry["source"]["artifact_ref"],
        gate_entry["source"]["artifact_digest"])
    Path(gate_entry["download_manifest"]).write_text(json.dumps(refreshed))
    with pytest.raises(ValueError, match="problem-ID overlaps"):
        pipeline_gpu._validate_external_sft_evidence(args, str(accepted))


def test_fresh_c8_central_resume_binds_lineage_and_runtime():
    runtime = {
        "version": grpo_gpu.GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": "central",
        "expected_steps": 100,
        "scientific_variant": "",
    }
    lineage = _signed_lineage(1000, runtime)
    recorded = {
        "phase": "central",
        "model": "selected-fresh-c8-model",
        "config_tier": "contract",
        "rl_validity_mode": "c8",
        "optimizer_state_saved": True,
        "resolved_lr": 1e-6,
        "external_lineage": lineage,
        "external_lineage_sha256": lineage["lineage_sha256"],
        "runtime_contract": copy.deepcopy(runtime),
    }
    kwargs = {
        "phase": "central",
        "model": "selected-fresh-c8-model",
        "config_tier": "contract",
        "rl_validity_mode_current": "c8",
        "resolved_lr_current": 1e-6,
        "external_lineage_current": lineage,
        "runtime_contract_current": runtime,
    }
    grpo_gpu.check_resume_compat(recorded, **kwargs)

    changed_lineage = copy.deepcopy(lineage)
    changed_lineage["selected_optimizer_step"] = 500
    with pytest.raises(RuntimeError, match="external_lineage"):
        grpo_gpu.check_resume_compat(
            recorded, **{**kwargs, "external_lineage_current": changed_lineage})

    changed_runtime = copy.deepcopy(runtime)
    changed_runtime["expected_steps"] = 99
    with pytest.raises(RuntimeError, match="runtime_contract"):
        grpo_gpu.check_resume_compat(
            recorded, **{**kwargs, "runtime_contract_current": changed_runtime})


def test_fresh_c8_completed_central_reuse_is_fail_closed(tmp_path):
    runtime = {
        "version": grpo_gpu.GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": "central",
        "expected_steps": 100,
        "scientific_variant": "",
    }
    lineage = _signed_lineage(1000, runtime)
    model_dir = tmp_path / "central-final"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"central")
    file_manifest = common.directory_file_manifest(
        str(model_dir), exclude_relative_paths=("trainer_state.json",))
    state = {
        "phase": "central",
        "checkpoint_kind": "deployable_model",
        "step": 100,
        "external_lineage": lineage,
        "external_lineage_sha256": lineage["lineage_sha256"],
        "runtime_contract": copy.deepcopy(runtime),
        "checkpoint_file_manifest": file_manifest,
    }
    state_path = model_dir / "trainer_state.json"
    state_path.write_text(json.dumps(state))
    assert pipeline_gpu._validate_completed_central(
        str(model_dir), lineage, expected_steps=100,
        expected_runtime_contract=runtime) == state

    tampered = copy.deepcopy(state)
    tampered["runtime_contract"]["expected_steps"] = 99
    state_path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="runtime contract mismatch"):
        pipeline_gpu._validate_completed_central(
            str(model_dir), lineage, expected_steps=100,
            expected_runtime_contract=runtime)
