"""Hardening-pass tests (massverify findings): atomic state writes, harvest
robustness + resume-at-scout-policy, trajectory-flag validation, lambda
precedence, probe sizing/NaN honesty, eval error-row exclusion and n>=k
guards, resume config validation, and vLLM pool crash paths."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from tmx_gpu import common, eval_gpu, grpo_gpu, pipeline_gpu, vllm_pool
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures
from tmx_gpu.vllm_pool import Completion, VLLMPool


def test_sft_gate_launcher_has_validated_configurable_server_timeout():
    script = (
        Path(__file__).resolve().parents[2]
        / "tmx_gpu" / "launch" / "sft_gate_job.sh"
    ).read_text()
    assert 'SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-600}"' in script
    assert '[[ "$SERVER_READY_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]' in script
    assert (
        'VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$SERVER_READY_TIMEOUT_SECONDS}"'
        in script
    )
    assert 'SERVER_READY_POLL_COUNT=$(( (SERVER_READY_TIMEOUT_SECONDS + 4) / 5 ))' in script
    assert 'seq 1 "$SERVER_READY_POLL_COUNT"' in script

# ------------------------------------------------------------ fakes/helpers ----

class FakeProc:
    """Popen stand-in: alive until terminate()/kill()."""

    def __init__(self, pid: int = 4_000_000):
        self.pid = pid
        self.rc = None

    def poll(self):
        return self.rc

    @property
    def returncode(self):
        return self.rc

    def terminate(self):
        self.rc = -15

    def kill(self):
        self.rc = -9


class FakeHarvestPool:
    """VLLMPool stand-in for pipeline harvest tests."""

    def __init__(self, completion_factory, *, managed=True, model_dir="base"):
        self.managed = managed
        self.model_dir = model_dir
        self.health_checks = 0
        self.reloads = []
        self.generate_calls = 0
        self._factory = completion_factory

    def health_check(self):
        self.health_checks += 1
        return 0

    def reload(self, model_dir):
        self.reloads.append(model_dir)
        self.model_dir = model_dir

    def generate(self, prompts, n, **kwargs):
        self.generate_calls += 1
        return [self._factory(int(n)) for _ in prompts]


def make_traj_row(*, correct=True, tokens=500, reward=1.0):
    text = " ".join(f"step {i} explores a fresh idea" for i in range(40))
    text += " \\boxed{4}"
    return {
        "problem_id": "p0", "prompt_text": "prompt", "completion_text": text,
        "ground_truth": "4", "is_correct": correct, "valid_answer": True,
        "terminated": True, "clipped": False,
        "completion_token_length": tokens, "blended_reward": reward,
        "reward": reward, "step": 1,
    }


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def make_examples(n=3):
    return [SimpleNamespace(problem_id=f"p{i}",
                            prompt_text=f"Compute {i} + {i}. [gt={2 * i}]",
                            ground_truth=str(2 * i)) for i in range(n)]


def pipeline_args(tmp_path, *extra):
    return pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", str(tmp_path), "--run-name", "hardening",
        *extra,
    ])


def no_pgid(*_a, **_k):
    raise ProcessLookupError("fake pid has no process group")


# ------------------------------------------------------------- atomic writes ----

class TestAtomicWrites:
    def test_atomic_write_json(self, tmp_path):
        path = str(tmp_path / "state.json")
        common.atomic_write_json(path, {"a": 1})
        common.atomic_write_json(path, {"a": 2})
        assert json.load(open(path)) == {"a": 2}
        assert not os.path.exists(path + ".tmp")

    def test_mark_stage_is_atomic_and_valid(self, tmp_path):
        args = SimpleNamespace(output_dir=str(tmp_path))
        state = {"stages": {}}
        pipeline_gpu._mark_stage(args, state, 1, "scout", final="f", traj_paths=["t"])
        on_disk = json.load(open(tmp_path / "pipeline_state.json"))
        assert on_disk["stages"]["r01/scout"]["final"] == "f"
        assert not os.path.exists(str(tmp_path / "pipeline_state.json") + ".tmp")

    def test_artifact_manifest_rejects_extra_files_and_root_escape(self, tmp_path):
        root = tmp_path / "artifact"
        root.mkdir()
        member = root / "model.safetensors"
        member.write_bytes(b"weights")
        source = {"artifact_ref": "entity/project/model:v1", "artifact_digest": "d"}
        files = [{
            "relative_path": "model.safetensors",
            "sha256": common.file_sha256(str(member)),
            "size_bytes": member.stat().st_size,
        }]
        normalized = {
            "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
            "source": source, "files": files,
        }
        manifest = {
            **normalized, "download_root": str(root),
            "file_manifest_sha256": common.canonical_json_sha256(normalized),
        }
        common.validate_artifact_download_manifest(manifest, expected_source=source)
        (root / "stale.bin").write_bytes(b"stale")
        with pytest.raises(ValueError, match="membership mismatch"):
            common.validate_artifact_download_manifest(manifest, expected_source=source)
        (root / "stale.bin").unlink()
        escaped = copy.deepcopy(manifest)
        escaped["files"][0]["relative_path"] = "../outside"
        with pytest.raises(ValueError, match="unsafe relative_path"):
            common.validate_artifact_download_manifest(escaped)

    def test_directory_manifests_reject_symlink_directories(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "weights.bin").write_bytes(b"outside")
        root = tmp_path / "root"
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "linked").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            common.directory_file_manifest(str(root))

        member = root / "config.json"
        source = {"artifact_ref": "entity/project/a:v0", "artifact_digest": "d"}
        files = [{
            "relative_path": "config.json",
            "sha256": common.file_sha256(str(member)),
            "size_bytes": member.stat().st_size,
        }]
        normalized = {
            "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
            "source": source, "files": files,
        }
        manifest = {
            **normalized, "download_root": str(root),
            "file_manifest_sha256": common.canonical_json_sha256(normalized),
        }
        with pytest.raises(ValueError, match="symlink"):
            common.validate_artifact_download_manifest(manifest)

    def test_trim_jsonl_swaps_no_tmp(self, tmp_path):
        path = str(tmp_path / "t.jsonl")
        write_jsonl(path, [{"step": s} for s in (1, 2, 3, 4)])
        common.trim_jsonl_to_step(path, 2)
        assert [json.loads(l)["step"] for l in open(path)] == [1, 2]
        assert not os.path.exists(path + ".tmp")

    def test_trim_metrics_swaps_no_tmp(self, tmp_path):
        path = str(tmp_path / "m.jsonl")
        write_jsonl(path, [{"phase": "scout_grpo", "round": 1, "global_step": s}
                           for s in (1, 5)])
        common.trim_metrics_stage(path, phase="scout_grpo", round_idx=1,
                                  keep_upto_global_step=1)
        assert len(open(path).readlines()) == 1
        assert not os.path.exists(path + ".tmp")


# --------------------------------------------------------- harvest robustness ----

class TestHarvestRobustness:
    def test_zero_row_pass_never_registers_harvest_file(self, tmp_path, tiny_model_dir):
        """All-failed generations: the harvest file must not be created OR
        registered (a registered-but-unwritten path crashed _policy_survivors
        with FileNotFoundError); health_check runs between passes."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        args = pipeline_args(tmp_path, "--min-accepted", "1",
                             "--harvest-max-passes", "2",
                             "--harvest-prompts-per-pass", "1",
                             "--num-generations", "2")
        pool = FakeHarvestPool(
            lambda n: [Completion(text="", finish_reason="error", error="x")] * n)
        harvest_path = str(tmp_path / "harvest_trajectories.jsonl")
        out = pipeline_gpu._harvest(args, pool, tokenizer, make_examples(),
                                    [], harvest_path, 1,
                                    expected_policy=None, strict_policy=False)
        assert out == []
        assert not os.path.exists(harvest_path)
        assert pool.health_checks == 2  # once per pass

    def test_resume_reloads_scout_final_and_below_min_stays_incomplete(
            self, tmp_path, tiny_model_dir):
        """Resume path (scout skipped via pipeline_state): harvest reloads the
        pool to the RECORDED scout-final policy before generating; a harvest
        still below --min-accepted is NOT marked complete (so --resume retries
        top-up) and the filter floor raises."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        scout_dir = tmp_path / "scout_final"
        scout_dir.mkdir()
        traj = write_jsonl(tmp_path / "scout_traj.jsonl", [make_traj_row()])
        args = pipeline_args(tmp_path, "--min-accepted", "3",
                             "--max-accepted", "10",
                             "--harvest-max-passes", "2",
                             "--harvest-prompts-per-pass", "1",
                             "--num-generations", "2")
        state = {"stages": {"r01/scout": {"final": str(scout_dir),
                                          "traj_paths": [traj]}}}
        # Top-up rows are all WRONG -> survivors stay at 1 (< 3).
        wrong = " ".join(f"step {i} reasons anew" for i in range(50)) + " \\boxed{999}."
        pool = FakeHarvestPool(
            lambda n: [Completion(text=wrong, finish_reason="stop")] * n)
        metrics_logger = common.MetricsLogger(str(tmp_path / "metrics.jsonl"))
        with pytest.raises(RuntimeError, match="min-accepted"):
            pipeline_gpu.run_round(args, pool, tokenizer, make_examples(),
                                   1, 0.5, "base", None, metrics_logger, state)
        assert pool.reloads == [str(scout_dir)], \
            "harvest must reload the pool to the recorded scout-final policy"
        assert "r01/harvest" not in state["stages"], \
            "below-min harvest must stay incomplete so resume retries top-up"

    def test_attach_pool_on_resume_path_refuses_to_harvest(self, tmp_path, tiny_model_dir):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        traj = write_jsonl(tmp_path / "scout_traj.jsonl", [make_traj_row()])
        args = pipeline_args(tmp_path, "--min-accepted", "3")
        pool = FakeHarvestPool(lambda n: [], managed=False)
        with pytest.raises(RuntimeError, match="refusing to harvest"):
            pipeline_gpu._harvest(args, pool, tokenizer, make_examples(),
                                  [traj], str(tmp_path / "h.jsonl"), 1,
                                  expected_policy=str(tmp_path / "scout_final"),
                                  strict_policy=True)
        assert pool.generate_calls == 0, "must refuse BEFORE any generation"

    def test_missing_scout_checkpoint_refuses(self, tmp_path, tiny_model_dir):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        traj = write_jsonl(tmp_path / "scout_traj.jsonl", [make_traj_row()])
        args = pipeline_args(tmp_path, "--min-accepted", "3")
        pool = FakeHarvestPool(lambda n: [], managed=True, model_dir="base")
        with pytest.raises(RuntimeError, match="does not exist"):
            pipeline_gpu._harvest(args, pool, tokenizer, make_examples(),
                                  [traj], str(tmp_path / "h.jsonl"), 1,
                                  expected_policy=str(tmp_path / "gone"),
                                  strict_policy=True)
        assert pool.generate_calls == 0


# ------------------------------------------------------------ arg validation ----

class TestArgValidation:
    def test_trajectories_without_skip_scout_errors(self, tmp_path):
        args = pipeline_args(tmp_path, "--trajectories", "a.jsonl")
        with pytest.raises(ValueError, match="--skip-scout"):
            pipeline_gpu.validate_args(args)

    def test_skip_scout_without_trajectories_errors(self, tmp_path):
        args = pipeline_args(tmp_path, "--skip-scout")
        with pytest.raises(ValueError, match="--trajectories"):
            pipeline_gpu.validate_args(args)

    def test_both_flags_ok(self, tmp_path):
        args = pipeline_args(tmp_path, "--skip-scout", "--trajectories", "a.jsonl")
        pipeline_gpu.validate_args(args)

    def test_min_above_max_accepted_errors(self, tmp_path):
        args = pipeline_args(tmp_path, "--min-accepted", "600", "--max-accepted", "500")
        with pytest.raises(ValueError, match="max-accepted"):
            pipeline_gpu.validate_args(args)

    def test_sft_model_requires_single_round_skip_scout_continuation(self, tmp_path):
        model = tmp_path / "selected_sft"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "sft_checkpoint.json").write_text(json.dumps({
            "termination_mode": "append_eos", "optimizer_step": 125,
        }))

        missing_evidence = pipeline_args(
            tmp_path, "--skip-scout", "--trajectories", "scout.jsonl",
            "--sft-model", str(model),
        )
        with pytest.raises(ValueError, match="exact gate/selection|missing"):
            pipeline_gpu.validate_args(missing_evidence)

        no_skip = pipeline_args(
            tmp_path, "--trajectories", "scout.jsonl", "--sft-model", str(model))
        with pytest.raises(ValueError, match="--skip-scout"):
            pipeline_gpu.validate_args(no_skip)

        multi = pipeline_args(
            tmp_path, "--rounds", "2", "--skip-scout", "--trajectories",
            "scout.jsonl", "--sft-model", str(model))
        with pytest.raises(ValueError, match="--rounds 1"):
            pipeline_gpu.validate_args(multi)

        stopped = pipeline_args(
            tmp_path, "--skip-scout", "--trajectories", "scout.jsonl",
            "--sft-model", str(model), "--stop-after-sft")
        with pytest.raises(ValueError, match="must continue to Central"):
            pipeline_gpu.validate_args(stopped)

        harvested = pipeline_args(
            tmp_path, "--skip-scout", "--trajectories", "scout.jsonl",
            "--sft-model", str(model), "--harvest-max-passes", "1")
        with pytest.raises(ValueError, match="original Scout trajectories"):
            pipeline_gpu.validate_args(harvested)

        no_central = pipeline_args(
            tmp_path, "--skip-scout", "--trajectories", "scout.jsonl",
            "--sft-model", str(model), "--central-steps", "0")
        with pytest.raises(ValueError, match="central-steps"):
            pipeline_gpu.validate_args(no_central)

    def test_sft_model_validation_checks_source_path_and_termination(self, tmp_path):
        source_only = pipeline_args(tmp_path, "--sft-evidence-index", "evidence.json")
        with pytest.raises(ValueError, match="requires --sft-model"):
            pipeline_gpu.validate_args(source_only)

        missing = pipeline_args(
            tmp_path, "--skip-scout", "--trajectories", "scout.jsonl",
            "--sft-model", str(tmp_path / "missing"))
        with pytest.raises(ValueError, match="not a local directory"):
            pipeline_gpu.validate_args(missing)

        model = tmp_path / "legacy_sft"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "sft_checkpoint.json").write_text(json.dumps({
            "termination_mode": "legacy",
        }))
        mismatch = pipeline_args(
            tmp_path, "--skip-scout", "--trajectories", "scout.jsonl",
            "--sft-model", str(model))
        with pytest.raises(ValueError, match="termination_mode"):
            pipeline_gpu.validate_args(mismatch)


class TestExternalSftContinuation:
    def test_frozen_library_route_trains_sft_byte_exact_and_stops(
            self, monkeypatch, tmp_path):
        from tmx_gpu import select_sft_checkpoint

        trajectory_root = tmp_path / "trajectory_artifact"
        trajectory_root.mkdir()
        trajectory_path = write_jsonl(
            trajectory_root / "scout.jsonl", [make_traj_row()])
        trajectory_manifest = {
            "version": common.TRAJECTORY_SOURCE_DIGEST_VERSION,
            "artifacts": [{
                "artifact_ref": "entity/project/scout:v0",
                "artifact_digest": "scout-digest",
                "download_root": str(trajectory_root.resolve()),
                "files": [{
                    "relative_path": "scout.jsonl",
                    "local_path": trajectory_path,
                    "sha256": common.file_sha256(trajectory_path),
                }],
            }],
        }
        trajectory_manifest_path = tmp_path / "trajectory.json"
        trajectory_manifest_path.write_text(json.dumps(trajectory_manifest))

        accepted_root = tmp_path / "accepted_artifact"
        accepted_root.mkdir()
        accepted_path = write_jsonl(
            accepted_root / "accepted.jsonl", [make_traj_row()])
        accepted_digest = common.semantic_sft_jsonl_digest(accepted_path)
        accepted_artifact_digest = "accepted-artifact-digest"
        source = {
            "artifact_ref": (
                "entity/project/"
                "gpu-jrl-c8-c7reuse-sft-ga8-checkpoints-20260713-results:v1"),
            "artifact_digest": accepted_artifact_digest,
        }
        file_entry = {
            "relative_path": "accepted.jsonl",
            "sha256": common.file_sha256(accepted_path),
            "size_bytes": os.path.getsize(accepted_path),
        }
        normalized = {
            "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
            "source": source, "files": [file_entry],
        }
        download = {
            **normalized, "download_root": str(accepted_root.resolve()),
            "file_manifest_sha256": common.canonical_json_sha256(normalized),
        }
        download_path = tmp_path / "accepted-download.json"
        download_path.write_text(json.dumps(download))
        evidence = {
            "version": pipeline_gpu.FROZEN_SFT_LIBRARY_EVIDENCE_INDEX_VERSION,
            "accepted": {
                "source": source,
                "download_manifest": str(download_path),
                "accepted_path": accepted_path,
            },
        }
        evidence_path = tmp_path / "frozen-evidence.json"
        evidence_path.write_text(json.dumps(evidence))
        monkeypatch.setattr(
            pipeline_gpu, "C7_REUSE_ACCEPTED_ARTIFACT_DIGEST",
            accepted_artifact_digest)
        monkeypatch.setattr(
            pipeline_gpu, "C7_REUSE_ACCEPTED_DIGEST", accepted_digest)

        args = pipeline_args(
            tmp_path / "run", "--skip-scout", "--trajectories", trajectory_path,
            "--trajectory-source-manifest", str(trajectory_manifest_path),
            "--frozen-sft-library-evidence-index", str(evidence_path),
            "--stop-after-sft", "--sft-grad-accum", "8",
            "--sft-save-steps", "16,32,64,126", "--min-accepted", "1",
            "--max-accepted", "500", "--probe-every", "0",
        )
        pipeline_gpu.validate_args(args)
        called = {}

        def forbidden_filter(*_args, **_kwargs):
            raise AssertionError("frozen-library route must not re-filter")

        def fake_sft(sargs):
            called["accepted_bytes"] = open(sargs.accepted, "rb").read()
            called["grad_accum"] = sargs.grad_accum
            called["save_steps"] = sargs.sft_save_steps
            final = tmp_path / "run/round_01/sft/actual_sft_final"
            final.mkdir(parents=True)
            (final / "config.json").write_text("{}")
            (final / "model.safetensors").write_bytes(b"fake-sft")
            model_manifest = common.directory_file_manifest(str(final))
            input_identity = pipeline_gpu._sft_input_identity(
                sargs.accepted, epochs=int(sargs.epochs), seed=int(sargs.seed))
            completed = {
                "version": "sft_completed_run_evidence_v1",
                "training_completed": True,
                "optimizer_steps_executed": 126,
                "micro_steps_executed": 1000,
                "saved_optimizer_steps": [16, 32, 64, 126],
                "training_evidence_sha256": None,
            }
            completed["evidence_sha256"] = common.canonical_json_sha256(
                completed)
            common.atomic_write_json(final / "sft_provenance.json", {
                **input_identity,
                "accepted_library_digest": accepted_digest,
                "trajectory_source": {
                    "digest": common.trajectory_source_manifest_digest(
                        trajectory_manifest),
                },
                "sft_runtime_contract": (
                    select_sft_checkpoint.DIAGNOSTIC_SFT_RUNTIME_CONTRACT),
                "optimizer_steps": 126,
                "micro_steps": 1000,
                "completed_run_evidence": completed,
                "checkpoint_file_manifest": model_manifest,
            })
            return str(final)

        monkeypatch.setattr(
            pipeline_gpu.filter_pool_mod, "pool_trajectories", forbidden_filter)
        monkeypatch.setattr(pipeline_gpu.sft_gpu, "run_sft", fake_sft)
        state = {"stages": {}}
        summary = pipeline_gpu.run_round(
            args, FakeHarvestPool(lambda _n: []), None, [], 1, 0.5, "base",
            None, common.MetricsLogger(str(tmp_path / "run/metrics.jsonl")), state)

        assert called["accepted_bytes"] == open(accepted_path, "rb").read()
        assert called["grad_accum"] == 8
        assert called["save_steps"] == (16, 32, 64, 126)
        assert summary["stopped_after_sft"] is True
        assert summary["canonical"] is False
        assert summary["scientific_variant"] == \
            pipeline_gpu.FROZEN_SFT_LIBRARY_VARIANT
        assert state["stages"]["r01/filter"]["stage_mode"] == \
            "frozen_c7_library_training_input"
        assert state["stages"]["r01/sft"]["stage_mode"] == \
            "frozen_c7_library_ga8_trained"

    def test_raw_all_four_selection_skips_sft_and_binds_central_lineage(
            self, monkeypatch, tmp_path, tiny_model_dir):
        from transformers import AutoTokenizer

        from tmx_gpu import select_sft_checkpoint
        from tmx_gpu.tests.test_sft_checkpoint_selection import make_gate_bundle

        def artifact_entry(root, ref, digest, manifest_name, **extra):
            root = root.resolve()
            files = []
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                files.append({
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": common.file_sha256(str(path)),
                    "size_bytes": path.stat().st_size,
                })
            source = {"artifact_ref": ref, "artifact_digest": digest}
            normalized = {
                "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
                "source": source, "files": files,
            }
            manifest = {
                **normalized,
                "download_root": str(root),
                "file_manifest_sha256": common.canonical_json_sha256(normalized),
            }
            manifest_path = tmp_path / manifest_name
            manifest_path.write_text(json.dumps(manifest))
            return {
                "source": source, "download_manifest": str(manifest_path), **extra,
            }, manifest

        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        source_row = make_traj_row()
        trajectory_root = tmp_path / "trajectory_artifact"
        trajectory_file = trajectory_root / "round_01/scout/explorer_trajectories.jsonl"
        trajectory_file.parent.mkdir(parents=True)
        trajectory = write_jsonl(trajectory_file, [source_row])
        trajectory_manifest = {
            "version": common.TRAJECTORY_SOURCE_DIGEST_VERSION,
            "artifacts": [{
                "artifact_ref": "entity/project/scout-results:v0",
                "artifact_digest": "trajectory-artifact-digest",
                "download_root": str(trajectory_root.resolve()),
                "files": [{
                    "relative_path": "round_01/scout/explorer_trajectories.jsonl",
                    "local_path": trajectory,
                    "sha256": common.file_sha256(trajectory),
                }],
            }],
        }
        trajectory_manifest_path = tmp_path / "trajectory_sources.json"
        trajectory_manifest_path.write_text(json.dumps(trajectory_manifest))
        trajectory_digest = common.trajectory_source_manifest_digest(trajectory_manifest)
        accepted_digest = common.semantic_sft_dataset_digest([source_row])
        monkeypatch.setattr(pipeline_gpu, "C7_REUSE_ACCEPTED_DIGEST", accepted_digest)
        monkeypatch.setattr(
            pipeline_gpu, "C7_REUSE_ACCEPTED_ARTIFACT_DIGEST",
            "accepted-artifact-digest")

        accepted_root = tmp_path / "accepted_artifact"
        accepted_root.mkdir()
        frozen_accepted = write_jsonl(accepted_root / "accepted.jsonl", [source_row])
        accepted_entry, accepted_download = artifact_entry(
            accepted_root,
            "entity/project/gpu-jrl-c8-c7reuse-sft-ga8-checkpoints-20260713-results:v1",
            "accepted-artifact-digest", "accepted-download.json",
            accepted_path=frozen_accepted,
        )
        accepted_sources = {
            "version": "wandb_accepted_sources_v2",
            "sources": [{
                **accepted_entry["source"],
                "file_manifest_sha256": accepted_download["file_manifest_sha256"],
            }],
        }
        accepted_sources["sha256"] = common.canonical_json_sha256(accepted_sources)

        selected = tmp_path / "selected_model_artifact"
        selected.mkdir()
        (selected / "config.json").write_text("{}")
        (selected / "model.safetensors").write_bytes(b"selected weights")
        selected_metadata = {
            "checkpoint_kind": "post_optimizer_step_hf_model",
            "optimizer_step": 126,
            **select_sft_checkpoint.DIAGNOSTIC_CONTRACT,
            "accepted_library_digest": accepted_digest,
            "trajectory_source": {
                "manifest": {}, "digest": trajectory_digest,
            },
        }
        (selected / "sft_checkpoint.json").write_text(json.dumps(selected_metadata))
        model_entry, selected_model_manifest = artifact_entry(
            selected, "entity/project/model-step-126:v0", "model-digest-126",
            "selected-model-download.json", model_dir=str(selected.resolve()))

        bundles = {}
        gate_sources = {
            step: f"entity/project/gate-step-{step}:v0#digest=gate-digest-{step}"
            for step in select_sft_checkpoint.PREREGISTERED_STEPS
        }
        gate_entries = []
        for step in select_sft_checkpoint.PREREGISTERED_STEPS:
            model_manifest = selected_model_manifest if step == 126 else None
            bundle = make_gate_bundle(
                tmp_path / f"gate-artifact-{step}", step,
                passed=step in (16, 32, 126),
                accepted_digest=accepted_digest,
                trajectory_digest=trajectory_digest,
                model_file_manifest=model_manifest,
                accepted_sources=accepted_sources,
            )
            bundles[step] = str(bundle)
            gate_entry, _ = artifact_entry(
                bundle, f"entity/project/gate-step-{step}:v0",
                f"gate-digest-{step}", f"gate-{step}-download.json",
                optimizer_step=step, bundle_dir=str(bundle.resolve()))
            gate_entries.append(gate_entry)
        selection_manifest = select_sft_checkpoint.build_selection_manifest_from_bundles(
            bundles, gate_sources)
        selection_root = tmp_path / "selection_artifact"
        selection_root.mkdir()
        selection_path = selection_root / "selection_manifest.json"
        selection_path.write_text(json.dumps(selection_manifest))
        selection_entry, _ = artifact_entry(
            selection_root, "entity/project/selection:v0", "selection-digest",
            "selection-download.json", manifest_path=str(selection_path.resolve()))
        evidence_index = {
            "version": pipeline_gpu.EXTERNAL_EVIDENCE_INDEX_VERSION,
            "model": model_entry,
            "accepted": accepted_entry,
            "gates": gate_entries,
            "selection": selection_entry,
        }
        evidence_path = tmp_path / "evidence-index.json"
        evidence_path.write_text(json.dumps(evidence_index))
        args = pipeline_args(
            tmp_path,
            "--skip-scout", "--trajectories", trajectory,
            "--sft-model", str(selected),
            "--sft-evidence-index", str(evidence_path),
            "--trajectory-source-manifest", str(trajectory_manifest_path),
            "--min-accepted", "1", "--max-accepted", "500",
            "--central-steps", "100", "--probe-every", "0",
        )
        pipeline_gpu.validate_args(args)
        pool = FakeHarvestPool(lambda n: [], managed=True, model_dir="base")
        central_dir = tmp_path / "central_final"
        central_dir.mkdir()
        called = {}

        def forbidden_sft(_args):
            raise AssertionError("external SFT continuation must not retrain SFT")

        def fake_grpo(cargs, pool):
            called["phase"] = cargs.phase
            called["model"] = cargs.model
            called["lambda"] = cargs.lambda_novelty
            called["scientific_variant"] = cargs.scientific_variant
            called["pool_model_at_entry"] = pool.model_dir
            lineage = json.load(open(cargs.external_lineage_json))
            (central_dir / "config.json").write_text("{}")
            (central_dir / "model.safetensors").write_bytes(b"central")
            file_manifest = common.directory_file_manifest(
                str(central_dir), exclude_relative_paths=("trainer_state.json",))
            common.atomic_write_json(central_dir / "trainer_state.json", {
                "step": 100,
                "phase": "central", "checkpoint_kind": "deployable_model",
                "external_lineage": lineage,
                "external_lineage_sha256": lineage["lineage_sha256"],
                "runtime_contract": lineage["central_runtime_contract"],
                "checkpoint_file_manifest": file_manifest,
            })
            return str(central_dir)

        monkeypatch.setattr(pipeline_gpu.sft_gpu, "run_sft", forbidden_sft)
        monkeypatch.setattr(pipeline_gpu.grpo_gpu, "run_grpo", fake_grpo)
        state = {"stages": {}}
        metrics_logger = common.MetricsLogger(str(tmp_path / "metrics.jsonl"))
        summary = pipeline_gpu.run_round(
            args, pool, tokenizer, make_examples(), 1, 0.5, "base",
            None, metrics_logger, state,
        )

        assert called == {
            "phase": "central", "model": str(selected), "lambda": 0.0,
            "pool_model_at_entry": str(selected),
            "scientific_variant": "external_sft_exposure_gated",
        }
        assert pool.reloads == [str(selected.resolve())]
        assert summary["central_final"] == str(central_dir)
        assert summary["sft_stage_mode"] == "external_health_gate_selected"
        assert summary["sft_training_skipped"] is True
        assert summary["sft_health_gate_selected"] is True

        stage = state["stages"]["r01/sft"]
        assert stage["final"] == str(selected)
        assert stage["stage_mode"] == "external_health_gate_selected"
        assert stage["health_gate_selected"] is True
        assert stage["sft_training_skipped"] is True
        provenance = json.load(open(tmp_path / "round_01" /
                                    "sft_selection_provenance.json"))
        assert provenance["source"] == \
            "entity/project/model-step-126:v0#digest=model-digest-126"
        assert provenance["metadata_file"] == "sft_checkpoint.json"
        assert provenance["metadata"]["optimizer_step"] == 126
        assert provenance["canonical"] is False
        assert provenance["scientific_variant"] == "external_sft_exposure_gated"
        assert provenance["validated_external_evidence"]["selected_optimizer_step"] == 126
        accepted_copy = tmp_path / "round_01" / "accepted.jsonl"
        assert os.path.isfile(accepted_copy)
        assert common.semantic_sft_jsonl_digest(str(accepted_copy)) == accepted_digest
        metric_rows = [json.loads(line) for line in open(tmp_path / "metrics.jsonl")]
        selection = [r for r in metric_rows if r.get("phase") == "sft_selection"]
        assert len(selection) == 1
        assert selection[0]["pipeline/sft_external_health_gate_selected"] == 1.0

        trainer_state_path = central_dir / "trainer_state.json"
        final_state = json.loads(trainer_state_path.read_text())
        final_state["step"] = 1
        trainer_state_path.write_text(json.dumps(final_state))
        with pytest.raises(RuntimeError, match="step mismatch"):
            pipeline_gpu._validate_completed_central(
                str(central_dir),
                state["stages"]["r01/sft"]["validated_external_evidence"],
                expected_steps=100,
                expected_runtime_contract=state["stages"]["r01/sft"]
                ["validated_external_evidence"]["central_runtime_contract"],
            )
        final_state["step"] = 100
        trainer_state_path.write_text(json.dumps(final_state))

        (central_dir / "model.safetensors").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="directory file manifest mismatch"):
            pipeline_gpu._validate_completed_central(
                str(central_dir),
                state["stages"]["r01/sft"]["validated_external_evidence"],
                expected_steps=100,
                expected_runtime_contract=state["stages"]["r01/sft"]
                ["validated_external_evidence"]["central_runtime_contract"],
            )
        (central_dir / "model.safetensors").write_bytes(b"central")

        tampered = copy.deepcopy(selection_manifest)
        tampered["candidates"][-1]["passed"] = False
        selection_path.write_text(json.dumps(tampered))
        # Refresh the selection artifact membership manifest so this tests the
        # local all-four rebuild, not merely the outer file hash.
        selection_entry, _ = artifact_entry(
            selection_root, "entity/project/selection:v0", "selection-digest",
            "selection-download.json", manifest_path=str(selection_path.resolve()))
        evidence_index["selection"] = selection_entry
        evidence_path.write_text(json.dumps(evidence_index))
        with pytest.raises(ValueError, match="local all-four raw-gate rebuild"):
            pipeline_gpu._validate_external_sft_evidence(
                args, str(accepted_copy))

        real_builder = select_sft_checkpoint.build_selection_manifest_from_bundles

        def builder_with_unattributed_gate_source(bundle_dirs, gate_sources):
            rebuilt = real_builder(bundle_dirs, gate_sources)
            rebuilt["candidates"][0]["gate_source"] = \
                "entity/project/unattributed:v0#digest=not-the-download"
            rebuilt["candidates"][0]["gate_artifact_identity"] = {
                "artifact_ref": "entity/project/unattributed:v0",
                "artifact_digest": "not-the-download",
            }
            return rebuilt

        monkeypatch.setattr(
            select_sft_checkpoint, "build_selection_manifest_from_bundles",
            builder_with_unattributed_gate_source)
        with pytest.raises(ValueError, match="validated artifact download manifest"):
            pipeline_gpu._validate_external_sft_evidence(
                args, str(accepted_copy))


class TestLambdaPrecedence:
    def test_single_round_lambda_wins(self, tmp_path):
        args = pipeline_args(tmp_path, "--lambda-novelty", "0.3",
                             "--lambda-schedule", "0.75,0.5")
        assert pipeline_gpu.parse_lambda_schedule(args) == [0.3]

    def test_multi_round_default_schedule(self, tmp_path):
        args = pipeline_args(tmp_path, "--rounds", "3")
        assert pipeline_gpu.parse_lambda_schedule(args) == [0.75, 0.5, 0.35]

    def test_multi_round_explicit_lambda_is_constant(self, tmp_path):
        args = pipeline_args(tmp_path, "--rounds", "3", "--lambda-novelty", "0.4")
        assert pipeline_gpu.parse_lambda_schedule(args) == [0.4, 0.4, 0.4]

    def test_multi_round_inconsistent_pair_errors(self, tmp_path):
        args = pipeline_args(tmp_path, "--rounds", "2", "--lambda-novelty", "0.4",
                             "--lambda-schedule", "0.75,0.5")
        with pytest.raises(ValueError, match="disagree"):
            pipeline_gpu.parse_lambda_schedule(args)

    def test_multi_round_consistent_pair_ok(self, tmp_path):
        args = pipeline_args(tmp_path, "--rounds", "2", "--lambda-novelty", "0.5",
                             "--lambda-schedule", "0.5,0.5")
        assert pipeline_gpu.parse_lambda_schedule(args) == [0.5, 0.5]

    def test_defaults_without_flags(self, tmp_path):
        args = pipeline_args(tmp_path)
        assert pipeline_gpu.parse_lambda_schedule(args) == [0.5]


# ------------------------------------------------------------- probe honesty ----

def grpo_args(*extra):
    return grpo_gpu.build_arg_parser().parse_args(["--output-dir", "/tmp/x", *extra])


class TestProbeSizing:
    def test_own_pool_auto_raised(self):
        args = grpo_args("--vllm-gpus", "1")
        grpo_gpu.enforce_probe_serving_len(args, None)
        assert args.vllm_max_model_len == 2048 + grpo_gpu.PROBE_MAX_TOKENS

    def test_attach_mode_fails_fast(self):
        args = grpo_args("--endpoints", "http://x")
        with pytest.raises(ValueError, match="probe"):
            grpo_gpu.enforce_probe_serving_len(args, None)

    def test_provided_managed_pool_too_small_fails(self):
        args = grpo_args()
        pool = VLLMPool(model_dir="m", gpu_ids=[0], served_model_name="m",
                        max_model_len=18432)
        with pytest.raises(ValueError, match="max_model_len"):
            grpo_gpu.enforce_probe_serving_len(args, pool)

    def test_provided_managed_pool_big_enough_ok(self):
        args = grpo_args()
        pool = VLLMPool(model_dir="m", gpu_ids=[0], served_model_name="m",
                        max_model_len=36864)
        grpo_gpu.enforce_probe_serving_len(args, pool)

    def test_probe_off_leaves_sizing_alone(self):
        args = grpo_args("--probe-every", "0", "--endpoints", "http://x")
        grpo_gpu.enforce_probe_serving_len(args, None)
        assert args.vllm_max_model_len == 18432

    def test_pipeline_attach_mode_probe_guard(self, tmp_path):
        args = pipeline_args(tmp_path, "--endpoints", "http://x")
        argv_ok = pipeline_args(tmp_path, "--endpoints", "http://x",
                                "--vllm-max-model-len", "36864")
        assert int(args.vllm_max_model_len) < 2048 + grpo_gpu.PROBE_MAX_TOKENS
        assert int(argv_ok.vllm_max_model_len) >= 2048 + grpo_gpu.PROBE_MAX_TOKENS


class TestProbeHonesty:
    class _Pool:
        def __init__(self, factory):
            self._factory = factory
            self.last_kwargs = None

        def generate(self, prompts, n, **kwargs):
            self.last_kwargs = kwargs
            return [self._factory(int(n)) for _ in prompts]

    def _args(self):
        return SimpleNamespace(gen_timeout=10.0, gen_concurrency=1)

    def test_all_error_probe_logs_no_average(self):
        pool = self._Pool(
            lambda n: [Completion(text="", finish_reason="error", error="x")] * n)
        metrics = grpo_gpu._run_probe(pool, [("p", "4")], self._args())
        assert "eval/AIME24_avg" not in metrics
        assert "eval/AIME24_avg_at_4" not in metrics
        assert metrics["eval/probe_error_rows"] == float(grpo_gpu.PROBE_NUM_SAMPLES)
        assert metrics["eval/probe_protocol_complete"] == 0.0
        assert metrics["eval/probe_expected_rows"] == 120.0

    def test_partial_error_probe_never_averages_an_incomplete_protocol(self):
        def factory(n):
            comps = [Completion(text="thus \\boxed{4}.", finish_reason="stop")] * (n - 1)
            return comps + [Completion(text="", finish_reason="error", error="x")]

        metrics = grpo_gpu._run_probe(self._Pool(factory), [("p", "4")], self._args())
        assert "eval/AIME24_avg" not in metrics
        assert "eval/AIME24_avg_at_4" not in metrics
        assert metrics["eval/probe_error_rows"] == 1.0
        assert metrics["eval/probe_graded_rows"] == 3.0
        assert metrics["eval/probe_complete_problem_count"] == 0.0
        assert metrics["eval/probe_protocol_complete"] == 0.0

    def test_complete_30_by_4_probe_logs_only_nonheadline_avg_at_4(self):
        pool = self._Pool(
            lambda n: [Completion(text="thus \\boxed{4}.", finish_reason="stop")] * n)
        metrics = grpo_gpu._run_probe(
            pool, [(f"p{i}", "4") for i in range(30)], self._args())
        assert metrics["eval/AIME24_avg_at_4"] == 1.0
        assert "eval/AIME24_avg" not in metrics
        assert metrics["eval/probe_graded_rows"] == 120.0
        assert metrics["eval/probe_complete_problem_count"] == 30.0
        assert metrics["eval/probe_protocol_complete"] == 1.0
        assert metrics["eval/probe_protocol"] == grpo_gpu.PROBE_PROTOCOL_ID
        assert metrics["eval/probe_headline_eligible"] == 0.0
        assert metrics["eval/probe_metric_scale"] == "fraction"
        assert metrics["eval/probe_sample_seed_rule"] == eval_gpu.SAMPLE_SEED_RULE
        assert len(metrics["eval/probe_prompt_matrix_sha256"]) == 64
        assert pool.last_kwargs["min_p"] == 0.0
        assert pool.last_kwargs["seed_base"] == 0

    def test_unexpected_finish_reason_makes_probe_incomplete(self):
        pool = self._Pool(
            lambda n: [Completion(
                text="thus \\boxed{4}.", finish_reason="cancelled") for _ in range(n)])
        metrics = grpo_gpu._run_probe(
            pool, [(f"p{i}", "4") for i in range(30)], self._args())
        assert metrics["eval/probe_protocol_complete"] == 0.0
        assert "eval/AIME24_avg_at_4" not in metrics

    def test_probe_average_empty_is_nan(self):
        assert math.isnan(grpo_gpu.probe_average([]))


# ------------------------------------------------------------ eval integrity ----

def rec(correct, *, pred="4", finish="stop", tokens=10):
    return {"pred": pred if finish != "error" else None,
            "correct": bool(correct), "finish_reason": finish,
            "completion_tokens": tokens}


class TestEvalErrorExclusion:
    def test_error_rows_excluded_and_counted(self):
        results = {0: {0: rec(True), 1: rec(False, finish="error"), 2: rec(False)}}
        per_problem, n_error, n_total = eval_gpu.build_per_problem(results)
        assert (n_error, n_total) == (1, 3)
        assert per_problem[0]["n"] == 2
        assert per_problem[0]["correct_flags"] == [True, False]

    def test_all_error_problem_scores_nothing(self):
        results = {0: {0: rec(False, finish="error"), 1: rec(False, finish="error")}}
        per_problem, n_error, n_total = eval_gpu.build_per_problem(results)
        assert per_problem[0]["n"] == 0 and n_error == 2
        assert eval_gpu.avg_at(per_problem, 1) is None
        assert eval_gpu.pass_at(per_problem, 1) is None

    @pytest.mark.parametrize("finish", [None, "", "cancelled", "content_filter"])
    def test_only_stop_and_length_are_complete(self, finish):
        results = {0: {0: rec(True, finish=finish), 1: rec(True, finish="length")}}
        per_problem, n_error, n_total = eval_gpu.build_per_problem(results)
        assert (n_error, n_total) == (1, 2)
        assert per_problem[0]["n"] == 1


class TestEvalExactSampleMatrix:
    def test_empty_expected_matrix_is_never_exact(self):
        status = eval_gpu.exact_sample_matrix_status(
            [], expected_problem_ids=set(), expected_samples=64, n_error=0)
        assert status["exact_sample_matrix"] is False

    def test_complete_matrix_is_exact(self):
        per_problem, n_error, _ = eval_gpu.build_per_problem({
            3: {0: rec(True), 1: rec(False)},
            7: {0: rec(False), 1: rec(True)},
        })
        status = eval_gpu.exact_sample_matrix_status(
            per_problem, expected_problem_ids={3, 7},
            expected_samples=2, n_error=n_error)
        assert status["exact_sample_matrix"] is True
        assert status["complete_problem_count"] == 2
        assert status["successful_sample_count_min"] == 2
        assert status["successful_sample_count_max"] == 2

    def test_one_error_is_never_exact_even_if_other_problem_is_complete(self):
        per_problem, n_error, _ = eval_gpu.build_per_problem({
            3: {0: rec(True), 1: rec(False, finish="error")},
            7: {0: rec(False), 1: rec(True)},
        })
        status = eval_gpu.exact_sample_matrix_status(
            per_problem, expected_problem_ids={3, 7},
            expected_samples=2, n_error=n_error)
        assert status["exact_sample_matrix"] is False
        assert status["generation_error_count"] == 1
        assert status["wrong_sample_counts"] == {"3": 1}

    def test_missing_and_unexpected_problem_ids_fail_closed(self):
        per_problem = [{
            "problem_idx": 8, "n": 2, "correct_flags": [True, True],
            "preds": ["4", "4"], "clipped_flags": [False, False],
            "valid_flags": [True, True], "tokens": [1, 1],
        }]
        status = eval_gpu.exact_sample_matrix_status(
            per_problem, expected_problem_ids={3, 7},
            expected_samples=2, n_error=0)
        assert status["exact_sample_matrix"] is False
        assert status["missing_problem_ids"] == [3, 7]
        assert status["unexpected_problem_ids"] == [8]

    def test_benchmark_rows_hash_is_order_and_content_sensitive(self):
        rows = [
            {"problem_idx": 0, "problem": "p0", "ground_truth": "1"},
            {"problem_idx": 1, "problem": "p1", "ground_truth": "2"},
        ]
        digest = eval_gpu.benchmark_rows_hash(rows)
        assert len(digest) == 64
        assert digest != eval_gpu.benchmark_rows_hash(list(reversed(rows)))
        changed = [dict(r) for r in rows]
        changed[1]["ground_truth"] = "3"
        assert digest != eval_gpu.benchmark_rows_hash(changed)

    def publication_args(self, **overrides):
        values = {
            "require_publication_protocol": True,
            "protocol": "P1", "benchmark": "AIME24", "max_problems": None,
            "model_family": "qwen3-1.7b", "server_max_model_len": 36864,
            "model": "Qwen/Qwen3-1.7B", "model_source_name": "Qwen/Qwen3-1.7B",
            "served_model_name": None, "method_name": "Base",
            "model_revision": eval_gpu.MODEL_REVISIONS["qwen3-1.7b"],
            "expected_vllm_version": "0.24.0",
            "container_image_identity": (
                eval_gpu.PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY),
            "model_artifact_ref": None, "model_artifact_digest": None,
            "training_campaign_binding_sha256": None,
            "code_artifact_ref": "entity/project/code:v1",
            "code_artifact_digest": "digest",
            "run_name": "unit-test",
            "_evaluation_execution_environment": (
                receipt_fixtures.evaluation_environment(
                    run_name="unit-test",
                    code_ref="entity/project/code:v1",
                    code_digest="digest",
                )
            ),
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "thinking_mode": "auto",
            "max_completion_tokens": 32768,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_publication_protocol_accepts_full_default_p1(self, monkeypatch):
        monkeypatch.setattr(
            eval_gpu, "runtime_stack_versions",
            lambda: dict(eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS))
        eval_gpu.validate_publication_protocol(
            self.publication_args(), benchmark_default_samples=64,
            resolved_samples=64)

    @pytest.mark.parametrize(
        "overrides, resolved_samples",
        [
            ({"max_problems": 5}, 64),
            ({}, 4),
            ({"temperature": 1.0}, 64),
            ({"min_p": 0.05}, 64),
            ({"max_completion_tokens": 16384}, 64),
            ({"expected_vllm_version": "unknown"}, 64),
            ({"container_image_identity": "vllm/vllm-openai:v0.24.0"}, 64),
        ],
    )
    def test_publication_protocol_rejects_debug_or_decoding_drift(
            self, overrides, resolved_samples):
        with pytest.raises(SystemExit, match="publication protocol drift"):
            eval_gpu.validate_publication_protocol(
                self.publication_args(**overrides),
                benchmark_default_samples=64,
                resolved_samples=resolved_samples)

    def test_publication_protocol_accepts_frozen_p2_budget(self, monkeypatch):
        monkeypatch.setattr(
            eval_gpu, "runtime_stack_versions",
            lambda: dict(eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS))
        eval_gpu.validate_publication_protocol(
            self.publication_args(protocol="P2", max_completion_tokens=38912),
            benchmark_default_samples=64, resolved_samples=64)

    def test_p2_manifest_is_explicitly_secondary_not_headline(self, monkeypatch):
        monkeypatch.setattr(
            eval_gpu, "runtime_stack_versions",
            lambda: dict(eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS))
        monkeypatch.setattr(
            eval_gpu, "executing_code_hashes", lambda: {
                path: "a" * 64 for path in eval_gpu.EXECUTING_CODE_FILES
            })
        args = self.publication_args(
            protocol="P2", max_completion_tokens=38912,
            server_max_model_len=40960,
        )
        manifest = eval_gpu.build_evaluation_manifest(
            args,
            rows=[{"problem_idx": 0, "problem": "p", "ground_truth": "1"}],
            rows_sha256="b" * 64,
            prompt_template_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
            model_config_signature_sha256="e" * 64,
            n_samples=64,
        )
        assert manifest["protocol"] == "P2"
        assert manifest["headline_eligible_requested"] is False

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({
                "model": "/workspace/model",
                "model_artifact_ref": "entity/project/fake-base:v1",
                "model_artifact_digest": "fake-digest",
            }, "Base model=.*canonical registry|Base cannot be supplied"),
            ({"served_model_name": "attacker/model"}, "Base served_model_name"),
            ({"method_name": "arbitrary-untrained"}, "unsupported publication method"),
            ({"training_campaign_binding_sha256": "a" * 64},
             "Base cannot carry a training campaign binding"),
        ],
    )
    def test_publication_protocol_rejects_base_alias_or_untrained_label(
        self, monkeypatch, overrides, message,
    ):
        monkeypatch.setattr(
            eval_gpu, "runtime_stack_versions",
            lambda: dict(eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS))
        with pytest.raises(SystemExit, match=message):
            eval_gpu.validate_publication_protocol(
                self.publication_args(**overrides),
                benchmark_default_samples=64,
                resolved_samples=64,
            )

    def test_p2_rejects_non_aime_benchmark(self):
        with pytest.raises(SystemExit, match="restricted"):
            eval_gpu.validate_publication_protocol(
                self.publication_args(
                    protocol="P2", benchmark="MATH500",
                    max_completion_tokens=38912),
                benchmark_default_samples=32, resolved_samples=32)

    def test_publication_rejects_self_attested_vllm_025(self, monkeypatch):
        drifted = dict(eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS)
        drifted["vllm"] = "0.25.0"
        monkeypatch.setattr(
            eval_gpu, "runtime_stack_versions", lambda: drifted)
        with pytest.raises(SystemExit, match="frozen publication version"):
            eval_gpu.validate_publication_protocol(
                self.publication_args(expected_vllm_version="0.25.0"),
                benchmark_default_samples=64, resolved_samples=64)

    def test_publication_rejects_non_vllm_runtime_drift(self, monkeypatch):
        drifted = dict(eval_gpu.PUBLICATION_RUNTIME_STACK_VERSIONS)
        drifted["datasets"] = "5.0.1"
        monkeypatch.setattr(
            eval_gpu, "runtime_stack_versions", lambda: drifted)
        with pytest.raises(SystemExit, match="runtime_stack_versions"):
            eval_gpu.validate_publication_protocol(
                self.publication_args(), benchmark_default_samples=64,
                resolved_samples=64)

    def test_minerva_sample_count_is_64_for_every_model(self):
        assert eval_gpu.benchmark_sample_count("Minerva-Math", "qwen3-1.7b") == 64
        assert eval_gpu.benchmark_sample_count("Minerva-Math", "qwen3-4b") == 64
        assert eval_gpu.benchmark_sample_count("Minerva-Math", "ministral-3-3b") == 64

    def test_artifact_identity_takes_precedence_over_default_model(self):
        assert eval_gpu.infer_model_family(
            None, "entity/project/trained-qwen3-4b:v1", "Qwen/Qwen3-1.7B",
        ) == "qwen3-4b"

    def test_loaded_benchmark_requires_count_and_frozen_hash(self, monkeypatch):
        rows = [{"problem_idx": 0, "problem": "p", "ground_truth": "4"}]
        spec = copy.deepcopy(eval_gpu.BENCHMARKS["AIME24"])
        spec["canonical_count"] = 1
        spec["canonical_rows_sha256"] = eval_gpu.benchmark_rows_hash(rows)
        monkeypatch.setitem(eval_gpu.BENCHMARKS, "AIME24", spec)
        assert eval_gpu.validate_loaded_benchmark(
            "AIME24", rows, publication=True) == spec["canonical_rows_sha256"]
        changed = [dict(rows[0], ground_truth="5")]
        with pytest.raises(SystemExit, match="row hash mismatch"):
            eval_gpu.validate_loaded_benchmark("AIME24", changed, publication=True)


class TestEvalManifestAndCheckpointBinding:
    def test_local_checkpoint_hash_reads_bytes(self, tmp_path):
        left, right = tmp_path / "left", tmp_path / "right"
        left.mkdir(); right.mkdir()
        (left / "model.safetensors").write_bytes(b"AAAA")
        (right / "model.safetensors").write_bytes(b"BBBB")
        assert eval_gpu.model_hash(str(left)) != eval_gpu.model_hash(str(right))

    def test_model_family_is_cross_checked_against_config(self, tmp_path):
        model = tmp_path / "model"
        model.mkdir()
        config = dict(eval_gpu.MODEL_CONFIG_SIGNATURES["qwen3-4b"])
        (model / "config.json").write_text(json.dumps(config))
        digest = eval_gpu.validate_model_family_config(
            str(model), "qwen3-4b")
        assert len(digest) == 64
        with pytest.raises(SystemExit, match="contradicts model_family"):
            eval_gpu.validate_model_family_config(
                str(model), "qwen3-1.7b")

    def test_manifest_mismatch_blocks_resume_and_removes_stale_aggregate(self, tmp_path):
        manifest_path = tmp_path / "evaluation_manifest.json"
        aggregate_path = tmp_path / "aggregate_metrics.json"
        rollouts_path = tmp_path / "rollouts.jsonl"
        manifest = {"model": "one", "protocol": "P1"}
        digest = eval_gpu.prepare_evaluation_manifest(
            manifest_path, aggregate_path, rollouts_path, manifest, resume=False)
        rollouts_path.write_text("{}\n")
        aggregate_path.write_text("stale\n")
        with pytest.raises(SystemExit, match="manifest mismatch"):
            eval_gpu.prepare_evaluation_manifest(
                manifest_path, aggregate_path, rollouts_path,
                {"model": "two", "protocol": "P1"}, resume=True)
        assert not aggregate_path.exists()
        assert digest == eval_gpu.payload_sha256(manifest)

    def test_provenance_free_resume_is_rejected(self, tmp_path):
        rollouts = tmp_path / "rollouts.jsonl"
        rollouts.write_text("{}\n")
        with pytest.raises(SystemExit, match="provenance-free resume"):
            eval_gpu.prepare_evaluation_manifest(
                tmp_path / "evaluation_manifest.json",
                tmp_path / "aggregate_metrics.json", rollouts,
                {"protocol": "P1"}, resume=True)

    def test_resume_recomputes_correctness_and_rejects_wrong_manifest(self, tmp_path):
        path = tmp_path / "rollouts.jsonl"
        digest = "a" * 64
        record = {
            "problem_idx": 0, "sample_idx": 0,
            "evaluation_manifest_sha256": digest,
            "gold": "4", "pred": "wrong", "correct": False,
            "finish_reason": "stop", "text": "answer \\boxed{4}",
        }
        path.write_text(json.dumps(record) + "\n")
        done, results = eval_gpu.load_resume_rows(
            path, manifest_sha256=digest, expected_problem_ids={0},
            expected_samples=1, ground_truth_by_problem={0: "4"})
        assert done == {(0, 0)}
        assert results[0][0]["correct"] is True
        with pytest.raises(SystemExit, match="wrong manifest digest"):
            eval_gpu.load_resume_rows(
                path, manifest_sha256="b" * 64, expected_problem_ids={0},
                expected_samples=1, ground_truth_by_problem={0: "4"})

    def test_resume_discards_only_an_unterminated_partial_tail(self, tmp_path):
        path = tmp_path / "rollouts.jsonl"
        digest = "a" * 64
        record = {
            "problem_idx": 0, "sample_idx": 0,
            "evaluation_manifest_sha256": digest,
            "gold": "4", "pred": "4", "correct": True,
            "finish_reason": "stop", "text": "\\boxed{4}",
        }
        prefix = (json.dumps(record) + "\n").encode()
        path.write_bytes(prefix + b'{"problem_idx": 0')
        done, _ = eval_gpu.load_resume_rows(
            path, manifest_sha256=digest, expected_problem_ids={0},
            expected_samples=2, ground_truth_by_problem={0: "4"})
        assert done == {(0, 0)}
        assert path.read_bytes() == prefix


class TestEvalKGuards:
    def _pp(self, flags_by_problem):
        return [{"problem_idx": i, "n": len(f), "correct_flags": f,
                 "preds": ["4" if x else "5" for x in f],
                 "clipped_flags": [False] * len(f),
                 "valid_flags": [True] * len(f), "tokens": [1] * len(f)}
                for i, f in enumerate(flags_by_problem)]

    def test_pass_at_skips_undersampled_problems(self):
        # Problem 1 has n=2 < k=4 and ZERO correct: pass_at_k would give 1.0.
        per_problem = self._pp([[False] * 8, [False, False]])
        assert eval_gpu.pass_at(per_problem, 4) == 0.0
        assert eval_gpu.pass_at(self._pp([[False, False]]), 4) is None

    def test_maj_at_skips_undersampled_and_uses_eligible_denominator(self):
        gt = {0: "4", 1: "4"}
        per_problem = self._pp([[True] * 4, [True]])  # problem 1 undersampled
        assert eval_gpu.maj_at(per_problem, gt, 4) == 1.0
        assert eval_gpu.maj_at(self._pp([[True]]), {0: "4"}, 4) is None

    def test_avg_at_unchanged_semantics(self):
        per_problem = self._pp([[True, False, True, True]])
        assert eval_gpu.avg_at(per_problem, 4) == 0.75
        assert eval_gpu.avg_at(per_problem, 8) is None


class TestEvalServerLen:
    def test_p2_too_small_aborts(self):
        args = SimpleNamespace(protocol="P2", max_completion_tokens=38912,
                               server_max_model_len=36864)
        with pytest.raises(SystemExit):
            eval_gpu.enforce_server_len(args)

    def test_p2_big_enough_ok(self):
        args = SimpleNamespace(protocol="P2", max_completion_tokens=38912,
                               server_max_model_len=40960)
        eval_gpu.enforce_server_len(args)

    def test_p1_default_sizing_ok(self):
        args = SimpleNamespace(protocol="P1", max_completion_tokens=32768,
                               server_max_model_len=36864)
        eval_gpu.enforce_server_len(args)

    def test_unknown_server_len_warns_not_raises(self):
        args = SimpleNamespace(protocol="P2", max_completion_tokens=38912,
                               server_max_model_len=None)
        eval_gpu.enforce_server_len(args)


# ------------------------------------------------------- resume config guard ----

class TestResumeConfigValidation:
    def test_matching_checkpoint_ok(self):
        grpo_gpu.check_resume_compat(
            {"phase": "scout", "model": "m", "config_tier": "contract"},
            phase="scout", model="m", config_tier="contract")

    @pytest.mark.parametrize("key,bad", [
        ("phase", "central"), ("model", "other-model"),
        ("config_tier", "gpu-max:completion_budget"),
    ])
    def test_mismatch_raises(self, key, bad):
        recorded = {"phase": "scout", "model": "m", "config_tier": "contract",
                    key: bad}
        with pytest.raises(RuntimeError, match="mismatch"):
            grpo_gpu.check_resume_compat(recorded, phase="scout", model="m",
                                         config_tier="contract")

    def test_legacy_checkpoint_without_fields_ok(self):
        grpo_gpu.check_resume_compat({"step": 25}, phase="scout", model="m",
                                     config_tier="contract")


# ------------------------------------------------------------- W&B step offset ----

class TestStageRunStepOffset:
    class _Run:
        def __init__(self):
            self.logged = []
            self.finished = False

        def log(self, data, step=None):
            self.logged.append((data, step))

        def finish(self):
            self.finished = True

    def test_offsets_explicit_steps_and_delegates(self):
        run = self._Run()
        wrapped = common.StageRunStepOffset(run, 30)
        wrapped.log({"a": 1}, step=5)
        wrapped.log({"b": 2})
        wrapped.finish()
        assert run.logged == [({"a": 1}, 35), ({"b": 2}, None)]
        assert run.finished


# --------------------------------------------------------------- pool crash ----

class TestPoolCrashPaths:
    def _pool(self, tmp_path, n_gpus=2):
        pool = VLLMPool(model_dir="m", gpu_ids=list(range(n_gpus)),
                        served_model_name="m", log_dir=str(tmp_path))
        pool.settle_seconds = 0.0
        return pool

    def test_partial_launch_cleans_up_and_raises(self, tmp_path, monkeypatch):
        pool = self._pool(tmp_path)
        monkeypatch.setattr(vllm_pool.os, "getpgid", no_pgid)
        monkeypatch.setattr(vllm_pool.os, "killpg", no_pgid)
        started = []

        def fake_launch_one(i):
            if i == 1:
                raise RuntimeError("port busy")
            proc = FakeProc()
            started.append(proc)
            pool.procs[i] = proc

        monkeypatch.setattr(pool, "_launch_one", fake_launch_one)
        with pytest.raises(RuntimeError, match="port busy"):
            pool.launch()
        assert pool.procs == [] and pool.log_files == []
        assert started[0].rc is not None, "started server must be killed, not orphaned"

    def test_kill_tolerates_none_and_dead_procs(self, tmp_path, monkeypatch):
        pool = self._pool(tmp_path)
        monkeypatch.setattr(vllm_pool.os, "getpgid", no_pgid)
        monkeypatch.setattr(vllm_pool.os, "killpg", no_pgid)
        dead = FakeProc()
        dead.rc = 1
        pool.procs = [None, dead]
        pool.log_files = [None, None]
        pool._kill()  # must not raise
        assert pool.procs == []

    def test_reload_retries_after_failed_launch(self, monkeypatch, tmp_path):
        pool = self._pool(tmp_path, n_gpus=1)
        monkeypatch.setattr(vllm_pool.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def fake_launch():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first launch flaked")

        monkeypatch.setattr(pool, "launch", fake_launch)
        monkeypatch.setattr(pool, "wait_healthy", lambda *a, **k: None)
        pool.reload("newdir")  # must not AttributeError on retry
        assert calls["n"] == 2 and pool.model_dir == "newdir"

    def test_timeout_strikes_accumulate_and_reset(self, monkeypatch):
        pool = VLLMPool.attach(["http://a", "http://b"], "m")
        monkeypatch.setattr(vllm_pool.time, "sleep", lambda s: None)

        def timeout_post(url, json=None, timeout=None):
            raise requests.exceptions.ReadTimeout("wedged")

        monkeypatch.setattr(vllm_pool.requests, "post", timeout_post)
        c = pool._post_one("p", 0, {"max_tokens": 4}, 1.0)
        assert c.finish_reason == "error"
        assert pool.timeout_strikes == [3, 3]  # 3 retry rounds x both servers

        good_body = {"choices": [{"text": "ok", "finish_reason": "stop"}],
                     "usage": {"completion_tokens": 1}}

        class GoodResp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return good_body

        seen = []

        def good_post(url, json=None, timeout=None):
            seen.append(url)
            return GoodResp()

        monkeypatch.setattr(vllm_pool.requests, "post", good_post)
        c = pool._post_one("p", 0, {"max_tokens": 4}, 1.0)
        assert c.finish_reason == "stop"
        # First target had strikes reset by the success.
        struck = 0 if seen[0].startswith("http://a") else 1
        assert pool.timeout_strikes[struck] == 0

    def test_generate_prefers_unstruck_servers(self, monkeypatch):
        pool = VLLMPool.attach(["http://a", "http://b"], "m")
        pool.timeout_strikes = [2, 0]
        assert pool._endpoint_order(0) == [1, 0]
        assert pool._endpoint_order(1) == [1, 0]
        pool.timeout_strikes = [0, 0]
        assert pool._endpoint_order(0) == [0, 1]

    def test_health_check_relaunches_hung_server(self, tmp_path, monkeypatch):
        pool = self._pool(tmp_path)
        monkeypatch.setattr(vllm_pool.os, "getpgid", no_pgid)
        monkeypatch.setattr(vllm_pool.os, "killpg", no_pgid)
        pool.procs = [FakeProc(), FakeProc()]
        pool.log_files = [None, None]
        pool.timeout_strikes = [0, 2]  # server 1 wedged (2 consecutive timeouts)
        relaunched = []
        monkeypatch.setattr(pool, "_launch_one", lambda i: relaunched.append(i))
        monkeypatch.setattr(pool, "wait_healthy", lambda *a, **k: None)
        assert pool.health_check() == 1
        assert relaunched == [1]
        assert pool.timeout_strikes == [0, 0]
        assert pool.procs[1].rc is not None, "hung server's process group must be killed"
