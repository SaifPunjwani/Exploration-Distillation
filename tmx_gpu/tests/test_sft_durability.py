"""SFT handoff durability, exact-resume, and W&B failure tests."""

from __future__ import annotations

import json
import inspect
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tmx_gpu import common, pipeline_gpu, sft_gpu


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def _completed_sft(tmp_path: Path):
    rows = [
        {"prompt_text": "p0", "completion_text": "c0"},
        {"prompt_text": "p1", "completion_text": "c1"},
    ]
    accepted = _write_jsonl(tmp_path / "accepted.jsonl", rows)
    model = tmp_path / "round_01" / "sft" / "actual_sft_final"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n")
    (model / "model.safetensors").write_bytes(b"model-bytes")
    runtime = {"version": "test-sft-runtime", "epochs": 2, "seed": 7}
    policy_manifest = common.directory_file_manifest(str(model))
    completed = {
        "version": "sft_completed_run_evidence_v1",
        "training_completed": True,
        "optimizer_steps_executed": 4,
        "micro_steps_executed": 4,
        "saved_optimizer_steps": [],
        "training_evidence_sha256": None,
    }
    completed["evidence_sha256"] = common.canonical_json_sha256(completed)
    identity = {
        "accepted_library_digest": common.semantic_sft_dataset_digest(rows),
        "accepted_file_sha256": common.file_sha256(accepted),
        "training_sequence_digest": common.sft_training_sequence_digest(
            rows, epochs=2, seed=7),
    }
    common.atomic_write_json(model / "sft_provenance.json", {
        **identity,
        "sft_runtime_contract": runtime,
        "optimizer_steps": 4,
        "micro_steps": 4,
        "completed_run_evidence": completed,
        "checkpoint_file_manifest": policy_manifest,
    })
    handoff = pipeline_gpu._validate_local_sft_handoff(
        str(model), accepted_path=accepted,
        expected_runtime_contract=runtime, epochs=2, seed=7)
    args = SimpleNamespace(
        output_dir=str(tmp_path), run_name="durable-sft", sft_epochs=2,
        seed=7, c9_execution_profile="", stabilization_cell="",
        scientific_variant="", expdis_stabilization_mode="",
        novelty_normalization="", _mrme_contract=None,
    )
    return args, accepted, model, runtime, handoff


class _File:
    def __init__(self, name: str):
        self.name = name


class _Logged:
    def __init__(self, names: list[str], *, fail_wait: bool = False):
        self.qualified_name = "entity/project/durable-sft:v3"
        self.name = "durable-sft:v3"
        self.digest = "remote-digest"
        self._names = names
        self.fail_wait = fail_wait
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        if self.fail_wait:
            raise RuntimeError("upload failed")
        return self

    def files(self):
        return [_File(name) for name in self._names]


class _Artifact:
    def __init__(self, name, type, metadata=None):  # noqa: A002
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.names: list[str] = []

    def add_dir(self, path, name=None):
        root = Path(path)
        prefix = f"{name}/" if name else ""
        self.names.extend(
            prefix + item.relative_to(root).as_posix()
            for item in sorted(root.rglob("*")) if item.is_file())

    def add_file(self, path, name=None):
        self.names.append(name or Path(path).name)


class _Run:
    def __init__(self, *, fail_wait: bool = False):
        self.fail_wait = fail_wait
        self.logged: list[_Logged] = []

    def log_artifact(self, artifact, aliases=None):
        logged = _Logged(list(artifact.names), fail_wait=self.fail_wait)
        self.logged.append(logged)
        return logged


def _install_fake_wandb(monkeypatch):
    module = SimpleNamespace(Artifact=_Artifact)
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return module


def test_training_sequence_digest_is_order_sensitive_but_semantic_is_not():
    rows = [
        {"prompt_text": "p0", "completion_text": "c0"},
        {"prompt_text": "p1", "completion_text": "c1"},
        {"prompt_text": "p2", "completion_text": "c2"},
    ]
    reversed_rows = list(reversed(rows))
    assert common.semantic_sft_dataset_digest(rows) == \
        common.semantic_sft_dataset_digest(reversed_rows)
    assert common.sft_training_sequence_digest(
        rows, epochs=2, seed=7) != common.sft_training_sequence_digest(
            reversed_rows, epochs=2, seed=7)
    assert common.sft_training_sequence_digest(
        rows, epochs=2, seed=7) == common.sft_training_sequence_digest(
            rows, epochs=2, seed=7)


@pytest.mark.parametrize("initial", [None, "online"])
def test_sft_offline_fallback_restores_process_wandb_mode(
        monkeypatch, initial):
    if initial is None:
        monkeypatch.delenv("WANDB_MODE", raising=False)
    else:
        monkeypatch.setenv("WANDB_MODE", initial)
    modes = []

    def init(**_kwargs):
        modes.append(os.environ.get("WANDB_MODE"))
        if len(modes) <= 3:
            raise RuntimeError("network down")
        return object()

    monkeypatch.setattr(common, "init_stage_wandb", init)
    monkeypatch.setattr(sft_gpu.time, "sleep", lambda _seconds: None)
    assert sft_gpu.init_sft_wandb_with_fallback({"name": "sft"}) is not None
    assert modes == [initial, initial, initial, "offline"]
    assert os.environ.get("WANDB_MODE") == initial


def test_sft_model_upload_waits_and_binds_complete_bytes(tmp_path, monkeypatch):
    args, _accepted, model, _runtime, handoff = _completed_sft(tmp_path)
    _install_fake_wandb(monkeypatch)
    run = _Run()
    binding = pipeline_gpu._persist_sft_model_artifact(
        run, args, model_dir=str(model), round_idx=1,
        stage_mode="trained_in_pipeline", handoff=handoff)
    assert run.logged[0].wait_calls == 1
    assert binding["durable_wandb_upload"] is True
    assert binding["source"] == {
        "artifact_ref": "entity/project/durable-sft:v3",
        "artifact_digest": "remote-digest",
    }
    assert binding["model_file_manifest"] == \
        common.directory_file_manifest(str(model))
    assert json.loads((tmp_path / "round_01" /
                       "sft_model_artifact.json").read_text()) == binding


def test_sft_model_upload_failure_never_marks_durable_state(tmp_path, monkeypatch):
    args, _accepted, model, _runtime, handoff = _completed_sft(tmp_path)
    _install_fake_wandb(monkeypatch)
    with pytest.raises(RuntimeError, match="upload failed"):
        pipeline_gpu._persist_sft_model_artifact(
            _Run(fail_wait=True), args, model_dir=str(model), round_idx=1,
            stage_mode="trained_in_pipeline", handoff=handoff)
    assert not (tmp_path / "round_01" / "sft_model_artifact.json").exists()
    assert not (tmp_path / "pipeline_state.json").exists()


def test_pipeline_orders_upload_state_barrier_before_central():
    source = inspect.getsource(pipeline_gpu.run_round)
    persist = source.index("_persist_sft_model_artifact(")
    stage_mark = source.index(
        '_mark_stage(args, state, round_idx, "sft"', persist)
    results_wait_barrier = source.index(
        '_push_results_artifact(wandb_run, args, f"r{round_idx}-sft")',
        stage_mark)
    central = source.index("# ---- Stage 5: Central GRPO", results_wait_barrier)
    assert persist < stage_mark < results_wait_barrier < central


def test_results_artifact_wait_is_synchronous_and_failure_is_fatal(
        tmp_path, monkeypatch):
    _install_fake_wandb(monkeypatch)
    (tmp_path / "pipeline_state.json").write_text('{"stages": {}}\n')
    args = SimpleNamespace(
        output_dir=str(tmp_path), run_name="results", seed=0,
        fresh_c8_sft_gate=False, fresh_c8_multiscout_component=False,
        c9_execution_profile="", stabilization_cell="",
        scientific_variant="", expdis_stabilization_mode="",
        novelty_normalization="", _mrme_contract=None,
    )
    run = _Run()
    pipeline_gpu._push_results_artifact(run, args, "sft")
    assert run.logged[0].wait_calls == 1
    with pytest.raises(RuntimeError, match="upload failed"):
        pipeline_gpu._push_results_artifact(
            _Run(fail_wait=True), args, "sft-failure")


def test_resume_restores_only_exact_digest_bound_sft_artifact(
        tmp_path, monkeypatch):
    args, accepted, model, runtime, handoff = _completed_sft(tmp_path)
    module = _install_fake_wandb(monkeypatch)
    binding = pipeline_gpu._persist_sft_model_artifact(
        _Run(), args, model_dir=str(model), round_idx=1,
        stage_mode="trained_in_pipeline", handoff=handoff)
    snapshot = tmp_path / "remote-model"
    shutil.copytree(model, snapshot)
    shutil.rmtree(model)

    class Remote:
        digest = "remote-digest"

        def download(self, root):
            shutil.copytree(snapshot, Path(root) / "model")

    module.Api = lambda: SimpleNamespace(artifact=lambda _ref: Remote())
    stage = {"final": str(model), "sft_model_artifact": binding}
    restored, restored_binding = pipeline_gpu._restore_or_validate_sft_model(
        args, stage, accepted_path=accepted,
        expected_runtime_contract=runtime, round_idx=1)
    assert restored == str(model)
    assert restored_binding == binding
    assert common.directory_file_manifest(restored) == \
        binding["model_file_manifest"]

    shutil.rmtree(model)

    class WrongRemote(Remote):
        digest = "wrong-digest"

    module.Api = lambda: SimpleNamespace(artifact=lambda _ref: WrongRemote())
    with pytest.raises(RuntimeError, match="artifact digest mismatch"):
        pipeline_gpu._restore_or_validate_sft_model(
            args, stage, accepted_path=accepted,
            expected_runtime_contract=runtime, round_idx=1)


def test_resume_rejects_order_or_local_model_drift(tmp_path, monkeypatch):
    args, accepted, model, runtime, handoff = _completed_sft(tmp_path)
    _install_fake_wandb(monkeypatch)
    binding = pipeline_gpu._persist_sft_model_artifact(
        _Run(), args, model_dir=str(model), round_idx=1,
        stage_mode="trained_in_pipeline", handoff=handoff)
    stage = {"final": str(model), "sft_model_artifact": binding}
    rows = [json.loads(line) for line in Path(accepted).read_text().splitlines()]
    _write_jsonl(Path(accepted), list(reversed(rows)))
    with pytest.raises(RuntimeError, match="input identity mismatch"):
        pipeline_gpu._restore_or_validate_sft_model(
            args, stage, accepted_path=accepted,
            expected_runtime_contract=runtime, round_idx=1)
