"""Fail-closed handoff tests for the C9 Scout/SFT/Central experiment."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from tmx_gpu import (
    c9_evidence, common, eval_gpu, filter_pool, grpo_gpu, pipeline_gpu,
)


_SHA = "a" * 64


def _sampling() -> dict:
    return {
        "num_prompts": c9_evidence.GATE_NUM_PROMPTS,
        "samples_per_prompt": c9_evidence.GATE_SAMPLES_PER_PROMPT,
        "selection_seed": c9_evidence.GATE_SELECTION_SEED,
        "generation_seed": c9_evidence.GATE_GENERATION_SEED,
        "temperature": c9_evidence.GATE_TEMPERATURE,
        "top_p": c9_evidence.GATE_TOP_P,
        "top_k": c9_evidence.GATE_TOP_K,
        "min_p": c9_evidence.GATE_MIN_P,
        "max_completion_tokens": c9_evidence.GATE_MAX_COMPLETION_TOKENS,
        "paired_seeds_across_prompt_modes": True,
    }


def _behavior_gate(*, profile: str = "canary", stage: str = "scout_step50"):
    cell = "C5"
    expected_step = {
        "scout_step50": 50,
        "scout_final": c9_evidence.SCOUT_STEPS,
        "sft_fixed_endpoint": c9_evidence.SFT_OPTIMIZER_STEPS,
    }[stage]
    if stage.startswith("scout"):
        runtime = {
            "phase": "scout",
            "expected_steps": expected_step,
            "lambda_novelty": 0.5,
            "c9_execution_profile": profile,
            "stabilization_cell": cell,
            "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
        }
    else:
        runtime = {
            "model_init": "Qwen/Qwen3-1.7B",
            "epochs": c9_evidence.SFT_EPOCHS,
            "grad_accum": 1,
        }
    accepted_rows = 0 if profile == "canary" else c9_evidence.ACCEPTED_ROWS
    accepted_sha = None if profile == "canary" else _SHA
    return c9_evidence.signed_payload(
        c9_evidence.BEHAVIOR_GATE_EVIDENCE_VERSION,
        {
            "c9_execution_profile": profile,
            "stabilization_cell": cell,
            "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
            "gate_stage": stage,
            "selection_uses_correctness": False,
            "panel_role": c9_evidence.GATE_PANEL_ROLE,
            "panel_held_out_from_scout_training":
                c9_evidence.GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING,
            "passed": True,
            "sampling": _sampling(),
            "thresholds": dict(c9_evidence.GATE_THRESHOLDS),
            "heldout_selection_sha256": _SHA,
            "heldout_selection_file_sha256": _SHA,
            "raw_rollout_count": (
                2 * c9_evidence.GATE_NUM_PROMPTS
                * c9_evidence.GATE_SAMPLES_PER_PROMPT),
            "raw_rollouts_file_sha256": _SHA,
            "aggregate_file_sha256": _SHA,
            "model_file_manifest": {
                "version": common.DIRECTORY_FILE_MANIFEST_VERSION},
            "model_runtime_contract": runtime,
            "model_step": expected_step,
            "accepted_library_digest": {
                "version": "semantic_sft_jsonl_digest_v1",
                "row_count": accepted_rows,
                "sha256": _SHA,
            },
            "accepted_file_sha256": accepted_sha,
            "code_artifact": {"file_manifest_sha256": _SHA},
        },
    )


def _resign(value: dict) -> dict:
    payload = dict(value)
    version = payload.pop("version")
    payload.pop("evidence_sha256")
    return c9_evidence.signed_payload(version, payload)


def test_canary_behavior_gate_binds_exact_panel_runtime_and_thresholds():
    gate = _behavior_gate()
    assert c9_evidence.validate_behavior_gate_evidence(
        gate, expected_profile="canary", expected_cell="C5",
        expected_stage="scout_step50") is gate

    drifted = dict(gate)
    drifted["thresholds"] = {
        **gate["thresholds"], "max_clipped_rate": 0.99}
    with pytest.raises(ValueError, match="threshold contract"):
        c9_evidence.validate_behavior_gate_evidence(_resign(drifted))

    drifted = dict(gate)
    drifted["model_runtime_contract"] = {
        **gate["model_runtime_contract"], "expected_steps": 49}
    with pytest.raises(ValueError, match="runtime mismatch"):
        c9_evidence.validate_behavior_gate_evidence(_resign(drifted))

    drifted = dict(gate)
    drifted["panel_held_out_from_scout_training"] = True
    with pytest.raises(ValueError, match="not held out from Scout training"):
        c9_evidence.validate_behavior_gate_evidence(_resign(drifted))


def test_canary_gate_cannot_claim_or_omit_the_wrong_library_binding():
    gate = _behavior_gate()
    drifted = dict(gate)
    drifted["accepted_file_sha256"] = _SHA
    with pytest.raises(ValueError, match="must not bind"):
        c9_evidence.validate_behavior_gate_evidence(_resign(drifted))

    full = _behavior_gate(profile="full", stage="scout_final")
    assert c9_evidence.validate_behavior_gate_evidence(
        full, expected_profile="full", expected_stage="scout_final") is full
    drifted = dict(full)
    drifted["accepted_library_digest"] = {
        **full["accepted_library_digest"], "row_count": 499}
    with pytest.raises(ValueError, match="exact 500-row"):
        c9_evidence.validate_behavior_gate_evidence(_resign(drifted))


def test_c9_signed_evidence_rejects_unsigned_mutation():
    gate = _behavior_gate()
    gate["passed"] = False
    with pytest.raises(ValueError, match="SHA256"):
        c9_evidence.validate_behavior_gate_evidence(gate)


def test_single_scout_filter_replay_reconstructs_exact_order_and_rejects_tamper(
        tmp_path, monkeypatch):
    """SingleScout gets the same independent raw->pool replay as MR-ME."""
    monkeypatch.setattr(c9_evidence, "SCOUT_TRAJECTORY_ROWS", 3)
    monkeypatch.setattr(c9_evidence, "ACCEPTED_ROWS", 2)
    source = tmp_path / "explorer_trajectories.jsonl"
    rows = []
    for index in range(3):
        rows.append({
            "trajectory_schema_version": (
                grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION),
            "problem_id": f"p{index}",
            "prompt_text": f"problem {index}",
            "completion_text": f"reasoning {index} \\boxed{{4}}",
            "ground_truth": "4",
            "is_correct": True,
            "valid_answer": True,
            "terminated": True,
            "clipped": False,
            "finish_reason": "stop",
            "completion_token_length": 4,
            "completion_token_length_true": 4,
            "completion_token_length_retokenized": 4,
            "correctness_reward": 1.0,
            "novelty_reward": 0.0,
            "blended_reward": 1.0,
            "step": 1,
        })
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    accepted = tmp_path / "accepted.jsonl"
    _, funnel = filter_pool.pool_trajectories(
        [str(source)], policy=filter_pool.C8_POLICY, max_examples=2,
        output=str(accepted), source_balance=False,
    )
    certificate = pipeline_gpu._replay_c9_single_scout_filter(
        SimpleNamespace(completion_budget=0),
        trajectory_paths=[str(source)], accepted_path=str(accepted),
        funnel=funnel,
    )
    assert certificate["source_row_count"] == 3
    assert certificate["accepted_row_count"] == 2
    assert certificate["source_file_sha256"] == common.file_sha256(str(source))

    serialized = accepted.read_text(encoding="utf-8").splitlines()
    accepted.write_text(
        "\n".join(reversed(serialized)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from deterministic"):
        pipeline_gpu._replay_c9_single_scout_filter(
            SimpleNamespace(completion_budget=0),
            trajectory_paths=[str(source)], accepted_path=str(accepted),
            funnel=funnel,
        )


def test_pipeline_v2_central_lineage_loads_and_resume_matches(
        tmp_path, monkeypatch):
    """The pipeline's current C9 marker must reach Central and survive resume.

    The nested evidence validator is exercised separately; this regression
    targets the loader's version dispatch and the checkpoint digest choice
    that previously hard-coded the obsolete v1 marker.
    """
    runtime = {"phase": "central", "expected_steps": 100}
    lineage = c9_evidence.signed_payload(
        c9_evidence.CENTRAL_LINEAGE_VERSION,
        {
            "lineage_contract_version": c9_evidence.CENTRAL_LINEAGE_VERSION,
            "scientific_variant": c9_evidence.C9_SCIENTIFIC_VARIANT,
            "pipeline_emitted": True,
            "central_runtime_contract": runtime,
        },
    )
    path = tmp_path / "c9_central_lineage.json"
    path.write_text(json.dumps(lineage), encoding="utf-8")
    args = SimpleNamespace(
        phase="central",
        external_lineage_json=str(path),
        scientific_variant=c9_evidence.C9_SCIENTIFIC_VARIANT,
    )
    validated = []

    def _validate(value, *, expected_runtime_contract=None):
        validated.append((value, expected_runtime_contract))
        return value

    monkeypatch.setattr(c9_evidence, "validate_central_lineage", _validate)
    loaded = grpo_gpu.load_external_lineage(
        args, expected_runtime_contract=runtime)
    assert loaded == lineage
    assert validated == [(lineage, runtime)]

    recorded = {
        "phase": "central",
        "model": "/workspace/sft-final",
        "config_tier": "contract",
        "rl_validity_mode": "c8",
        "optimizer_state_saved": True,
        "resolved_lr": 1e-6,
        "external_lineage": lineage,
        "external_lineage_sha256": lineage["evidence_sha256"],
        "runtime_contract": runtime,
    }
    grpo_gpu.check_resume_compat(
        recorded,
        phase="central",
        model="/workspace/sft-final",
        config_tier="contract",
        rl_validity_mode_current="c8",
        resolved_lr_current=1e-6,
        external_lineage_current=loaded,
        runtime_contract_current=runtime,
    )


def test_canary_pipeline_state_rejects_old_or_drifted_execution(
        tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    args = SimpleNamespace(output_dir=str(out), resume=True)
    expected = {"version": "contract", "contract_sha256": _SHA}
    monkeypatch.setattr(
        pipeline_gpu, "_resolved_c9_execution_contract", lambda _args: expected)

    (out / "pipeline_state.json").write_text(json.dumps({"stages": {}}))
    with pytest.raises(RuntimeError, match="execution contract differs"):
        pipeline_gpu._load_state(args)

    (out / "pipeline_state.json").write_text(json.dumps({
        "stages": {}, "c9_execution_contract": expected}))
    assert pipeline_gpu._load_state(args)["c9_execution_contract"] == expected

    changed = {**expected, "contract_sha256": "b" * 64}
    monkeypatch.setattr(
        pipeline_gpu, "_resolved_c9_execution_contract", lambda _args: changed)
    with pytest.raises(RuntimeError, match="execution contract differs"):
        pipeline_gpu._load_state(args)


def test_canary_behavior_gate_runs_before_scout_only_return():
    source = inspect.getsource(pipeline_gpu.run_round)
    assert source.index('== "canary"') < source.index(
        'if getattr(args, "scout_only", False)')
    assert "_validate_c9_canary_scout_endpoint" in source
    assert 'stage="scout_step50"' in source


def _probe_metric_row(*, global_step: int, phase: str, local_step: int) -> dict:
    return {
        "global_step": global_step,
        "phase": phase,
        "eval/probe_protocol": grpo_gpu.PROBE_PROTOCOL_ID,
        "eval/probe_headline_eligible": 0.0,
        "eval/probe_metric_scale": "fraction",
        "eval/probe_sample_seed_rule": grpo_gpu.PROBE_SAMPLE_SEED_RULE,
        "eval/probe_benchmark_revision": grpo_gpu.PROBE_BENCHMARK_REVISION,
        "eval/probe_benchmark_rows_sha256": (
            grpo_gpu.PROBE_BENCHMARK_ROWS_SHA256),
        "eval/probe_prompt_matrix_sha256": "a" * 64,
        "eval/probe_protocol_complete": 1.0,
        "eval/probe_graded_rows": 120.0,
        "eval/probe_expected_rows": 120.0,
        "eval/probe_complete_problem_count": 30.0,
        "eval/probe_expected_problem_count": 30.0,
        "eval/probe_samples_per_problem": 4.0,
        "eval/probe_max_completion_tokens": 32768.0,
        "eval/probe_temperature": 0.6,
        "eval/probe_top_p": 0.95,
        "eval/probe_top_k": 20.0,
        "eval/probe_min_p": 0.0,
        "eval/probe_seed_base": float(grpo_gpu.PROBE_SEED_BASE),
        "eval/probe_policy_step": local_step,
        "eval/probe_staleness_updates": 0,
        "eval/probe_fresh_weights": 1.0,
        "eval/AIME24_avg_at_4": 0.5,
    }


def test_full_c9_probe_history_requires_every_exact_fresh_boundary(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        _probe_metric_row(
            global_step=step, phase="scout_grpo", local_step=step)
        for step in range(25, 201, 25)
    ]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = SimpleNamespace(probe_every=25, metrics_jsonl=str(metrics))
    facts = pipeline_gpu._c9_probe_history_facts(
        args, phase="scout", local_steps=200, global_step_offset=0)
    assert facts["expected_global_steps"] == list(range(25, 201, 25))
    assert facts["row_count"] == 8
    assert facts["prompt_matrix_sha256"] == "a" * 64
    assert len(facts["canonical_rows_sha256"]) == 64

    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(RuntimeError, match="exactly one row"):
        pipeline_gpu._c9_probe_history_facts(
            args, phase="scout", local_steps=200, global_step_offset=0)


def test_probe_uses_publication_loader_prompt_renderer_and_grader(monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps({"messages": messages, "kwargs": kwargs}, sort_keys=True)

    monkeypatch.setattr(
        eval_gpu, "load_benchmark",
        lambda name: [{"problem": "1+1?", "ground_truth": "2"}]
        if name == "AIME24" else [])
    validated = []
    monkeypatch.setattr(
        eval_gpu, "validate_loaded_benchmark",
        lambda name, rows, publication: (
            validated.append((name, rows, publication))
            or grpo_gpu.PROBE_BENCHMARK_ROWS_SHA256
        ),
    )
    tokenizer = Tokenizer()
    probe = grpo_gpu._load_probe_examples(
        SimpleNamespace(probe_dataset="aime24"), tokenizer)
    assert probe == [(eval_gpu.render_prompt(tokenizer, "1+1?", "auto"), "2")]
    assert eval_gpu.grade("therefore \\boxed{2}", probe[0][1])[1] is True
    assert validated == [("AIME24", [
        {"problem": "1+1?", "ground_truth": "2"}], True)]


def test_probe_rejects_corrupted_canonical_rows_before_rendering(monkeypatch):
    monkeypatch.setattr(
        eval_gpu, "load_benchmark",
        lambda _name: [{
            "problem_idx": 0, "problem": "corrupted", "ground_truth": "0"}],
    )
    monkeypatch.setattr(
        eval_gpu,
        "validate_loaded_benchmark",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemExit("canonical AIME24 row hash mismatch")),
    )
    with pytest.raises(SystemExit, match="row hash mismatch"):
        grpo_gpu._load_probe_examples(
            SimpleNamespace(probe_dataset="aime24"), object())


def test_canonical_probe_never_falls_back_to_plain_text(monkeypatch):
    class BrokenTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            raise RuntimeError("missing canonical chat template")

    monkeypatch.setattr(
        eval_gpu, "load_benchmark",
        lambda name: [{"problem": "1+1?", "ground_truth": "2"}])
    monkeypatch.setattr(
        eval_gpu, "validate_loaded_benchmark",
        lambda *_args, **_kwargs: grpo_gpu.PROBE_BENCHMARK_ROWS_SHA256)
    with pytest.raises(RuntimeError, match="missing canonical chat template"):
        grpo_gpu._load_probe_examples(
            SimpleNamespace(probe_dataset="aime24"), BrokenTokenizer())


def test_c9_probe_failure_precedes_checkpoint_promotion():
    source = inspect.getsource(grpo_gpu.run_grpo)
    strict_failure = source.index("C9 required probe failed at step")
    deferred_save = source.index(
        "if checkpoint_due and defer_checkpoint_until_probe")
    assert strict_failure < deferred_save
    assert "checkpoint promotion is forbidden" in source


def test_final_artifact_upload_is_conditioned_on_pipeline_success():
    launcher = (
        Path(__file__).resolve().parents[1] / "launch" / "expdis_job.sh"
    ).read_text()
    assert 'export PIPELINE_EXIT_STATUS="$STATUS"' in launcher
    assert "pipeline_status == 0" in launcher
    assert "c9_publishable_final = None" in launcher
    assert "final_model = c9_publishable_final" in launcher


def test_publication_eval_launcher_requires_qwen_protocol_and_exact_matrix():
    launcher = (
        Path(__file__).resolve().parents[1] / "launch" / "base_eval_job.sh"
    ).read_text()
    assert "--require-exact-sample-matrix --require-publication-protocol" in launcher
    assert "ALLOW_PARTIAL_EVAL" in launcher
    assert math.isclose(eval_gpu.BENCHMARKS["AIME24"]["n"], 64)
