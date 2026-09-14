"""Signed C10 canary evidence, pipeline identity, and ordering tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tmx_gpu import c10_evidence, common, grpo_gpu, pipeline_gpu
from tmx_jax.data import training_dataset_identity
from tmx_jax.prompting import prompt_metadata


_SHA = "a" * 64


def _runtime() -> dict:
    return {
        "version": grpo_gpu.GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": "scout",
        "scientific_variant": c10_evidence.C10_SCIENTIFIC_VARIANT,
        "c10_execution_profile": "canary",
        "config_tier": (
            "gpu-max:expdis_stabilization_mode,mask_truncated,"
            "novelty_normalization,truncation_baseline_mode+"
            "scientific-variant:"
            "expdis_c10_survivor_consistent_overlong_filter"),
        "model_init": common.CANONICAL_TRAINING_MODEL_ID,
        "base_model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "expected_steps": 50,
        "resolved_learning_rate": 1e-6,
        "optimizer": "adamw",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "parameter_dtype": "float32",
        "rl_validity_mode": "c8",
        "completion_budget_flag": 0,
        "resolved_max_prompt_len": 2048,
        "resolved_max_completion_len": 16384,
        "resolved_vllm_max_model_len": 36864,
        "reward_scoring_text_scope": grpo_gpu.REWARD_SCORING_TEXT_SCOPE,
        "trajectory_record_schema_version":
            grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION,
        "soft_overlong_expected_len": 13107,
        "soft_overlong_cache_len": 3277,
        "loss_denominator_len": 16384,
        "loss_normalization": "dr_grpo_fixed_denominator",
        "advantage_standardization": False,
        "kl_beta": 0.0,
        "prompts_per_step": 4,
        "num_generations": 16,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "training_prompt": prompt_metadata(),
        "dynamic_sampling": True,
        "dynamic_max_attempts": 64,
        "mask_truncated": True,
        "clip_epsilon_low": 0.2,
        "clip_epsilon_high": 0.28,
        "ppo_mode": False,
        "updates_per_rollout": 1,
        "incorrect_novelty_scale": 0.0,
        "lambda_novelty": 0.5,
        "dataset": "dapo_math_17k",
        "training_dataset_identity":
            training_dataset_identity("dapo_math_17k"),
        "max_train_examples": 20000,
        "seed": 0,
        "round": 1,
        "generation_failure_policy": {
            "owner": "c10_preregistered_generation_error_row_rate",
            "legacy_consecutive_failure_limit_enforced": False,
            "maximum_generation_error_row_rate": 0.05,
            "excess_classification": "transport_invalid",
            "dynamic_sampling_attempt_limit": 64,
        },
        "generation_timeout_seconds": 3600.0,
        "generation_concurrency_per_server": 8,
        "chunk_tokens": 256,
        "execution_environment": {"validated": True},
        "survivor_overlong_filter_contract": {
            "version": grpo_gpu.C10_OVERLONG_FILTER_CONTRACT_VERSION,
            "c10_execution_profile": "canary",
            "truncation_baseline_mode": "unmasked_survivors",
            "dynamic_sampling_eligibility_population": "unmasked_survivors",
            "minimum_dynamic_sampling_survivors": 2,
            "masked_row_advantage": 0.0,
            "masked_rows_in_policy_forward": False,
            "masked_rows_in_policy_backward": False,
            "loss_denominator_population": "full_sampled_rows",
            "rnd_update_population": "all_selected_raw_rows",
            "novelty_normalization_population": "all_selected_raw_scores",
        },
        "scout_novelty_contract": {
            "novelty_layers": "7,14,21",
            "rnd_learning_rate": 1e-4,
            "rnd_hidden_dim": 512,
            "novelty_feature_max_length": 16384,
            "incorrect_novelty_scale": 0.0,
            "rnd_lifecycle": grpo_gpu.C9_RND_LIFECYCLE,
            "rnd_scoring_predictor_state": "frozen_for_entire_learner_step",
            "novelty_normalization": "selected_batch_zscore",
            "rnd_update_scope": "final_selected_learner_rows",
            "rnd_updates_per_completed_step": 1,
            "novelty_normalization_population": "all_selected_raw_scores",
            "novelty_gate_order": "normalize_then_correct_only_gate",
            "novelty_normalization_epsilon":
                grpo_gpu.C9_NORMALIZATION_EPSILON,
            "dynamic_sampling_eligibility_novelty": "raw_correct_gated",
            "stabilization_cell": "",
        },
    }


def _evidence() -> dict:
    runtime = _runtime()
    sampling = {
        "num_prompts": c10_evidence.GATE_NUM_PROMPTS,
        "samples_per_prompt": c10_evidence.GATE_SAMPLES_PER_PROMPT,
        "selection_seed": c10_evidence.GATE_SELECTION_SEED,
        "generation_seed": c10_evidence.GATE_GENERATION_SEED,
        "temperature": c10_evidence.GATE_TEMPERATURE,
        "top_p": c10_evidence.GATE_TOP_P,
        "top_k": c10_evidence.GATE_TOP_K,
        "min_p": c10_evidence.GATE_MIN_P,
        "max_completion_tokens": c10_evidence.GATE_MAX_COMPLETION_TOKENS,
        "paired_seeds_across_prompt_modes": True,
    }
    return c10_evidence.signed_payload({
        "c10_execution_profile": "canary",
        "scientific_variant": c10_evidence.C10_SCIENTIFIC_VARIANT,
        "gate_stage": "scout_step50",
        "selection_uses_correctness": False,
        "panel_role": c10_evidence.GATE_PANEL_ROLE,
        "panel_held_out_from_scout_training":
            c10_evidence.GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING,
        "passed": True,
        "gate_outcome": "passed",
        "sampling": sampling,
        "thresholds": dict(c10_evidence.GATE_THRESHOLDS),
        "heldout_selection_sha256":
            c10_evidence.GATE_HELDOUT_SELECTION_SHA256,
        "heldout_selection_file_sha256":
            c10_evidence.GATE_HELDOUT_SELECTION_FILE_SHA256,
        "raw_rollout_count": (
            2 * c10_evidence.GATE_NUM_PROMPTS
            * c10_evidence.GATE_SAMPLES_PER_PROMPT),
        "raw_rollouts_file_sha256": _SHA,
        "aggregate_file_sha256": _SHA,
        "endpoint_replay_evidence_sha256": _SHA,
        "model_file_manifest": {
            "version": common.DIRECTORY_FILE_MANIFEST_VERSION},
        "model_runtime_contract": runtime,
        "model_runtime_contract_sha256":
            common.canonical_json_sha256(runtime),
        "model_step": 50,
        "accepted_library_digest": {
            "version": "c10_no_sft_library_v1",
            "row_count": 0,
            "sha256": _SHA,
        },
        "accepted_file_sha256": None,
        "code_artifact": {"file_manifest_sha256": _SHA},
        "preregistration_file": c10_evidence.C10_PREREGISTRATION_FILE,
        "preregistration_sha256": c10_evidence.C10_PREREGISTRATION_SHA256,
    })


def _resign(value: dict) -> dict:
    payload = dict(value)
    payload.pop("version")
    payload.pop("evidence_sha256")
    return c10_evidence.signed_payload(payload)


def _lease_holder(char: str = "b") -> dict:
    return {
        "orchestrator_binding_sha256": char * 64,
        "receipt_artifact_ref": f"entity/project/receipt-{char}:v1",
        "receipt_artifact_digest": f"receipt-digest-{char}",
    }


def _active_lease_claim(char: str = "b") -> dict:
    return c10_evidence.signed_lease_claim({
        "run_name": c10_evidence.C10_CANONICAL_RUN_NAME,
        "preregistration_sha256":
            c10_evidence.C10_PREREGISTRATION_SHA256,
        "lease_state": "active",
        "holder": _lease_holder(char),
    })


def _lease_binding(version: int = 0, char: str = "b") -> dict:
    return c10_evidence.build_exclusive_lease_binding(
        claim=_active_lease_claim(char),
        artifact_ref=(
            f"entity/project/{c10_evidence.C10_EXCLUSIVE_LEASE_ARTIFACT}"
            f":v{version}"),
        artifact_digest=f"lease-digest-{version}",
    )


def _lease_record(binding: dict) -> dict:
    return {
        "artifact_ref": binding["source"]["artifact_ref"],
        "artifact_digest": binding["source"]["artifact_digest"],
        "claim": binding["claim"],
    }


def test_c10_preregistration_bytes_are_bound(tmp_path):
    source = Path(c10_evidence.__file__).resolve().parents[1] / \
        c10_evidence.C10_PREREGISTRATION_FILE
    target = tmp_path / c10_evidence.C10_PREREGISTRATION_FILE
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    assert c10_evidence.validate_preregistration_file(tmp_path) == target
    target.write_text(target.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        c10_evidence.validate_preregistration_file(tmp_path)


def test_c10_signed_gate_binds_runtime_preregistration_and_panel(monkeypatch):
    monkeypatch.setattr(
        grpo_gpu, "require_canonical_training_execution_environment_contract",
        lambda runtime: runtime["execution_environment"])
    evidence = _evidence()
    assert c10_evidence.validate_behavior_gate_evidence(evidence) is evidence

    drifted = dict(evidence)
    drifted["preregistration_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="identity or endpoint"):
        c10_evidence.validate_behavior_gate_evidence(_resign(drifted))

    drifted = dict(evidence)
    runtime = dict(evidence["model_runtime_contract"])
    nested = dict(runtime["survivor_overlong_filter_contract"])
    nested["dynamic_sampling_eligibility_population"] = "all_rows"
    runtime["survivor_overlong_filter_contract"] = nested
    drifted["model_runtime_contract"] = runtime
    drifted["model_runtime_contract_sha256"] = \
        common.canonical_json_sha256(runtime)
    with pytest.raises(ValueError, match="runtime contract drift"):
        c10_evidence.validate_behavior_gate_evidence(_resign(drifted))


def test_c10_unsigned_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(
        grpo_gpu, "require_canonical_training_execution_environment_contract",
        lambda runtime: runtime["execution_environment"])
    evidence = _evidence()
    evidence["passed"] = False
    with pytest.raises(ValueError, match="SHA256"):
        c10_evidence.validate_behavior_gate_evidence(evidence)


def test_c10_gate_outcome_separates_transport_from_terminal_semantics():
    metrics = {
        mode: {"generation_error_rate": 0.0}
        for mode in pipeline_gpu.sft_gate_gpu.PROMPT_MODES
    }
    assert c10_evidence.classify_gate_outcome(
        metrics, {"passed": True}) == "passed"
    assert c10_evidence.classify_gate_outcome(
        metrics, {"passed": False}) == "semantic_failed"
    metrics[pipeline_gpu.sft_gate_gpu.PROMPT_MODES[0]][
        "generation_error_rate"] = 0.051
    assert c10_evidence.classify_gate_outcome(
        metrics, {"passed": False}) == "transport_invalid"


def test_all_error_training_transport_accepts_empty_candidate_file(monkeypatch):
    monkeypatch.setattr(
        grpo_gpu, "require_canonical_training_execution_environment_contract",
        lambda runtime: runtime.get("execution_environment"))
    attempts = [{
        "attempt": attempt,
        "requested_group_count": 4,
        "candidate_group_count": 0,
        "generation_error_group_count": 4,
        "generation_error_row_count": 64,
        "selected_group_count_after_attempt": 0,
    } for attempt in range(1, 65)]
    details = {
        "dynamic_max_attempts": 64,
        "selected_group_count": 0,
        "needed_group_count": 4,
        "candidate_group_count": 0,
        "discarded_group_count": 0,
        "generation_error_group_count": 256,
        "generation_error_row_count": 4096,
        "generation_requested_row_count": 4096,
        "observed_generation_error_rate": 1.0,
        "generation_attempt_summaries": attempts,
        "generation_attempt_summaries_sha256":
            common.canonical_json_sha256(attempts),
        "candidate_group_summaries": [],
        "candidate_group_summaries_sha256":
            common.canonical_json_sha256([]),
    }
    runtime = _runtime()
    section = {
        "file_sha256": _SHA,
        "canonical_rows_sha256": common.canonical_json_sha256([]),
        "row_count": 0,
    }
    evidence = c10_evidence.signed_training_transport({
        "run_name": c10_evidence.C10_CANONICAL_RUN_NAME,
        "c10_execution_profile": c10_evidence.C10_PROFILE,
        "scientific_variant": c10_evidence.C10_SCIENTIFIC_VARIANT,
        "transport_invalid": True,
        "failure_kind": "dynamic_sampling_transport_invalid",
        "completed_step": 0,
        "failed_step": 1,
        "preregistration_file": c10_evidence.C10_PREREGISTRATION_FILE,
        "preregistration_sha256":
            c10_evidence.C10_PREREGISTRATION_SHA256,
        "runtime_contract": runtime,
        "runtime_contract_sha256": common.canonical_json_sha256(runtime),
        "code_artifact": {"file_manifest_sha256": _SHA},
        "trajectory": dict(section),
        "metrics": dict(section),
        "failed_step_candidates": dict(section),
        "failure_details": details,
    })
    assert c10_evidence.validate_training_transport_invalid(
        evidence) is evidence
    c10_evidence._replay_failed_dynamic_candidates([], details)


def test_c10_signed_negative_gate_is_terminal_not_passable(monkeypatch):
    monkeypatch.setattr(
        grpo_gpu, "require_canonical_training_execution_environment_contract",
        lambda runtime: runtime["execution_environment"])
    failed = dict(_evidence())
    failed["passed"] = False
    failed["gate_outcome"] = "semantic_failed"
    failed = _resign(failed)
    assert c10_evidence.validate_behavior_gate_evidence(
        failed, require_passed=False) is failed
    with pytest.raises(ValueError, match="did not pass"):
        c10_evidence.validate_behavior_gate_evidence(failed)


def test_c10_runtime_rejects_boolean_substitution_for_numeric_fields(
        monkeypatch):
    monkeypatch.setattr(
        grpo_gpu, "require_canonical_training_execution_environment_contract",
        lambda runtime: runtime["execution_environment"])
    runtime = _runtime()
    assert c10_evidence.validate_runtime_contract(runtime) is runtime

    boolean_seed = dict(runtime, seed=False)
    with pytest.raises(ValueError, match="runtime contract drift"):
        c10_evidence.validate_runtime_contract(boolean_seed)

    boolean_nested = dict(runtime)
    boolean_nested["survivor_overlong_filter_contract"] = dict(
        runtime["survivor_overlong_filter_contract"],
        minimum_dynamic_sampling_survivors=True)
    with pytest.raises(ValueError, match="survivor_overlong_filter"):
        c10_evidence.validate_runtime_contract(boolean_nested)


def test_c10_exclusive_lease_binding_is_exact_and_immutable():
    binding = _lease_binding()
    assert c10_evidence.validate_exclusive_lease_binding(binding) is binding
    assert c10_evidence.lease_claim_metadata(binding["claim"])[
        "lease_state"] == "active"

    bad = dict(binding)
    bad["source"] = dict(binding["source"])
    bad["source"]["artifact_digest"] = ""
    unsigned = dict(bad)
    unsigned.pop("binding_sha256")
    bad["binding_sha256"] = common.canonical_json_sha256(unsigned)
    with pytest.raises(ValueError, match="source is invalid"):
        c10_evidence.validate_exclusive_lease_binding(bad)

    bad = dict(binding)
    bad["source"] = dict(binding["source"])
    bad["source"]["artifact_ref"] = "entity/project/wrong:v0"
    unsigned = dict(bad)
    unsigned.pop("binding_sha256")
    bad["binding_sha256"] = common.canonical_json_sha256(unsigned)
    with pytest.raises(ValueError, match="collection name"):
        c10_evidence.validate_exclusive_lease_binding(bad)


def test_c10_lease_epoch_release_is_attempt_and_result_bound():
    first = _lease_binding(version=0)
    release = c10_evidence.build_transport_release_claim(
        active_binding=first,
        result_artifact_ref=(
            f"entity/project/{c10_evidence.C10_RESULTS_ARTIFACT}:v0"),
        result_artifact_digest="result-digest-0",
        transport_scope="training",
        transport_evidence_sha256="c" * 64,
    )
    release_record = {
        "artifact_ref": (
            f"entity/project/{c10_evidence.C10_EXCLUSIVE_LEASE_ARTIFACT}:v1"),
        "artifact_digest": "lease-digest-1",
        "claim": release,
    }
    second = _lease_binding(version=2)
    authorized = []
    winner = c10_evidence.replay_exclusive_lease_records(
        [_lease_record(first), release_record, _lease_record(second),
         # A delayed duplicate release for v0 cannot close epoch v2.
         {**release_record, "artifact_ref": (
             f"entity/project/{c10_evidence.C10_EXCLUSIVE_LEASE_ARTIFACT}:v3"),
          "artifact_digest": "lease-digest-3"}],
        authorize_transport_release=lambda claim: authorized.append(
            claim["evidence_sha256"]),
    )
    assert winner is not None
    assert winner["artifact_ref"] == second["source"]["artifact_ref"]
    assert authorized == [release["evidence_sha256"]] * 2

    unknown = dict(release)
    unknown.pop("evidence_sha256")
    unknown["release"] = json.loads(json.dumps(unknown["release"]))
    unknown["release"]["active_claim"]["artifact_ref"] = (
        f"entity/project/{c10_evidence.C10_EXCLUSIVE_LEASE_ARTIFACT}:v99")
    unknown = c10_evidence.signed_lease_claim({
        key: value for key, value in unknown.items() if key != "version"})
    with pytest.raises(ValueError, match="not in its history"):
        c10_evidence.replay_exclusive_lease_records([
            _lease_record(first),
            {"artifact_ref": (
                f"entity/project/{c10_evidence.C10_EXCLUSIVE_LEASE_ARTIFACT}"
                ":v4"), "artifact_digest": "lease-digest-4",
             "claim": unknown},
        ])


def test_same_holder_claim_race_only_exact_published_version_can_proceed():
    first = _lease_binding(version=0, char="b")
    second = _lease_binding(version=1, char="b")
    winner = c10_evidence.replay_exclusive_lease_records([
        _lease_record(first), _lease_record(second)])
    assert c10_evidence.validate_published_lease_winner(
        winner,
        published_claim=first["claim"],
        published_artifact_ref=first["source"]["artifact_ref"],
        published_artifact_digest=first["source"]["artifact_digest"],
    ) is winner
    with pytest.raises(ValueError, match="lost to another active"):
        c10_evidence.validate_published_lease_winner(
            winner,
            published_claim=second["claim"],
            published_artifact_ref=second["source"]["artifact_ref"],
            published_artifact_digest=second["source"]["artifact_digest"],
        )


def test_transport_release_result_requires_bound_portable_proof(monkeypatch):
    binding = _lease_binding()
    release = c10_evidence.build_transport_release_claim(
        active_binding=binding,
        result_artifact_ref=(
            f"entity/project/{c10_evidence.C10_RESULTS_ARTIFACT}:v1"),
        result_artifact_digest="result-digest",
        transport_scope="behavior_gate",
        transport_evidence_sha256="d" * 64,
    )
    observed = {}

    def fake_validate(root, *, portable_model=False):
        observed.update({"root": root, "portable_model": portable_model})
        return {
            "behavior_gate": {"evidence_sha256": "d" * 64},
            "execution_contract": {"exclusive_lease": binding},
        }

    monkeypatch.setattr(
        c10_evidence, "validate_behavior_transport_bundle", fake_validate)
    metadata = {
        "c10_outcome": "transport_invalid",
        "c10_terminal_scientific_failure": False,
        "c10_transport_scope": "behavior_gate",
        "c10_preregistration_sha256":
            c10_evidence.C10_PREREGISTRATION_SHA256,
        "c10_canary_behavior_gate_evidence_sha256": "d" * 64,
    }
    c10_evidence.validate_transport_release_result(
        release,
        result_artifact_ref=(
            f"entity/project/{c10_evidence.C10_RESULTS_ARTIFACT}:v1"),
        result_artifact_digest="result-digest",
        result_metadata=metadata,
        result_root="/tmp/thin-result",
    )
    assert observed == {
        "root": "/tmp/thin-result", "portable_model": True}

    other_binding = _lease_binding(version=2)
    monkeypatch.setattr(
        c10_evidence, "validate_behavior_transport_bundle",
        lambda *_args, **_kwargs: {
            "behavior_gate": {"evidence_sha256": "d" * 64},
            "execution_contract": {"exclusive_lease": other_binding},
        })
    with pytest.raises(ValueError, match="different lease epoch"):
        c10_evidence.validate_transport_release_result(
            release,
            result_artifact_ref=(
                f"entity/project/{c10_evidence.C10_RESULTS_ARTIFACT}:v1"),
            result_artifact_digest="result-digest",
            result_metadata=metadata,
            result_root="/tmp/thin-result",
        )

    monkeypatch.setattr(
        c10_evidence, "validate_behavior_transport_bundle", fake_validate)
    metadata["c10_outcome"] = "incomplete"
    with pytest.raises(ValueError, match="metadata is invalid"):
        c10_evidence.validate_transport_release_result(
            release,
            result_artifact_ref=(
                f"entity/project/{c10_evidence.C10_RESULTS_ARTIFACT}:v1"),
            result_artifact_digest="result-digest",
            result_metadata=metadata,
            result_root="/tmp/thin-result",
        )


def test_c10_pipeline_validation_is_exact_and_separate(monkeypatch):
    from tmx_gpu import orchestrator_receipt

    monkeypatch.setattr(
        pipeline_gpu, "_validated_executing_code_source_manifest",
        lambda args, profile_label: {"profile": profile_label})
    monkeypatch.setattr(
        grpo_gpu, "load_code_source_manifest",
        lambda _path: {"source": {
            "artifact_ref": "entity/project/code:v1",
            "artifact_digest": "code-digest"}})
    monkeypatch.setattr(
        orchestrator_receipt, "load_binding", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        orchestrator_receipt, "validate_binding",
        lambda value, **kwargs: value)
    monkeypatch.setattr(
        c10_evidence, "load_exclusive_lease_binding",
        lambda *args, **kwargs: {"validated": "lease"})
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--output-dir", "/tmp/c10", "--run-name",
        c10_evidence.C10_CANONICAL_RUN_NAME,
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "--orchestrator-command-sha256", _SHA,
        "--orchestrator-receipt-artifact", "entity/project/receipt:latest",
        "--c10-exclusive-lease-binding", "/tmp/c10-lease.json",
        "--c10-execution-profile", "canary",
        "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
        "--novelty-normalization", "selected_batch_zscore",
        "--scientific-variant", grpo_gpu.C10_SCIENTIFIC_VARIANT,
        "--scout-lr", "1e-6", "--scout-steps", "50",
        "--scout-only", "--probe-every", "0", "--reload-every", "1",
        "--save-every", "25", "--keep-checkpoints", "1",
        "--ckpt-artifact-every", "25",
        "--mask-truncated",
        "--truncation-baseline-mode", "unmasked_survivors",
        "--vllm-gpus", "1", "--vllm-max-model-len", "36864",
        "--wandb",
    ])
    args.code_source_manifest = "/tmp/immutable-code-manifest.json"
    pipeline_gpu.validate_args(args)
    assert not args.c9_execution_profile
    assert not args.stabilization_cell

    args.mask_truncated = False
    with pytest.raises(ValueError, match="C10 canary contract drift"):
        pipeline_gpu.validate_args(args)


def test_c10_negative_outcomes_expose_portable_validation(
        tmp_path, monkeypatch):
    observed = {}

    def fake_terminal(root, *, require_passed, require_gate_stage=True,
                      portable_model=False):
        observed.update({
            "root": root,
            "require_passed": require_passed,
            "require_gate_stage": require_gate_stage,
            "portable_model": portable_model,
        })
        return {
            "root": str(tmp_path),
            "behavior_gate": {"gate_outcome": "semantic_failed"},
        }

    monkeypatch.setattr(
        c10_evidence, "_validate_terminal_bundle", fake_terminal)
    c10_evidence.validate_terminal_semantic_failure(
        tmp_path, portable_model=True)
    assert observed["portable_model"] is True
    training_signature = inspect.signature(
        c10_evidence.validate_terminal_training_failure)
    assert training_signature.parameters["portable"].default is False


def test_c10_resume_state_contract_is_independent(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    expected = {"version": "c10", "contract_sha256": _SHA}
    monkeypatch.setattr(
        pipeline_gpu, "_resolved_c9_execution_contract", lambda _args: None)
    monkeypatch.setattr(
        pipeline_gpu, "_c10_canary_execution_contract",
        lambda _args: expected)
    args = SimpleNamespace(
        output_dir=str(out), resume=True, c10_execution_profile="canary")
    (out / "pipeline_state.json").write_text(json.dumps({"stages": {}}))
    with pytest.raises(RuntimeError, match="C10 resume execution contract"):
        pipeline_gpu._load_state(args)
    (out / "pipeline_state.json").write_text(json.dumps({
        "stages": {}, "c10_execution_contract": expected}))
    assert pipeline_gpu._load_state(args)["c10_execution_contract"] == expected


def test_c10_gate_runs_before_scout_only_return_and_has_distinct_stage():
    source = inspect.getsource(pipeline_gpu.run_round)
    assert source.index("grpo_gpu.run_grpo") < source.index(
        "_durably_publish_c10_training_outcome_if_present")
    assert source.index("_validate_c10_canary_scout_endpoint") < source.index(
        'if getattr(args, "scout_only", False)')
    assert '"c10_canary_behavior_gate"' in source


def test_c10_training_transport_is_synchronously_published(tmp_path, monkeypatch):
    out = tmp_path / "run"
    transport_path = out / "round_01" / "scout" / \
        "c10_training_transport_invalid.json"
    transport_path.parent.mkdir(parents=True)
    transport_path.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        output_dir=str(out), c10_execution_profile="canary")
    validated = {
        "execution_contract": {"contract_sha256": _SHA},
        "training_transport": {"evidence_sha256": "c" * 64},
    }
    monkeypatch.setattr(
        c10_evidence, "validate_training_transport_bundle",
        lambda root: validated)
    pushed = []
    monkeypatch.setattr(
        pipeline_gpu, "_push_results_artifact",
        lambda run, observed_args, note: pushed.append(
            (run, observed_args, note,
             dict(observed_args._c10_terminal_metadata))))
    assert pipeline_gpu._durably_publish_c10_training_outcome_if_present(
        args, "wandb-run", round_idx=1) == "transport_invalid"
    assert pushed[0][2] == "r1-c10-training-transport_invalid"
    assert pushed[0][3]["c10_transport_scope"] == "training"
    assert pushed[0][3]["c10_training_transport_evidence_sha256"] == \
        "c" * 64


def test_launcher_exposes_protected_c10_identity():
    source = (pipeline_gpu.REPO_ROOT / "tmx_gpu" / "launch" /
              "expdis_job.sh").read_text(encoding="utf-8")
    for token in (
        "C10_EXECUTION_PROFILE", "TRUNCATION_BASELINE_MODE",
        "--c10-execution-profile", "--truncation-baseline-mode",
        "--c10-exclusive-lease-binding",
        'EXPECTED_C9_VARIANT="expdis_c10_survivor_consistent_overlong_filter"',
        'elif (os.environ.get("C10_EXECUTION_PROFILE") == "canary"',
        "recover_durable_transport_release",
        "same-holder resume are forbidden",
        "validate_published_lease_winner",
        "validate_prelease_environment",
        "build_transport_release_claim",
        "validate_transport_release_result",
        'final_model = c10_publishable_final',
        '"round_*/scout/c10_failed_selected_batch.jsonl"',
        '"round_*/scout/c10_failed_selected_batch_manifest.json"',
    ):
        assert token in source
    c10_variant = source.index(
        'EXPECTED_C9_VARIANT="expdis_c10_survivor_consistent_overlong_filter"')
    generic_variant_check = source.index(
        'if [ "$SCIENTIFIC_VARIANT" != "$EXPECTED_C9_VARIANT" ]')
    assert c10_variant < generic_variant_check
    assert source.index("logged_results.wait()") < source.index(
        "build_transport_release_claim(", source.index("logged_results.wait()"))
    assert "--gen-concurrency" in source
    assert source.index("validate_prelease_environment") < source.index(
        "publish_claim(active_claim_payload())")


def _canonical_prelease_environment():
    return {
        "RUN_NAME": c10_evidence.C10_CANONICAL_RUN_NAME,
        "MODEL": common.CANONICAL_TRAINING_MODEL_ID,
        "MODEL_REVISION": common.CANONICAL_TRAINING_MODEL_REVISION,
        "C10_EXECUTION_PROFILE": "canary",
        "C9_EXECUTION_PROFILE": "",
        "C9_MULTIEXPLORER_PROFILE": "",
        "SINGLE_SCOUT_MULTIROUND_PROFILE": "",
        "STABILIZATION_CELL": "",
        "SCIENTIFIC_VARIANT": c10_evidence.C10_SCIENTIFIC_VARIANT,
        "ROUNDS": "1", "SCOUTS_PER_ROUND": "1", "SCOUT_STEPS": "50",
        "SCOUT_ONLY": "1", "SEED": "0", "DATASET": "dapo_math_17k",
        "MAX_TRAIN_EXAMPLES": "20000", "PROMPTS_PER_STEP": "4",
        "NUM_GENERATIONS": "16", "RL_VALIDITY_MODE": "c8",
        "RELOAD_EVERY": "1", "SAVE_EVERY": "25",
        "KEEP_CHECKPOINTS": "1", "CKPT_ARTIFACT_EVERY": "25",
        "PROBE_EVERY": "0", "COMPLETION_BUDGET": "0",
        "MASK_TRUNCATED": "1",
        "TRUNCATION_BASELINE_MODE": "unmasked_survivors",
        "PPO_MODE": "0", "UPDATES_PER_ROLLOUT": "1",
        "EXPDIS_STABILIZATION_MODE": "c9_selected_batch_update",
        "NOVELTY_NORMALIZATION": "selected_batch_zscore",
        "POLICY": "coverage_pool_c8", "MAX_MODEL_LEN": "36864",
        "OPTIMIZER": "adamw", "SFT_OPTIMIZER": "adamw",
        "SKIP_SCOUT": "0", "STOP_AFTER_SFT": "0",
        "TRAJECTORIES": "", "RESUME": "1",
        "FRESH_C8_SFT_GATE": "0", "FRESH_C8_MULTISCOUT_COMPONENT": "0",
        "LAMBDA": "0.5", "SCOUT_LR": "1e-6",
        "DYNAMIC_MAX_ATTEMPTS": "0",
        "LAMBDA_SCHEDULE_EXPLICIT": "",
    }


@pytest.mark.parametrize("key,bad", [
    ("RUN_NAME", "other"),
    ("MODEL", "other/model"),
    ("SEED", "1"),
    ("SCOUT_STEPS", "51"),
    ("PROMPTS_PER_STEP", "8"),
    ("RELOAD_EVERY", "2"),
    ("SCOUT_LR", "5e-6"),
    ("MASK_TRUNCATED", "0"),
    ("DYNAMIC_MAX_ATTEMPTS", "63"),
])
def test_c10_prelease_environment_rejects_scientific_drift(key, bad):
    environment = _canonical_prelease_environment()
    assert c10_evidence.validate_prelease_environment(environment)
    environment[key] = bad
    with pytest.raises(ValueError, match="prelease environment drift"):
        c10_evidence.validate_prelease_environment(environment)


@pytest.mark.parametrize("key", c10_evidence.C10_FORBIDDEN_DATASET_ENV)
def test_c10_prelease_rejects_ambient_dataset_overrides(key):
    environment = _canonical_prelease_environment()
    environment[key] = "/tmp/override" if key.endswith("PATH") or \
        key.endswith("JSONL") else "1"
    with pytest.raises(ValueError, match="forbids ambient overrides"):
        c10_evidence.validate_prelease_environment(environment)
