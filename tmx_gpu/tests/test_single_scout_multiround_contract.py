"""Strict K=1 multi-round contract and publication-boundary tests."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tmx_gpu import (common, eval_matrix, grpo_gpu, mrme_contract,
                     pipeline_gpu)
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures
from tmx_jax import data as data_mod


def _build_contract(*, rounds: int) -> dict:
    schedule = (0.5, 0.35, 0.25, 0.15)[:rounds]
    return mrme_contract.build_contract(
        model=common.CANONICAL_TRAINING_MODEL_ID,
        model_revision=common.CANONICAL_TRAINING_MODEL_REVISION,
        dataset="dapo_math_17k",
        dataset_identity=data_mod.training_dataset_identity(
            "dapo_math_17k"),
        max_train_examples=20_000,
        rounds=rounds,
        scouts_per_round=1,
        lambda_schedule=schedule,
        base_seed=0,
        scout_steps=200,
        central_steps=100,
        prompts_per_step=4,
        num_generations=16,
        pooling_policy=mrme_contract.MRME_POOLING_POLICY,
        max_accepted=500,
        harvest_max_passes=0,
        stabilization_profile=mrme_contract.C9_MULTIEXPLORER_PROFILE,
        training_contract={"fixture": True},
        code_identity={
            "source": {
                "artifact_ref": "entity/project/code:v1",
                "artifact_digest": "code-digest",
            },
        },
    )


@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
def test_k1_round_contract_is_distinct_versioned_and_canonical(rounds):
    contract = _build_contract(rounds=rounds)

    assert contract["method"] == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
    assert contract["method_version"] == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
    assert contract["single_scout_multiround"] is True
    assert contract["scouts_per_round"] == 1
    assert contract["dataset_membership_mode"] == \
        "deterministic_disjoint_one_over_r"
    assert contract["lambda_schedule"] == [0.5, 0.35, 0.25, 0.15][:rounds]
    assert [row["shard_index"] for row in contract["rounds_spec"]] == \
        list(range(rounds))
    assert all(row["shard_count"] == rounds
               for row in contract["rounds_spec"])
    assert all(len(row["scouts"]) == 1
               for row in contract["rounds_spec"])
    assert [row["scouts"][0]["seed"]
            for row in contract["rounds_spec"]] == list(range(rounds))
    assert mrme_contract.is_single_scout_multiround(contract)
    assert mrme_contract.validate_contract(contract) == contract


def test_k1_round_contract_requires_c9_and_at_most_four_rounds():
    with pytest.raises(ValueError, match="stabilized C9"):
        mrme_contract.build_contract(
            **{
                key: value for key, value in _contract_kwargs(rounds=2).items()
                if key != "stabilization_profile"
            },
            stabilization_profile=mrme_contract.C8_MULTIEXPLORER_PROFILE,
        )
    with pytest.raises(ValueError, match="1 <= --rounds <= 4"):
        mrme_contract.build_contract(**_contract_kwargs(rounds=5))


def _contract_kwargs(*, rounds: int) -> dict:
    values = {
        "model": common.CANONICAL_TRAINING_MODEL_ID,
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "dataset": "dapo_math_17k",
        "dataset_identity": data_mod.training_dataset_identity(
            "dapo_math_17k"),
        "max_train_examples": 20_000,
        "rounds": rounds,
        "scouts_per_round": 1,
        "lambda_schedule": tuple(
            0.5 / (index + 1) for index in range(rounds)),
        "base_seed": 0,
        "scout_steps": 200,
        "central_steps": 100,
        "prompts_per_step": 4,
        "num_generations": 16,
        "pooling_policy": mrme_contract.MRME_POOLING_POLICY,
        "max_accepted": 500,
        "harvest_max_passes": 0,
        "stabilization_profile": mrme_contract.C9_MULTIEXPLORER_PROFILE,
        "training_contract": {"fixture": True},
        "code_identity": {"fixture": True},
    }
    return values


def test_k1_round_partition_binds_new_method_even_at_r1():
    manifest = {
        "shard_index": 0,
        "shard_count": 1,
        "shard_policy": mrme_contract.MRME_SHARD_POLICY,
        "shard_seed": 0,
        "required": False,
        "fraction_numerator": 1,
        "fraction_denominator": 1,
        "total_rows": 2,
        "start": 0,
        "end": 2,
        "full_dataset_sha256": "f" * 64,
        "members": [
            {"source_index": 0, "row_sha256": "a" * 64},
            {"source_index": 1, "row_sha256": "b" * 64},
        ],
    }
    partition = mrme_contract.assert_round_manifests_disjoint(
        [manifest], expected_rounds=1,
        expected_method=
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD)

    assert partition["method"] == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
    assert partition["dataset_membership_mode"] == \
        "deterministic_disjoint_one_over_r"


def test_k_greater_than_one_contract_shape_is_unchanged():
    contract = mrme_contract.build_contract(
        **{
            **_contract_kwargs(rounds=2),
            "scouts_per_round": 2,
        })

    assert contract["method"] == mrme_contract.DISJOINT_SHARD_MRME_METHOD
    assert "method_version" not in contract
    assert "single_scout_multiround" not in contract


def _pipeline_args(tmp_path: Path, *, rounds: int = 4):
    code_path = tmp_path / "code-source-manifest.json"
    code_path.write_text("{}\n", encoding="utf-8")
    schedule = ",".join(("0.5", "0.35", "0.25", "0.15")[:rounds])
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--model", common.CANONICAL_TRAINING_MODEL_ID,
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "--output-dir", str(tmp_path / "run"),
        "--run-name", f"k1-r{rounds}",
        "--rounds", str(rounds),
        "--scouts-per-round", "1",
        "--single-scout-multiround-profile",
        mrme_contract.SINGLESCOUT_MULTIROUND_PROFILE,
        "--scout-steps", "200",
        "--central-steps", "100",
        "--lambda-schedule", schedule,
        "--policy", mrme_contract.MRME_POOLING_POLICY,
        "--min-accepted", "500",
        "--max-accepted", "500",
        "--harvest-max-passes", "0",
        "--prompts-per-step", "4",
        "--num-generations", "16",
        "--dataset", "dapo_math_17k",
        "--max-train-examples", "20000",
        "--sync-dir", str(tmp_path / "sync"),
        "--code-source-manifest", str(code_path),
        "--vllm-gpus", "1,2,3,4,5,6,7",
        "--vllm-max-model-len", "36864",
        "--probe-every", "25",
        "--reload-every", "1",
        "--dynamic-max-attempts", "0",
        "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
        "--novelty-normalization", "selected_batch_zscore",
        "--scientific-variant",
        mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "--wandb",
        "--scout-lr", "1e-6",
        "--seed", "0",
    ])
    code_ref = "entity/project/code:v33"
    code_digest = "code-digest"
    receipt = receipt_fixtures.binding(
        run_name=args.run_name, code_ref=code_ref,
        code_digest=code_digest)
    args.orchestrator_command_sha256 = receipt[
        "receipt"]["job"]["command_sha256"]
    args.orchestrator_receipt_artifact = receipt[
        "receipt"]["job"]["environment"][
            "ORCHESTRATOR_RECEIPT_ARTIFACT"]
    args._code_artifact_binding = {
        "source": {
            "artifact_ref": code_ref,
            "artifact_digest": code_digest,
        },
    }
    args._orchestrator_receipt_binding = receipt
    args._training_execution_environment = receipt_fixtures.environment(
        run_name=args.run_name, code_ref=code_ref,
        code_digest=code_digest)
    return args


@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
def test_explicit_k1_profile_enters_generalized_campaign_path(
    tmp_path, monkeypatch, rounds,
):
    args = _pipeline_args(tmp_path, rounds=rounds)
    pipeline_gpu.validate_args(args)
    assert pipeline_gpu._true_mrme(args)
    assert pipeline_gpu._is_single_scout_multiround_profile(args)
    assert pipeline_gpu._is_c9_round_campaign(args)

    code_identity = {
        "source": {
            "artifact_ref": "entity/project/code:v33",
            "artifact_digest": "code-digest",
        },
        "files": [],
        "file_manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        pipeline_gpu, "_validated_executing_code_source_manifest",
        lambda _args, *, profile_label: code_identity)
    contract = pipeline_gpu._build_mrme_contract(
        args, pipeline_gpu.parse_lambda_schedule(args))

    assert contract["method"] == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
    assert contract["method_version"] == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
    assert contract["single_scout_multiround"] is True
    assert contract["scouts_per_round"] == 1
    assert contract["scout_steps_per_scout"] == 200
    assert contract["central_steps_per_round"] == 100
    assert contract["max_accepted"] == 500
    assert contract["training_contract"]["stabilization_profile"] == \
        mrme_contract.C9_MULTIEXPLORER_PROFILE
    assert contract["training_contract"]["scout_grpo"][
        "scout_novelty_contract"]["rnd_lifecycle"] == \
        grpo_gpu.C9_RND_LIFECYCLE
    assert pipeline_gpu._round_campaign_summary_identity(contract) == {
        "method":
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD,
        "method_version":
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
        "mrme": False,
        "multi_explorer": False,
        "single_scout_multiround": True,
    }
    args._mrme_contract = contract
    args._mrme_shard_manifest_sha256 = "b" * 64
    child = pipeline_gpu._grpo_args(
        args, phase="scout", model=args.model,
        output_dir=str(tmp_path / "child"),
        lam=float(contract["lambda_schedule"][0]), steps=200,
        round_idx=1, global_step_offset=0, seed_override=0,
        num_scouts_override=1)
    assert child.mrme_method == contract["method"]
    assert child.mrme_method_version == contract["method_version"]
    assert child.single_scout_multiround is True
    child.mrme_scout_index = 1
    grpo_gpu.validate_expdis_stabilization(
        child, phase="scout",
        lambda_novelty=float(contract["lambda_schedule"][0]))
    grpo_gpu.apply_completion_budget(child)
    grpo_gpu.enforce_probe_serving_len(child, pool=None)
    runtime = grpo_gpu.build_runtime_contract(
        child, resolved_lr=1e-6, resolved_steps=200)
    assert runtime["multi_explorer_contract"] == {
        "contract_sha256": contract["contract_sha256"],
        "method": contract["method"],
        "method_version": contract["method_version"],
        "single_scout_multiround": True,
        "mrme": False,
        "multi_explorer": False,
        "stabilization_profile": contract["stabilization_profile"],
        "scout_index": 1,
        "scouts_per_round": 1,
        "shard_manifest_sha256": "b" * 64,
    }


def _k1_stage_args(**overrides):
    values = {
        "mrme_contract_sha256": "a" * 64,
        "mrme_shard_manifest_sha256": "b" * 64,
        "mrme_scout_index": 1,
        "mrme_scouts_per_round": 1,
        "mrme_method":
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD,
        "mrme_method_version":
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
        "single_scout_multiround": True,
        "mrme_stabilization_profile":
            mrme_contract.C9_MULTIEXPLORER_PROFILE,
        "scientific_variant":
            mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "phase": "scout",
        "seed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_grpo_artifacts_accept_only_exact_versioned_k1_identity():
    metadata = grpo_gpu._mrme_artifact_metadata(_k1_stage_args())
    assert metadata["method_version"] == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
    assert metadata["single_scout_multiround"] is True
    assert metadata["mrme"] is False
    assert metadata["multi_explorer"] is False
    assert metadata["mrme_scouts_per_round"] == 1

    for mutation in (
        {"mrme_method_version": ""},
        {"mrme_method_version": "wrong-version"},
        {"single_scout_multiround": False},
        {"mrme_method": mrme_contract.DISJOINT_SHARD_MRME_METHOD},
    ):
        with pytest.raises(
                ValueError, match="invalid multi-explorer stage artifact"):
            grpo_gpu._mrme_artifact_metadata(
                _k1_stage_args(**mutation))


def test_k1_without_explicit_profile_remains_ordinary_pipeline(tmp_path):
    args = _pipeline_args(tmp_path, rounds=2)
    args.single_scout_multiround_profile = ""
    assert not pipeline_gpu._true_mrme(args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"scouts_per_round": 2}, "scouts-per-round 1"),
        ({"rounds": 5}, "1 <= --rounds <= 4"),
        ({"c9_execution_profile": "full", "stabilization_cell": "C5"},
         "mutually exclusive"),
        ({"c9_multiexplorer_profile":
          mrme_contract.C9_MULTIEXPLORER_PROFILE},
         "requires --scouts-per-round > 1"),
        ({"lambda_schedule": "0.5,0.5,0.25,0.15"},
         "strictly decreasing"),
        ({"wandb": False}, "online --wandb"),
        ({"scout_steps": 199}, "contract drift"),
        ({"central_steps": 99}, "contract drift"),
        ({"max_accepted": 499}, "contract drift"),
    ],
)
def test_k1_pipeline_profile_fails_closed(tmp_path, mutation, message):
    args = _pipeline_args(tmp_path, rounds=4)
    for key, value in mutation.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=message):
        pipeline_gpu.validate_args(args)


def test_launcher_exposes_and_publishes_distinct_k1_profile():
    launcher = Path(pipeline_gpu.__file__).parent / "launch" / "expdis_job.sh"
    source = launcher.read_text(encoding="utf-8")
    assert '--single-scout-multiround-profile ' \
        '"$SINGLE_SCOUT_MULTIROUND_PROFILE"' in source
    assert '"method": ' \
        '"disjoint_shard_annealed_single_scout_multiround"' in source
    assert '"single_scout_multiround": True' in source
    assert '"mrme": False' in source
    assert '"multi_explorer": False' in source


def _publication_binding() -> dict:
    sha = "a" * 64
    return eval_matrix._signed_canonical_payload(
        eval_matrix.TRAINING_CAMPAIGN_BINDING_VERSION,
        {
            "method": eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD,
            "method_version":
                eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
            "campaign_artifact": {
                "artifact_ref": "entity/project/campaign:v1",
                "artifact_digest": "campaign-digest",
                "artifact_type": "expdis-results",
            },
            "model_artifact": {
                "artifact_ref": "entity/project/model:v2",
                "artifact_digest": "model-digest",
                "artifact_type": "model",
            },
            "code_artifact": {
                "artifact_ref": "entity/project/code:v3",
                "artifact_digest": "code-digest",
            },
            "evaluated_checkpoint_sha256": sha,
            "completion_kind": "signed_multi_explorer_completed_campaign",
            "completion_evidence_sha256": sha,
            "execution_contract_sha256": sha,
            "final_model_identity_sha256": sha,
            "trainer_state_sha256": sha,
            "checkpoint_file_manifest_sha256": sha,
        },
    )


def test_publication_recognizes_only_versioned_k1_multiround_identity():
    metadata = {
        "method": eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD,
        "method_version":
            eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
    }
    assert eval_matrix.infer_expdis_method_from_artifact_metadata(
        metadata, where="fixture") == \
        eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD
    binding = _publication_binding()
    assert eval_matrix.validate_training_campaign_binding(binding) == binding

    for version in (None, "wrong-version"):
        invalid_metadata = copy.deepcopy(metadata)
        invalid_metadata["method_version"] = version
        with pytest.raises(
                eval_matrix.MatrixValidationError,
                match="method-version mismatch"):
            eval_matrix.infer_expdis_method_from_artifact_metadata(
                invalid_metadata, where="fixture")

        unsigned = {
            key: copy.deepcopy(value) for key, value in binding.items()
            if key not in {"schema_version", "binding_sha256"}
        }
        unsigned["method_version"] = version
        invalid_binding = eval_matrix._signed_canonical_payload(
            eval_matrix.TRAINING_CAMPAIGN_BINDING_VERSION, unsigned)
        with pytest.raises(
                eval_matrix.MatrixValidationError,
                match="binding method version"):
            eval_matrix.validate_training_campaign_binding(invalid_binding)


def test_eval_and_training_method_constants_cannot_drift():
    assert eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
    assert eval_matrix.EXPDIS_SINGLESCOUT_MULTIROUND_METHOD_VERSION == \
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION


def test_publication_campaign_replay_requires_k1_method_version(
    tmp_path, monkeypatch,
):
    campaign_root = tmp_path / "campaign"
    model_root = tmp_path / "model"
    campaign_root.mkdir()
    model_root.mkdir()
    code = {
        "source": {
            "artifact_ref": "entity/project/code:v7",
            "artifact_digest": "code-digest",
        },
    }
    manifest = {"sha256": "9" * 64, "files": []}
    contract = {
        "method": mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD,
        "method_version":
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION,
        "single_scout_multiround": True,
        "stabilization_profile": "fixture-profile",
        "scientific_variant": "fixture-variant",
        "contract_sha256": "1" * 64,
        "rounds": 4,
        "scouts_per_round": 1,
        "code_identity": code,
    }
    original_final = "/workspace/runs/k1-r4/round_04/central/final"
    local_identity = {
        "path": str(model_root),
        "phase": "central",
        "checkpoint_file_manifest": manifest,
        "identity_sha256": "local-only",
    }
    portable_identity = copy.deepcopy(local_identity)
    portable_identity.pop("identity_sha256")
    portable_identity["path"] = original_final
    completion_payload = {
        "version": "gpu_multi_explorer_completed_campaign_v2",
        "method": contract["method"],
        "method_version": contract["method_version"],
        "single_scout_multiround": True,
        "stabilization_profile": contract["stabilization_profile"],
        "scientific_variant": contract["scientific_variant"],
        "contract_sha256": contract["contract_sha256"],
        "partition_sha256": "2" * 64,
        "rounds": 4,
        "scouts_per_round": 1,
        "external_p1_required": True,
        "headline_eligible": False,
        "external_publication_evaluation":
            mrme_contract.STRICT_EXTERNAL_P1_REQUIREMENT,
        "final_model": original_final,
        "final_model_identity_sha256": common.canonical_json_sha256(
            portable_identity),
    }
    completion = {
        **completion_payload,
        "evidence_sha256": common.canonical_json_sha256(completion_payload),
    }
    portable_payload = {
        "version": mrme_contract.MRME_PORTABLE_CAMPAIGN_VERSION,
        "completion": completion,
    }
    portable = {
        **portable_payload,
        "evidence_sha256": common.canonical_json_sha256(portable_payload),
    }
    common.atomic_write_json(
        str(campaign_root / "mrme_completed_campaign.json"), completion)
    common.atomic_write_json(
        str(campaign_root / "mrme_contract.json"), {"thin": True})
    common.atomic_write_json(
        str(campaign_root / "mrme_portable_campaign.json"), portable)
    common.atomic_write_json(str(model_root / "trainer_state.json"), {
        "checkpoint_file_manifest": manifest,
        "mrme_provenance": {
            "method": contract["method"],
            "mrme_contract_sha256": contract["contract_sha256"],
            "phase": "central",
            "mrme_scouts_per_round": 1,
        },
        "code_artifact_binding": code,
    })
    monkeypatch.setattr(
        mrme_contract, "validate_contract", lambda _value: contract)
    monkeypatch.setattr(
        mrme_contract, "validate_portable_completed_campaign",
        lambda _root, *, allowed_output_parent: copy.deepcopy(completion))
    monkeypatch.setattr(
        mrme_contract, "is_c9_multi_explorer", lambda _contract: False)
    monkeypatch.setattr(
        mrme_contract, "model_handoff_identity",
        lambda _root, expected_phase: copy.deepcopy(local_identity))
    metadata = {
        "method": contract["method"],
        "method_version": contract["method_version"],
        "single_scout_multiround": True,
        "stabilization_profile": contract["stabilization_profile"],
        "scientific_variant": contract["scientific_variant"],
        "external_p1_required": True,
        "headline_eligible": False,
        "mrme_contract_sha256": contract["contract_sha256"],
        "mrme_partition_sha256": completion["partition_sha256"],
        "mrme_completion_evidence_sha256": completion["evidence_sha256"],
        "mrme_portable_campaign_evidence_sha256": portable["evidence_sha256"],
        "mrme_final_model_identity_sha256": completion[
            "final_model_identity_sha256"],
    }

    facts = eval_matrix._validate_mrme_campaign(
        campaign_root, model_root,
        campaign_metadata=metadata, model_metadata=metadata)
    assert facts["method"] == contract["method"]
    assert facts["method_version"] == contract["method_version"]

    invalid = copy.deepcopy(completion)
    invalid["method_version"] = "wrong-version"
    invalid["evidence_sha256"] = common.canonical_json_sha256({
        key: value for key, value in invalid.items()
        if key != "evidence_sha256"
    })
    common.atomic_write_json(
        str(campaign_root / "mrme_completed_campaign.json"), invalid)
    invalid_portable_payload = {
        "version": mrme_contract.MRME_PORTABLE_CAMPAIGN_VERSION,
        "completion": invalid,
    }
    common.atomic_write_json(
        str(campaign_root / "mrme_portable_campaign.json"), {
            **invalid_portable_payload,
            "evidence_sha256": common.canonical_json_sha256(
                invalid_portable_payload),
        })
    monkeypatch.setattr(
        mrme_contract, "validate_portable_completed_campaign",
        lambda _root, *, allowed_output_parent: copy.deepcopy(invalid))
    with pytest.raises(
            eval_matrix.MatrixValidationError,
            match="completion method version"):
        eval_matrix._validate_mrme_campaign(
            campaign_root, model_root,
            campaign_metadata=metadata, model_metadata=metadata)
