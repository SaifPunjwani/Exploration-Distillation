"""CPU-only end-to-end/resume test for the true K x R MR-ME pipeline."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from tmx_gpu import common, grpo_gpu, mrme_contract, pipeline_gpu, sft_gpu
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures
from tmx_jax import data as data_mod


class FakeManagedPool:
    managed = True

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def reload(self, model_dir: str) -> None:
        self.model_dir = model_dir


def _write_deployable_grpo_final(gargs, *, code_identity=None) -> str:
    final = Path(gargs.output_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    (final / "config.json").write_text("{}\n", encoding="utf-8")
    (final / "model.safetensors").write_bytes(
        f"{gargs.phase}:{gargs.round}:{gargs.seed}".encode())
    grpo_gpu.apply_completion_budget(gargs)
    grpo_gpu.enforce_probe_serving_len(gargs, pool=None)
    lr = float(gargs.lr) if gargs.lr is not None else (
        1e-6 if gargs.phase == "central" else 5e-6)
    runtime = grpo_gpu.build_runtime_contract(
        gargs, resolved_lr=lr, resolved_steps=int(gargs.steps))
    manifest = common.directory_file_manifest(
        str(final), exclude_relative_paths=("trainer_state.json",))
    common.atomic_write_json(
        str(final / "trainer_state.json"),
        {
            "phase": str(gargs.phase),
            "checkpoint_kind": "deployable_model",
            "round": int(gargs.round),
            "step": int(gargs.steps),
            "lambda": (0.0 if gargs.phase == "central"
                       else float(gargs.lambda_novelty)),
            "external_lineage": None,
            "external_lineage_sha256": None,
            "runtime_contract": runtime,
            "code_artifact_binding": code_identity,
            "mrme_provenance": grpo_gpu._mrme_artifact_metadata(gargs),
            "checkpoint_file_manifest": manifest,
        },
    )
    return str(final)


def _write_full_scout_trajectories(gargs) -> None:
    path = Path(gargs.output_dir) / "explorer_trajectories.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_per_step = int(gargs.prompts_per_step) * int(gargs.num_generations)
    with path.open("w", encoding="utf-8") as handle:
        for step in range(1, int(gargs.steps) + 1):
            for index in range(rows_per_step):
                problem = f"round-{gargs.round}-problem-{index}"
                handle.write(json.dumps({
                    "problem_id": problem,
                    "prompt_text": problem,
                    "completion_text": "reasoning \\boxed{0}",
                    "ground_truth": "0",
                    "correctness_reward": 1.0,
                    "novelty_reward": 0.1,
                    "blended_reward": 1.05,
                    "reward": 1.05,
                    "r_correct": 1.0,
                    "r_novel": 0.1,
                    "is_correct": True,
                    "explorer_step": step,
                    "step": step,
                    "finish_reason": "stop",
                    "completion_token_length": 4,
                    "clipped": False,
                    "terminated": True,
                    "valid_answer": True,
                }) + "\n")


def _fake_sft(sargs) -> str:
    final = Path(sargs.output_dir) / "actual_sft_final"
    final.mkdir(parents=True, exist_ok=True)
    (final / "config.json").write_text("{}\n", encoding="utf-8")
    (final / "model.safetensors").write_bytes(b"fake-sft")
    with open(sargs.trajectory_source_manifest, encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    manifest = common.directory_file_manifest(
        str(final), exclude_relative_paths=("sft_provenance.json",))
    input_identity = pipeline_gpu._sft_input_identity(
        sargs.accepted, epochs=int(sargs.epochs), seed=int(sargs.seed))
    completed = {
        "version": "sft_completed_run_evidence_v1",
        "training_completed": True,
        "optimizer_steps_executed": 1000,
        "micro_steps_executed": 1000,
        "saved_optimizer_steps": [],
        "training_evidence_sha256": None,
    }
    completed["evidence_sha256"] = common.canonical_json_sha256(completed)
    resolved_max_total_len = (
        2048 + int(sargs.completion_budget)
        if int(sargs.completion_budget) > 0 else int(sargs.max_total_len))
    common.atomic_write_json(
        str(final / "sft_provenance.json"),
        {
            **input_identity,
            "termination_mode": str(sargs.sft_termination_mode),
            "model_init": str(sargs.model),
            "sft_runtime_contract": sft_gpu.build_sft_runtime_contract(
                sargs, resolved_max_total_len=resolved_max_total_len),
            "accepted_library_digest": common.semantic_sft_jsonl_digest(
                sargs.accepted),
            "optimizer_steps": 1000,
            "micro_steps": 1000,
            "completed_run_evidence": completed,
            "trajectory_source": {
                "manifest": source_manifest,
                "digest": common.trajectory_source_manifest_digest(
                    source_manifest),
            },
            "checkpoint_file_manifest": manifest,
        },
    )
    return str(final)


def _args(tmp_path: Path):
    code = tmp_path / "code-manifest.json"
    code.write_text("{}\n", encoding="utf-8")
    return pipeline_gpu.build_arg_parser().parse_args([
        "--model", "test/base-model",
        "--model-revision", "test-base-revision",
        "--output-dir", str(tmp_path / "run"),
        "--run-name", "mrme-dry",
        "--rounds", "2",
        "--scouts-per-round", "2",
        "--scout-steps", "1",
        "--central-steps", "1",
        "--lambda-schedule", "0.5,0.25",
        "--policy", "coverage_pool_c8",
        "--min-accepted", "500",
        "--max-accepted", "500",
        "--harvest-max-passes", "0",
        # Tiny mocked learners still produce the exact per-Scout contract:
        # 1 step x 500 prompts x 1 generation = 500 raw rows.
        "--prompts-per-step", "500",
        "--num-generations", "1",
        "--dataset", "dapo_math_17k",
        "--max-train-examples", "6",
        "--sync-dir", str(tmp_path / "sync"),
        "--code-source-manifest", str(code),
        "--vllm-gpus", "1",
        "--probe-every", "0",
        "--ckpt-artifact-every", "0",
        "--device", "cpu",
        "--seed", "0",
    ])


def _contract(args):
    dataset_identity = data_mod.training_dataset_identity(args.dataset)
    return mrme_contract.build_contract(
        model=args.model,
        model_revision=args.model_revision,
        dataset=args.dataset,
        dataset_identity=dataset_identity,
        max_train_examples=int(args.max_train_examples),
        rounds=2,
        scouts_per_round=2,
        lambda_schedule=(0.5, 0.25),
        base_seed=0,
        scout_steps=1,
        central_steps=1,
        prompts_per_step=500,
        num_generations=1,
        pooling_policy="coverage_pool_c8",
        max_accepted=500,
        harvest_max_passes=0,
        training_contract=pipeline_gpu._resolved_mrme_training_contract(
            args, [0.5, 0.25]),
        code_identity={"test": "code"},
    )


def _c9_multi_args(tmp_path: Path, *, rounds: int, scouts: int):
    code = tmp_path / "c9-code-manifest.json"
    code.write_text("{}\n", encoding="utf-8")
    schedule = "0.5" if rounds == 1 else ",".join(
        "0.5" if index == 0 else "0.25" for index in range(rounds))
    args = pipeline_gpu.build_arg_parser().parse_args([
        "--model", "Qwen/Qwen3-1.7B",
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "--output-dir", str(tmp_path / "c9-run"),
        "--run-name", f"c9-r{rounds}-k{scouts}",
        "--rounds", str(rounds),
        "--scouts-per-round", str(scouts),
        "--scout-steps", "200",
        "--central-steps", "100",
        "--lambda-schedule", schedule,
        "--policy", "coverage_pool_c8",
        "--min-accepted", "500",
        "--max-accepted", "500",
        "--harvest-max-passes", "0",
        "--prompts-per-step", "4",
        "--num-generations", "16",
        "--dataset", "dapo_math_17k",
        "--max-train-examples", "20000",
        "--sync-dir", str(tmp_path / "c9-sync"),
        "--code-source-manifest", str(code),
        "--vllm-gpus", "1,2,3,4,5,6,7",
        "--vllm-max-model-len", "36864",
        "--probe-every", "25",
        "--reload-every", "1",
        "--dynamic-max-attempts", "0",
        "--expdis-stabilization-mode", "c9_selected_batch_update",
        "--novelty-normalization", "selected_batch_zscore",
        "--scientific-variant",
        mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "--c9-multiexplorer-profile",
        mrme_contract.C9_MULTIEXPLORER_PROFILE,
        "--wandb",
        "--scout-lr", "1e-6",
        "--seed", "0",
    ])
    code_ref = "entity/project/c9-code:v32"
    code_digest = "abc"
    receipt = receipt_fixtures.binding(
        run_name=args.run_name, code_ref=code_ref, code_digest=code_digest)
    args.orchestrator_command_sha256 = receipt[
        "receipt"]["job"]["command_sha256"]
    args.orchestrator_receipt_artifact = receipt[
        "receipt"]["job"]["environment"]["ORCHESTRATOR_RECEIPT_ARTIFACT"]
    args._code_artifact_binding = {
        "source": {"artifact_ref": code_ref, "artifact_digest": code_digest},
    }
    args._orchestrator_receipt_binding = receipt
    args._training_execution_environment = receipt_fixtures.environment(
        run_name=args.run_name, code_ref=code_ref, code_digest=code_digest)
    return args


@pytest.mark.parametrize(
    ("rounds", "scouts", "method", "seeds"),
    [
        (1, 3, mrme_contract.INTEGRATED_MULTISCOUT_METHOD, [[0, 1, 2]]),
        (2, 2, mrme_contract.DISJOINT_SHARD_MRME_METHOD,
         [[0, 1], [2, 3]]),
    ],
)
def test_c9_multi_explorer_true_contract_profiles(
    tmp_path, monkeypatch, rounds, scouts, method, seeds,
):
    args = _c9_multi_args(tmp_path, rounds=rounds, scouts=scouts)
    pipeline_gpu.validate_args(args)
    code_identity = {
        "source": {
            "artifact_ref": "entity/project/c9-code:v32",
            "artifact_digest": "abc",
        },
        "files": [], "file_manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        pipeline_gpu, "_validated_executing_code_source_manifest",
        lambda _args, *, profile_label: code_identity)
    contract = pipeline_gpu._build_mrme_contract(
        args, pipeline_gpu.parse_lambda_schedule(args))
    assert contract["method"] == method
    assert contract["stabilization_profile"] == \
        mrme_contract.C9_MULTIEXPLORER_PROFILE
    assert contract["scientific_variant"] == \
        mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT
    assert [[row["seed"] for row in spec["scouts"]]
            for spec in contract["rounds_spec"]] == seeds
    assert contract["scout_steps_per_scout"] == 200
    assert contract["expected_raw_rows_per_scout"] == 12_800
    assert contract["training_contract"]["rnd_reset"] == "fresh_per_scout"
    assert contract["training_contract"]["scout_grpo"][
        "scientific_variant"] == \
        mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT
    assert contract["training_contract"]["scout_grpo"][
        "scout_novelty_contract"]["rnd_lifecycle"] == \
        "c9_selected_batch_update"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scout_lr", 5e-6, "scout-lr 1e-6"),
        ("novelty_normalization", "raw", "contract drift"),
        ("expdis_stabilization_mode", "c8_candidate_attempt_update",
         "contract drift"),
        ("scientific_variant", "expdis_c9_selected_batch_rnd",
         "contract drift"),
        ("c9_execution_profile", "full", "incompatible options"),
        ("wandb", False, "online --wandb"),
    ],
)
def test_c9_multi_explorer_runtime_drift_fails_closed(
    tmp_path, field, value, message,
):
    args = _c9_multi_args(tmp_path, rounds=2, scouts=2)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        pipeline_gpu.validate_args(args)


def _round_manifest(args, contract, round_index: int):
    path = (
        Path(args.output_dir) / f"round_{round_index:02d}"
        / "dataset_shard_manifest.json")
    env = mrme_contract.shard_environment(contract, round_index, str(path))
    rows = [{"problem": f"p{i}", "answer": str(i)} for i in range(6)]
    with pipeline_gpu._temporary_environment(env):
        _selected, manifest = data_mod.build_round_shard(rows)
    return manifest


def test_pipeline_builds_and_binds_resolved_mrme_contract(tmp_path, monkeypatch):
    args = _args(tmp_path)
    pipeline_gpu.validate_args(args)
    code_identity = {"source": {"artifact_ref": "code:v1"}, "sha256": "abc"}
    monkeypatch.setattr(
        pipeline_gpu, "_validated_executing_code_source_manifest",
        lambda _args, *, profile_label: code_identity)
    contract = pipeline_gpu._build_mrme_contract(
        args, pipeline_gpu.parse_lambda_schedule(args))
    assert contract["rounds_spec"][0]["scouts"][0]["seed"] == 0
    assert contract["rounds_spec"][1]["scouts"][1]["seed"] == 3
    assert contract["training_contract"]["rnd_reset"] == "fresh_per_scout"
    assert contract["training_contract"]["central_sft_init"] == \
        "base_model_every_round"
    assert contract["training_contract"]["scout_grpo"][
        "prompts_per_step"] == 500

    os.makedirs(args.output_dir)
    state = pipeline_gpu._prepare_mrme_state(args, {"stages": {}}, contract)
    assert state["mrme_binding"]["contract_sha256"] == contract[
        "contract_sha256"]
    assert json.loads((Path(args.output_dir) / "mrme_contract.json").read_text()) \
        == contract

    changed = copy.deepcopy(contract)
    changed["base_seed"] = 9
    raw = {k: v for k, v in changed.items() if k != "contract_sha256"}
    changed["contract_sha256"] = mrme_contract.canonical_json_sha256(raw)
    args.resume = True
    with pytest.raises(ValueError, match="not canonical|contract mismatch"):
        pipeline_gpu._prepare_mrme_state(args, state, changed)


def test_mrme_resume_refuses_orphaned_round_outputs(tmp_path):
    args = _args(tmp_path)
    args.resume = True
    contract = _contract(args)
    round_dir = Path(args.output_dir) / "round_01"
    round_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="without pipeline_state"):
        pipeline_gpu._prepare_mrme_state(args, {"stages": {}}, contract)


def test_production_launcher_forwards_and_publishes_mrme_contract():
    launcher = Path(pipeline_gpu.__file__).parent / "launch" / "expdis_job.sh"
    source = launcher.read_text(encoding="utf-8")
    assert '--scouts-per-round "$SCOUTS_PER_ROUND"' in source
    assert 'MRME_CODE_ARGS=(--code-source-manifest "$CODE_SOURCE_MANIFEST")' \
        in source
    assert "mrme_contract.validate_completed_campaign(out_dir)" in source
    assert '"mrme_completed_campaign.json"' in source


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--rounds", "1", "integrated MultiScout"),
        ("--policy", "quality_pool", "contract drift"),
        ("--harvest-max-passes", "1", "contract drift"),
        ("--min-accepted", "499", "contract drift"),
        ("--endpoints", "http://stale:8000", "incompatible options"),
        ("--vllm-gpus", "", "managed --vllm-gpus"),
    ],
)
def test_true_mrme_rejects_semantic_drift(tmp_path, flag, value, message):
    args = _args(tmp_path)
    setattr(args, flag.removeprefix("--").replace("-", "_"), (
        int(value) if value.isdigit() else value))
    with pytest.raises(ValueError, match=message):
        pipeline_gpu.validate_args(args)


def test_true_mrme_two_by_two_end_to_end_and_resume(tmp_path, monkeypatch):
    args = _args(tmp_path)
    pipeline_gpu.validate_args(args)
    contract = _contract(args)
    args._mrme_contract = contract
    os.makedirs(args.output_dir)
    state = mrme_contract.bind_pipeline_state({"stages": {}}, contract)
    common.atomic_write_json(
        os.path.join(args.output_dir, "pipeline_state.json"), state)
    pool = FakeManagedPool(args.model)
    metrics = common.MetricsLogger(os.path.join(args.output_dir, "metrics.jsonl"))
    calls = []
    sft_inits = []

    def fake_grpo(gargs, pool):
        assert os.environ["TMX_DATASET_SHARD_REQUIRED"] == "1"
        assert int(os.environ["TMX_DATASET_SHARD_INDEX"]) == int(gargs.round) - 1
        assert os.environ["TMX_DATASET_SHARD_COUNT"] == "2"
        calls.append((
            str(gargs.phase), int(gargs.round), int(gargs.seed), str(gargs.model)))
        if gargs.phase == "scout":
            _write_full_scout_trajectories(gargs)
        return _write_deployable_grpo_final(
            gargs,
            code_identity=(contract["code_identity"]
                           if gargs.phase == "scout" else None),
        )

    monkeypatch.setattr(pipeline_gpu.grpo_gpu, "run_grpo", fake_grpo)
    def fake_sft(sargs):
        sft_inits.append(str(sargs.model))
        return _fake_sft(sargs)

    monkeypatch.setattr(pipeline_gpu.sft_gpu, "run_sft", fake_sft)

    summaries = []
    scout_init = args.model
    manifests = []
    for round_index in (1, 2):
        manifest = _round_manifest(args, contract, round_index)
        manifests.append(manifest)
        env = mrme_contract.shard_environment(
            contract,
            round_index,
            str(Path(args.output_dir) / f"round_{round_index:02d}"
                / "dataset_shard_manifest.json"),
        )
        with pipeline_gpu._temporary_environment(env):
            summary = pipeline_gpu.run_mrme_round(
                args, contract, pool, tokenizer=None, examples=[],
                round_idx=round_index, scout_init=scout_init,
                shard_manifest=manifest, wandb_run=None,
                metrics_logger=metrics, state=state)
        summaries.append(summary)
        scout_init = summary["central_final"]

    assert [(phase, round_index, seed) for phase, round_index, seed, _ in calls] == [
        ("scout", 1, 0), ("scout", 1, 1), ("central", 1, 0),
        ("scout", 2, 2), ("scout", 2, 3), ("central", 2, 1),
    ]
    assert [model for phase, round_index, _, model in calls
            if phase == "scout" and round_index == 1] == [args.model, args.model]
    assert [model for phase, round_index, _, model in calls
            if phase == "scout" and round_index == 2] == [
                summaries[0]["central_final"], summaries[0]["central_final"]]
    assert sft_inits == [args.model, args.model]
    assert summaries[1]["scout_init"] == summaries[0]["central_final"]
    assert all(summary["scouts_per_round"] == 2 for summary in summaries)
    assert all(summary["funnel"]["raw"] == 1_000 for summary in summaries)
    assert all(summary["funnel"]["accepted"] == 500 for summary in summaries)
    assert mrme_contract.assert_round_manifests_disjoint(
        manifests, expected_rounds=2)["counts"] == [3, 3]
    expected_stages = [
        f"r{round_index:02d}/{stage}"
        for round_index in (1, 2)
        for stage in (
            "scout_01", "scout_02", "harvest", "filter", "sft", "central")
    ]
    assert list(state["stages"]) == expected_stages

    # Full round resume must revalidate every raw source/SFT/Central byte and
    # perform no training or filtering work again.
    monkeypatch.setattr(
        pipeline_gpu.grpo_gpu, "run_grpo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GRPO reran during resume")))
    monkeypatch.setattr(
        pipeline_gpu.sft_gpu, "run_sft",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("SFT reran during resume")))
    with pipeline_gpu._temporary_environment(mrme_contract.shard_environment(
        contract, 2,
        str(Path(args.output_dir) / "round_02" / "dataset_shard_manifest.json"),
    )):
        resumed = pipeline_gpu.run_mrme_round(
            args, contract, pool, tokenizer=None, examples=[], round_idx=2,
            scout_init=summaries[0]["central_final"],
            shard_manifest=manifests[1], wandb_run=None,
            metrics_logger=metrics, state=state)
    assert resumed["central_final"] == summaries[1]["central_final"]

    # Recreate the same durable objects written by main(), then require the
    # publication guard to traverse and re-hash the complete K x R chain.
    common.atomic_write_json(
        os.path.join(args.output_dir, "mrme_contract.json"), contract)
    partition = mrme_contract.assert_round_manifests_disjoint(
        manifests, expected_rounds=2)
    common.atomic_write_json(
        os.path.join(args.output_dir, "mrme_shard_partition.json"), partition)
    common.atomic_write_json(
        os.path.join(args.output_dir, "run_summary.json"),
        {
            "rounds": summaries,
            "final_model": summaries[-1]["central_final"],
            **pipeline_gpu._round_campaign_summary_identity(contract),
            "mrme_contract_sha256": contract["contract_sha256"],
            "mrme_shard_partition": partition,
        },
    )
    evidence = mrme_contract.validate_completed_campaign(args.output_dir)
    assert evidence["rounds"] == 2
    assert evidence["scouts_per_round"] == 2
    assert evidence["final_model"] == os.path.realpath(
        summaries[-1]["central_final"])

    with open(Path(args.output_dir) / "round_02" / "accepted.jsonl", "a",
              encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="deterministic coverage_pool_c8 replay"):
        mrme_contract.validate_completed_campaign(args.output_dir)
