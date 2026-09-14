"""Focused, CPU-only tests for MR-ME orchestration and exact data shards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmx_gpu import common, filter_pool, mrme_contract
from tmx_jax import data as data_mod


def _portable_fixture(tmp_path, monkeypatch):
    package_root = tmp_path / "downloaded-results"
    package_root.mkdir()
    allowed_parent = tmp_path / "restore-parent"
    allowed_parent.mkdir()
    recorded_root = allowed_parent / "campaign"
    completion = {
        "version": "fixture-completion-v1",
        "contract_sha256": "e" * 64,
        "partition_sha256": "f" * 64,
        "evidence_sha256": "a" * 64,
    }
    common.atomic_write_json(
        str(package_root / "mrme_completed_campaign.json"), completion)
    (package_root / "marker.txt").write_text("portable evidence\n", encoding="utf-8")

    remote_root = tmp_path / "remote-artifact"
    remote_model = remote_root / "model"
    remote_model.mkdir(parents=True)
    (remote_model / "config.json").write_text("{}\n", encoding="utf-8")
    (remote_model / "model.safetensors").write_bytes(b"model-bytes")
    manifest = common.directory_file_manifest(str(remote_model))
    metadata = {
        "round": 1,
        "stage_mode": "c9_multi_explorer_fixed_endpoint_trained",
        "model_file_manifest_sha256": manifest["sha256"],
        "accepted_file_sha256": "b" * 64,
        "training_sequence_sha256": "c" * 64,
    }
    binding = {
        "version": "pipeline_sft_model_artifact_v1",
        "durable_wandb_upload": True,
        "source": {
            "artifact_ref": "entity/project/portable-sft:v3",
            "artifact_digest": "remote-digest",
        },
        "artifact_type": "sft-model",
        "artifact_metadata": metadata,
        "artifact_subdir": "model",
        "model_file_manifest": manifest,
        "policy_file_manifest": manifest,
        "accepted_library_digest": {"row_count": 500, "sha256": "d" * 64},
        "accepted_file_sha256": "b" * 64,
        "training_sequence_digest": {"sha256": "c" * 64},
        "round": 1,
        "stage_mode": "c9_multi_explorer_fixed_endpoint_trained",
    }
    binding["binding_sha256"] = common.canonical_json_sha256(binding)
    files = []
    for relative in ("marker.txt", "mrme_completed_campaign.json"):
        path = package_root / relative
        files.append({
            "relative_path": relative,
            "sha256": common.file_sha256(str(path)),
            "size_bytes": path.stat().st_size,
        })
    logical_path = recorded_root / "round_01" / "sft" / "actual_sft_final"
    package = mrme_contract._signed_payload(
        mrme_contract.MRME_PORTABLE_CAMPAIGN_VERSION,
        {
            "recorded_output_root": str(recorded_root),
            "contract_sha256": "e" * 64,
            "completion": completion,
            "completion_evidence_sha256": completion["evidence_sha256"],
            "partition_sha256": "f" * 64,
            "files": files,
            "model_artifacts": [{
                "role": "sft",
                "round_index": 1,
                "scout_index": None,
                "logical_path": str(logical_path),
                "relative_logical_path":
                    "round_01/sft/actual_sft_final",
                "artifact_subdir": "model",
                "artifact_type": "sft-model",
                "artifact_ref": "entity/project/portable-sft:v3",
                "artifact_digest": "remote-digest",
                "artifact_metadata": metadata,
                "model_file_manifest": manifest,
                "binding": binding,
            }],
        },
    )
    common.atomic_write_json(
        str(package_root / "mrme_portable_campaign.json"), package)

    def resolver(_record, _destination):
        return {
            "download_root": str(remote_root),
            "artifact_ref": "entity/project/portable-sft:v3",
            "artifact_digest": "remote-digest",
            "artifact_type": "sft-model",
            "artifact_metadata": metadata,
        }

    def validate_restored(root):
        assert Path(root) == recorded_root
        assert (recorded_root / "marker.txt").read_text(
            encoding="utf-8") == "portable evidence\n"
        common.verify_directory_file_manifest(str(logical_path), manifest)
        return completion

    monkeypatch.setattr(
        mrme_contract, "validate_completed_campaign", validate_restored)
    return {
        "package_root": package_root,
        "allowed_parent": allowed_parent,
        "recorded_root": recorded_root,
        "remote_model": remote_model,
        "resolver": resolver,
        "completion": completion,
    }


def test_portable_campaign_rehydrates_exact_model_and_cleans_up(
    tmp_path, monkeypatch,
):
    fixture = _portable_fixture(tmp_path, monkeypatch)
    completed = mrme_contract.validate_portable_completed_campaign(
        str(fixture["package_root"]),
        artifact_resolver=fixture["resolver"],
        allowed_output_parent=str(fixture["allowed_parent"]))
    assert completed == fixture["completion"]
    assert not fixture["recorded_root"].exists()


def test_portable_campaign_rejects_missing_support_file(
    tmp_path, monkeypatch,
):
    fixture = _portable_fixture(tmp_path, monkeypatch)
    (fixture["package_root"] / "marker.txt").unlink()
    with pytest.raises(ValueError, match="thin evidence file drifted"):
        mrme_contract.validate_portable_completed_campaign(
            str(fixture["package_root"]),
            artifact_resolver=fixture["resolver"],
            allowed_output_parent=str(fixture["allowed_parent"]))


def test_portable_campaign_rejects_tampered_model_bytes(
    tmp_path, monkeypatch,
):
    fixture = _portable_fixture(tmp_path, monkeypatch)
    (fixture["remote_model"] / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="directory file manifest mismatch"):
        mrme_contract.validate_portable_completed_campaign(
            str(fixture["package_root"]),
            artifact_resolver=fixture["resolver"],
            allowed_output_parent=str(fixture["allowed_parent"]))


def test_portable_campaign_rejects_wrong_remote_digest(
    tmp_path, monkeypatch,
):
    fixture = _portable_fixture(tmp_path, monkeypatch)

    def wrong_digest(record, destination):
        resolved = dict(fixture["resolver"](record, destination))
        resolved["artifact_digest"] = "different"
        return resolved

    with pytest.raises(ValueError, match="resolved artifact identity drifted"):
        mrme_contract.validate_portable_completed_campaign(
            str(fixture["package_root"]), artifact_resolver=wrong_digest,
            allowed_output_parent=str(fixture["allowed_parent"]))


def build_contract(**overrides):
    values = {
        "model": "Qwen/Qwen3-1.7B",
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "dataset": "dapo_math_17k",
        "dataset_identity": data_mod.training_dataset_identity(
            "dapo_math_17k"),
        "max_train_examples": 20_000,
        "rounds": 4,
        "scouts_per_round": 3,
        "lambda_schedule": (0.75, 0.5, 0.35, 0.25),
        "base_seed": 7,
        "scout_steps": 200,
        "central_steps": 100,
        "prompts_per_step": 4,
        "num_generations": 16,
        "pooling_policy": "coverage_pool_c8",
        "max_accepted": 500,
        "harvest_max_passes": 0,
        "training_contract": {
            "expdis_stabilization_mode": "selected_batch_once_per_learner_step",
            "novelty_normalization": "selected_batch_zscore",
            "scout_lr": 1e-6,
            "sft_epochs": 2,
            "sft_grad_accum": 1,
        },
        "code_identity": {"artifact": "code:v1", "digest": "abc"},
    }
    values.update(overrides)
    return mrme_contract.build_contract(**values)


class TestMRMEContract:
    def test_c9_integrated_r1_k3_is_unsharded_and_distinct(self, tmp_path):
        contract = build_contract(
            rounds=1,
            scouts_per_round=3,
            lambda_schedule=(0.5,),
            base_seed=0,
            stabilization_profile=mrme_contract.C9_MULTIEXPLORER_PROFILE,
        )
        assert contract["method"] == mrme_contract.INTEGRATED_MULTISCOUT_METHOD
        assert contract["dataset_membership_mode"] == "full_unsharded"
        assert contract["method_is_failure_targeted_curriculum"] is False
        assert contract["scientific_variant"] == \
            mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT
        assert [row["seed"] for row in contract["rounds_spec"][0]["scouts"]] \
            == [0, 1, 2]
        env = mrme_contract.shard_environment(
            contract, 1, str(tmp_path / "full-membership.json"))
        assert env["TMX_DATASET_SHARD_REQUIRED"] == "0"
        assert env["TMX_DATASET_SHARD_COUNT"] == "1"
        assert env["TMX_DATASET_SHARD_INDEX"] == "0"
        rows = [
            {"problem": f"problem-{index}", "answer": str(index)}
            for index in range(7)
        ]
        with pytest.MonkeyPatch.context() as monkeypatch:
            for name, value in env.items():
                monkeypatch.setenv(name, value)
            selected, manifest = data_mod.build_round_shard(rows)
        assert selected == rows
        assert manifest["algorithm"] == "identity_no_shard"
        assert manifest["required"] is False
        assert manifest["num_rows"] == len(rows)
        assert json.loads((tmp_path / "full-membership.json").read_text()) \
            == manifest
        partition = mrme_contract.assert_round_manifests_disjoint(
            [manifest], expected_rounds=1)
        assert partition["method"] == mrme_contract.INTEGRATED_MULTISCOUT_METHOD
        assert partition["dataset_membership_mode"] == "full_unsharded"
        assert partition["counts"] == [len(rows)]
        assert contract["external_publication_evaluation"]["required"] is True
        assert contract["external_publication_evaluation"][
            "training_completion_is_headline_eligible"] is False

    def test_c9_r2_k2_is_disjoint_shard_annealed_not_failure_targeted(self):
        contract = build_contract(
            rounds=2,
            scouts_per_round=2,
            lambda_schedule=(0.5, 0.25),
            base_seed=0,
            stabilization_profile=mrme_contract.C9_MULTIEXPLORER_PROFILE,
        )
        assert contract["method"] == mrme_contract.DISJOINT_SHARD_MRME_METHOD
        assert contract["dataset_membership_mode"] == \
            "deterministic_disjoint_one_over_r"
        assert contract["round_assignment_signal"] == \
            "predeclared_round_index_only"
        assert contract["method_is_failure_targeted_curriculum"] is False
        assert [[scout["seed"] for scout in row["scouts"]]
                for row in contract["rounds_spec"]] == [[0, 1], [2, 3]]

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"rounds": 1, "lambda_schedule": (0.75,),
              "stabilization_profile":
                  mrme_contract.C9_MULTIEXPLORER_PROFILE}, "lambda=0.5"),
            ({"rounds": 2, "lambda_schedule": (0.25, 0.5),
              "stabilization_profile":
                  mrme_contract.C9_MULTIEXPLORER_PROFILE}, "strictly decreasing"),
            ({"rounds": 2, "lambda_schedule": (0.5, 0.5),
              "stabilization_profile":
                  mrme_contract.C9_MULTIEXPLORER_PROFILE}, "strictly decreasing"),
            ({"rounds": 1, "lambda_schedule": (0.5,),
              "stabilization_profile": "unknown"}, "unknown"),
        ],
    )
    def test_c9_multi_explorer_identity_drift_fails_closed(
        self, overrides, message,
    ):
        with pytest.raises(ValueError, match=message):
            build_contract(**overrides)

    def test_true_mrme_rounds_scouts_and_seed_identity(self):
        contract = build_contract()
        assert mrme_contract.validate_contract(contract) == contract
        assert contract["rounds"] == 4
        assert contract["scouts_per_round"] == 3
        assert contract["pooling_role"] == "quality_pool"
        assert contract["pooling_policy"] == "coverage_pool_c8"
        assert contract["pooling_scope"] == "one_union_of_all_k_raw_scout_sources_per_round"
        assert contract["per_scout_preselection"] is False
        assert contract["harvest_or_topup"] is False
        assert contract["expected_raw_rows_per_scout"] == 12_800
        assert contract["rollout_budget_mode"] == "full_per_scout_per_round"
        assert contract["shard_changes_rollout_budget"] is False

        # Seed 0 is the preregistered SingleScout; seeds 1/2 are the added K=3
        # components. Later rounds get non-overlapping seed blocks so RND and
        # data-order RNG initializations are fresh for every (round, Scout).
        assert [r["scouts"][0]["seed"] for r in contract["rounds_spec"]] == [
            7,
            10,
            13,
            16,
        ]
        assert [s["seed"] for s in contract["rounds_spec"][0]["scouts"]] == [7, 8, 9]
        assert [s["seed"] for s in contract["rounds_spec"][1]["scouts"]] == [10, 11, 12]
        assert contract["rounds_spec"][2]["shard_fraction_denominator"] == 4
        assert mrme_contract.expected_scout_stage_keys(contract)[-1] == "r04/scout_03"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"rounds": 1, "lambda_schedule": (0.5,)}, "integrated R=1"),
            ({"scouts_per_round": 0}, "scouts-per-round >= 1"),
            ({"pooling_policy": "quality_pool"}, "concretely bound"),
            ({"harvest_max_passes": 1}, "harvest/top-up is forbidden"),
            ({"max_accepted": 501}, "global QualityPool cap"),
        ],
    )
    def test_fail_closed_design(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            build_contract(**overrides)

    def test_json_round_trip_and_tamper_detection(self):
        contract = build_contract()
        loaded = json.loads(json.dumps(contract))
        assert mrme_contract.validate_contract(loaded) == loaded
        loaded["rounds_spec"][1]["scouts"][0]["seed"] += 1
        loaded["contract_sha256"] = mrme_contract.canonical_json_sha256(
            {k: v for k, v in loaded.items() if k != "contract_sha256"}
        )
        with pytest.raises(ValueError, match="not canonical"):
            mrme_contract.validate_contract(loaded)

    def test_resume_state_cannot_be_relabelled_or_mixed(self):
        contract = build_contract()
        state = {"stages": {}}
        mrme_contract.bind_pipeline_state(state, contract)
        assert state["mrme_binding"]["contract_sha256"] == contract["contract_sha256"]
        assert mrme_contract.bind_pipeline_state(state, contract) is state

        changed = build_contract(base_seed=8)
        with pytest.raises(ValueError, match="contract mismatch"):
            mrme_contract.bind_pipeline_state(state, changed)

        with pytest.raises(ValueError, match="non-empty unbound"):
            mrme_contract.bind_pipeline_state(
                {"stages": {"r01/scout": {"final": "legacy"}}}, contract
            )

    def test_execution_plan_has_exact_seed_grid_and_isolated_paths(self, tmp_path):
        contract = build_contract(base_seed=0)
        plans = mrme_contract.scout_execution_plan(
            contract,
            1,
            output_dir=str(tmp_path / "run"),
            sync_dir=str(tmp_path / "sync"),
            run_name="experiment",
            scout_init="Qwen/Qwen3-1.7B",
        )
        assert [plan["seed"] for plan in plans] == [0, 1, 2]
        assert [plan["scout_index"] for plan in plans] == [1, 2, 3]
        assert len({plan["output_dir"] for plan in plans}) == 3
        assert len({plan["sync_dir"] for plan in plans}) == 3
        assert all(plan["expected_raw_rows"] == 12_800 for plan in plans)
        assert all(plan["scout_init_identity"] == plans[0]["scout_init_identity"] for plan in plans)

    def test_quality_pool_replay_rejects_arbitrary_valid_subset(self, tmp_path):
        sources = [tmp_path / "scout-1.jsonl", tmp_path / "scout-2.jsonl"]
        raw_rows = []
        for source_index, source in enumerate(sources):
            rows = []
            for index in range(300):
                value = source_index * 300 + index
                rows.append({
                    "problem_id": f"p-{value}",
                    "prompt_text": f"problem {value}",
                    "completion_text": f"work \\boxed{{{value}}}",
                    "ground_truth": str(value),
                    "is_correct": True,
                    "valid_answer": True,
                    "terminated": True,
                    "clipped": False,
                    "completion_token_length": 4,
                })
            raw_rows.extend((str(source), row) for row in rows)
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
        accepted = tmp_path / "accepted.jsonl"
        _path, funnel = filter_pool.pool_trajectories(
            [str(path) for path in sources], policy=filter_pool.C8_POLICY,
            max_examples=500, output=str(accepted), source_balance=False)
        assert mrme_contract.replay_coverage_pool_c8(
            [str(path) for path in sources], accepted_path=str(accepted),
            recorded_funnel=funnel, max_examples=500,
            max_completion_tokens=filter_pool.MAX_COMPLETION_TOKENS)[
                "funnel"] == funnel

        selected = [json.loads(line) for line in accepted.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        selected_ids = {row["problem_id"] for row in selected}
        source_path, replacement = next(
            (path, row) for path, row in raw_rows
            if row["problem_id"] not in selected_ids)
        selected[-1] = {
            "prompt_text": replacement["prompt_text"],
            "completion_text": replacement["completion_text"],
            **{key: replacement[key] for key in filter_pool.METADATA_KEYS
               if key in replacement},
            "source_file": source_path,
            "selection_policy": filter_pool.C8_POLICY,
            "selection_source_balance": False,
        }
        accepted.write_text(
            "".join(json.dumps(row) + "\n" for row in selected),
            encoding="utf-8")
        with pytest.raises(ValueError, match="deterministic coverage_pool_c8"):
            mrme_contract.replay_coverage_pool_c8(
                [str(path) for path in sources], accepted_path=str(accepted),
                recorded_funnel=funnel, max_examples=500,
                max_completion_tokens=filter_pool.MAX_COMPLETION_TOKENS)


class TestExactRoundShards:
    @staticmethod
    def rows(n=17):
        return [{"id": f"p{i}", "problem": f"problem {i}", "answer": str(i)} for i in range(n)]

    def test_four_disjoint_one_over_r_shards_and_exact_manifests(
        self,
        tmp_path,
        monkeypatch,
    ):
        rows = self.rows()
        manifests = []
        selected_id_sets = []
        for round_index in range(4):
            manifest_path = tmp_path / f"round_{round_index + 1:02d}.json"
            monkeypatch.setenv("TMX_DATASET_SHARD_REQUIRED", "1")
            monkeypatch.setenv("TMX_DATASET_SHARD_INDEX", str(round_index))
            monkeypatch.setenv("TMX_DATASET_SHARD_COUNT", "4")
            monkeypatch.setenv("TMX_DATASET_SHARD_POLICY", "uniform")
            monkeypatch.setenv("TMX_DATASET_SHARD_SEED", "7")
            monkeypatch.setenv("TMX_DATASET_SHARD_MANIFEST_PATH", str(manifest_path))
            selected, manifest = data_mod.build_round_shard(rows)
            # A second Scout in the same round must resolve and verify the
            # byte-identical shared membership manifest.
            again, manifest_again = data_mod.build_round_shard(rows)
            assert again == selected
            assert manifest_again == manifest
            with manifest_path.open(encoding="utf-8") as handle:
                assert json.load(handle) == manifest
            assert data_mod.validate_round_shard_manifest(manifest) == manifest
            manifests.append(manifest)
            selected_id_sets.append({row["id"] for row in selected})

        assert [len(items) for items in selected_id_sets] == [4, 4, 4, 5]
        assert all(
            selected_id_sets[i].isdisjoint(selected_id_sets[j])
            for i in range(4)
            for j in range(i + 1, 4)
        )
        assert set().union(*selected_id_sets) == {row["id"] for row in rows}
        partition = mrme_contract.assert_round_manifests_disjoint(manifests, expected_rounds=4)
        assert partition["counts"] == [4, 4, 4, 5]
        assert partition["total_rows"] == 17

    def test_required_shard_has_no_silent_fallback(self, monkeypatch):
        monkeypatch.setenv("TMX_DATASET_SHARD_REQUIRED", "1")
        for name in (
            "TMX_DATASET_SHARD_INDEX",
            "TMX_DATASET_SHARD_COUNT",
            "TMX_DATASET_SHARD_POLICY",
            "TMX_DATASET_SHARD_SEED",
            "TMX_DATASET_SHARD_MANIFEST_PATH",
        ):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(ValueError, match="environment is incomplete"):
            data_mod.build_round_shard(self.rows())

    def test_existing_manifest_rejects_changed_membership(
        self,
        tmp_path,
        monkeypatch,
    ):
        path = tmp_path / "shard.json"
        settings = {
            "TMX_DATASET_SHARD_REQUIRED": "1",
            "TMX_DATASET_SHARD_INDEX": "0",
            "TMX_DATASET_SHARD_COUNT": "2",
            "TMX_DATASET_SHARD_POLICY": "uniform",
            "TMX_DATASET_SHARD_SEED": "5",
            "TMX_DATASET_SHARD_MANIFEST_PATH": str(path),
        }
        for key, value in settings.items():
            monkeypatch.setenv(key, value)
        data_mod.build_round_shard(self.rows())
        changed = self.rows()
        changed[0]["answer"] = "different"
        with pytest.raises(ValueError, match="differs from the existing"):
            data_mod.build_round_shard(changed)

    def test_single_round_order_remains_unchanged(self, monkeypatch):
        for name in (
            "TMX_DATASET_SHARD_REQUIRED",
            "TMX_DATASET_SHARD_INDEX",
            "TMX_DATASET_SHARD_COUNT",
            "TMX_DATASET_SHARD_POLICY",
            "TMX_DATASET_SHARD_SEED",
            "TMX_DATASET_SHARD_MANIFEST_PATH",
        ):
            monkeypatch.delenv(name, raising=False)
        rows = self.rows()
        assert data_mod._apply_round_shard(rows) == rows


class TestScoutSourceHandoff:
    def _trajectory_file(self, path: Path, steps=2, rows_per_step=4):
        with path.open("w", encoding="utf-8") as handle:
            for step in range(1, steps + 1):
                for index in range(rows_per_step):
                    handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "problem_id": f"p{step}-{index}",
                                "completion_text": "answer",
                            }
                        )
                        + "\n"
                    )

    def _scout_model(
        self,
        path: Path,
        *,
        round_index: int,
        seed: int,
        lambda_novelty: float,
        steps: int,
    ) -> str:
        path.mkdir()
        (path / "config.json").write_text("{}\n")
        (path / "model.safetensors").write_bytes(b"fake-model")
        manifest = common.directory_file_manifest(
            str(path), exclude_relative_paths=("trainer_state.json",)
        )
        common.atomic_write_json(
            str(path / "trainer_state.json"),
            {
                "phase": "scout",
                "checkpoint_kind": "deployable_model",
                "round": round_index,
                "step": steps,
                "lambda": lambda_novelty,
                "runtime_contract": {
                    "round": round_index,
                    "seed": seed,
                    "lambda_novelty": lambda_novelty,
                },
                "code_artifact_binding": {
                    "artifact": "code:v1", "digest": "abc"},
                "checkpoint_file_manifest": manifest,
            },
        )
        return str(path)

    def test_exact_k_raw_sources_validate_once(self, tmp_path):
        contract = build_contract(
            rounds=2,
            scouts_per_round=2,
            lambda_schedule=(0.75, 0.5),
            scout_steps=2,
            prompts_per_step=2,
            num_generations=2,
        )
        shard = {"manifest_sha256": "a" * 64}
        init_identity = mrme_contract.model_handoff_identity(
            "Qwen/Qwen3-1.7B",
            model_revision=common.CANONICAL_TRAINING_MODEL_REVISION)
        sources = []
        for scout_index in (1, 2):
            path = tmp_path / f"scout-{scout_index}.jsonl"
            self._trajectory_file(path)
            seed = contract["rounds_spec"][0]["scouts"][scout_index - 1]["seed"]
            final = self._scout_model(
                tmp_path / f"model-{scout_index}",
                round_index=1,
                seed=seed,
                lambda_novelty=0.75,
                steps=2,
            )
            sources.append(
                mrme_contract.build_scout_source_record(
                    contract=contract,
                    round_index=1,
                    scout_index=scout_index,
                    trajectory_path=str(path),
                    shard_manifest=shard,
                    scout_init_identity=init_identity,
                    scout_final=final,
                )
            )
        validated = mrme_contract.validate_round_scout_sources(
            contract,
            1,
            list(reversed(sources)),
            expected_shard_manifest_sha256="a" * 64,
            expected_scout_init_identity=init_identity,
        )
        assert [source["scout_index"] for source in validated] == [1, 2]
        assert all(source["per_scout_preselection"] is False for source in validated)

        by_index = {source["scout_index"]: source for source in sources}
        callback_order = []

        def execute_one(plan):
            callback_order.append(plan["scout_index"])
            return by_index[plan["scout_index"]]

        executed = mrme_contract.execute_round_scouts(
            contract,
            1,
            output_dir=str(tmp_path / "run"),
            sync_dir=str(tmp_path / "sync"),
            run_name="mrme-test",
            scout_init="Qwen/Qwen3-1.7B",
            expected_shard_manifest_sha256="a" * 64,
            execute_one=execute_one,
        )
        assert callback_order == [1, 2]
        assert [source["scout_index"] for source in executed] == [1, 2]
        source_manifest = mrme_contract.build_trajectory_source_manifest(contract, 1, executed)
        assert common.verify_trajectory_source_manifest(source_manifest)["artifact_count"] == 2
        assert [
            artifact["files"][0]["local_path"] for artifact in source_manifest["artifacts"]
        ] == [source["trajectory_path"] for source in executed]

        with pytest.raises(ValueError, match="exactly 2"):
            mrme_contract.validate_round_scout_sources(
                contract,
                1,
                sources[:1],
                expected_shard_manifest_sha256="a" * 64,
                expected_scout_init_identity=init_identity,
            )

        with pytest.raises(ValueError, match="exact same round model"):
            mrme_contract.validate_round_scout_sources(
                contract,
                1,
                sources,
                expected_shard_manifest_sha256="a" * 64,
                expected_scout_init_identity=mrme_contract.model_handoff_identity(
                    "Qwen/different-init", model_revision="different-revision"
                ),
            )

    def test_rejects_partial_selected_batch_source(self, tmp_path):
        contract = build_contract(
            rounds=2,
            scouts_per_round=2,
            lambda_schedule=(0.5, 0.25),
            scout_steps=2,
            prompts_per_step=2,
            num_generations=2,
        )
        path = tmp_path / "partial.jsonl"
        self._trajectory_file(path, steps=2, rows_per_step=3)
        with pytest.raises(ValueError, match="expected exactly 8"):
            mrme_contract.build_scout_source_record(
                contract=contract,
                round_index=1,
                scout_index=1,
                trajectory_path=str(path),
                shard_manifest={"manifest_sha256": "a" * 64},
                scout_init_identity=mrme_contract.model_handoff_identity(
                    "Qwen/Qwen3-1.7B",
                    model_revision=common.CANONICAL_TRAINING_MODEL_REVISION),
                scout_final=str(tmp_path / "model"),
            )

    def test_scout_final_binds_seed_round_and_lambda(self, tmp_path):
        model = self._scout_model(
            tmp_path / "model",
            round_index=2,
            seed=1,
            lambda_novelty=0.5,
            steps=200,
        )
        identity = mrme_contract.validate_scout_final(
            model,
            expected_round=2,
            expected_seed=1,
            expected_lambda=0.5,
            expected_steps=200,
        )
        assert identity["phase"] == "scout"
        with pytest.raises(ValueError, match="runtime seed"):
            mrme_contract.validate_scout_final(
                model,
                expected_round=2,
                expected_seed=2,
                expected_lambda=0.5,
                expected_steps=200,
            )
