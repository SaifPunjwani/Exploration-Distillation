from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tmx_gpu import (
    common, fresh_c8_sft, grpo_gpu, select_sft_checkpoint, sft_gate_gpu,
)
from tmx_jax.data import training_dataset_identity
from tmx_jax.prompting import prompt_metadata


def fresh_training_evidence(accepted_path: Path | None = None) -> dict:
    scout_runtime = {
        "version": grpo_gpu.GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": "scout",
        "scientific_variant": "",
        "config_tier": "contract",
        "model_init": fresh_c8_sft.MODEL_INIT,
        "base_model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "expected_steps": 200,
        "resolved_learning_rate": 5e-6,
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
        "trajectory_record_schema_version": (
            grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION),
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
        "mask_truncated": False,
        "clip_epsilon_low": 0.2,
        "clip_epsilon_high": 0.28,
        "ppo_mode": False,
        "updates_per_rollout": 1,
        "incorrect_novelty_scale": 0.0,
        "lambda_novelty": 0.5,
        "dataset": "dapo_math_17k",
        "training_dataset_identity": training_dataset_identity(
            "dapo_math_17k"),
        "max_train_examples": 20000,
        "seed": 0,
        "round": 1,
        "max_consecutive_generation_failures": 3,
        "generation_timeout_seconds": 3600.0,
        "generation_concurrency_per_server": 8,
        "chunk_tokens": 256,
        "scout_novelty_contract": {
            "novelty_layers": "7,14,21",
            "rnd_learning_rate": 1e-4,
            "rnd_hidden_dim": 512,
            "novelty_feature_max_length": 16384,
            "incorrect_novelty_scale": 0.0,
        },
    }
    accepted_digest = (
        common.semantic_sft_jsonl_digest(str(accepted_path))
        if accepted_path is not None else {
            "version": common.SFT_DATASET_DIGEST_VERSION,
            "row_count": 500,
            "sha256": "a" * 64,
        })
    accepted_file_sha = (
        common.file_sha256(str(accepted_path))
        if accepted_path is not None else "c" * 64)
    if accepted_path is not None:
        accepted_rows = [
            json.loads(line) for line in accepted_path.read_text().splitlines()
            if line.strip()
        ]
        training_sequence = common.sft_training_sequence_digest(
            accepted_rows, epochs=2, seed=0)
    else:
        training_sequence = {
            "version": common.SFT_TRAINING_SEQUENCE_DIGEST_VERSION,
            "seed": 0,
            "epochs": 2,
            "input_row_count": 500,
            "sequence_length": 1000,
            "sha256": "9" * 64,
        }
    code_source = {
        "artifact_ref": "entity/project/code:v26",
        "artifact_digest": "code-digest",
    }
    code_files = [{
        "relative_path": "tmx_gpu/pipeline_gpu.py",
        "sha256": "d" * 64,
        "size_bytes": 10,
    }]
    code_normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": code_source,
        "files": code_files,
    }
    code_artifact = {
        **code_normalized,
        "file_manifest_sha256": common.canonical_json_sha256(code_normalized),
    }
    scout = {
        "phase": "scout",
        "checkpoint_kind": "deployable_model",
        "step": 200,
        "model": fresh_c8_sft.MODEL_INIT,
        "config_tier": "contract",
        "rl_validity_mode": "c8",
        "lambda_novelty": 0.5,
        "runtime_contract": scout_runtime,
        "checkpoint_file_manifest": {"sha256": "e" * 64},
    }
    trajectory = {
        "row_count": 12800,
        "steps": 200,
        "rows_per_step": 64,
        "file_sha256": "b" * 64,
        "semantic_digest": {
            "version": fresh_c8_sft.TRAJECTORY_SEMANTIC_DIGEST_VERSION,
            "row_count": 12800,
            "sha256": "f" * 64,
        },
    }
    completion = fresh_c8_sft.signed_payload(
        fresh_c8_sft.SCOUT_COMPLETION_EVIDENCE_VERSION,
        {
            "profile": fresh_c8_sft.PROFILE,
            "code_artifact": code_artifact,
            "trajectory": trajectory,
            "scout": scout,
        },
    )
    return fresh_c8_sft.signed_payload(
        fresh_c8_sft.TRAINING_EVIDENCE_VERSION,
        {
            "profile": fresh_c8_sft.PROFILE,
            "scout_completion_evidence": completion,
            "code_source": {
                **code_source,
                "file_manifest_sha256": code_artifact["file_manifest_sha256"],
            },
            "scout": scout,
            "trajectories": trajectory,
            "accepted_library": {
                "policy": "coverage_pool_c8",
                "row_count": 500,
                "unique_problem_count": 500,
                "file_sha256": accepted_file_sha,
                "semantic_digest": accepted_digest,
                "training_sequence_digest": training_sequence,
            },
            "filter_funnel": {"raw": 12800, "accepted": 500},
            "sft_runtime_contract": fresh_c8_sft.SFT_RUNTIME_CONTRACT,
        },
    )


def _model_file_manifest(step: int) -> dict:
    source = {
        "artifact_ref": f"entity/project/model-step-{step}:v0",
        "artifact_digest": f"model-digest-{step}",
    }
    files = [
        {"relative_path": "config.json", "sha256": "1" * 64, "size_bytes": 2},
        {"relative_path": "model.safetensors", "sha256": "2" * 64,
         "size_bytes": 10},
        {"relative_path": "sft_checkpoint.json", "sha256": "3" * 64,
         "size_bytes": 20},
    ]
    normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": source,
        "files": files,
    }
    return {
        **normalized,
        "download_root": f"/ephemeral/model-{step}",
        "file_manifest_sha256": common.canonical_json_sha256(normalized),
    }


def make_gate_bundle(
    root: Path, step: int, *, passed: bool = True, correctness: bool = True,
    accepted_digest: dict | None = None,
    trajectory_digest: dict | None = None,
    model_file_manifest: dict | None = None,
    accepted_sources: dict | None = None,
    selection_profile: str = select_sft_checkpoint.DIAGNOSTIC_PROFILE,
    training_evidence: dict | None = None,
) -> Path:
    root.mkdir(parents=True)
    heldout = []
    for i in range(32):
        problem_id = f"p{i}"
        problem = f"heldout problem {i}"
        training_prompt = f"heldout training prompt {i}"
        heldout.append({
            "problem_id": problem_id,
            "problem": problem,
            "ground_truth": "1",
            "training_prompt": training_prompt,
            "selection_digest": sft_gate_gpu._sha256(
                f"20260713\0{problem_id}\0{problem}"),
            "training_prompt_sha256": sft_gate_gpu._sha256(training_prompt),
        })
    heldout_payload = [{
        "problem_id": row["problem_id"],
        "selection_digest": row["selection_digest"],
        "training_prompt_sha256": row["training_prompt_sha256"],
    } for row in heldout]
    rollouts = []
    for mode in sft_gate_gpu.PROMPT_MODES:
        for prompt_idx, row in enumerate(heldout):
            for sample_idx in range(2):
                # One fully clipped policy is an unambiguous failing fixture.
                finish = "stop" if passed else "length"
                tokens = 100 if passed else 32768
                rollouts.append({
                    "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
                    "problem_id": row["problem_id"],
                    "prompt_mode": mode,
                    "sample_idx": sample_idx,
                    "sample_seed": 20260713 + prompt_idx * 100003 + sample_idx,
                    "finish_reason": finish,
                    "generation_error_detail": None,
                    "completion_tokens": tokens,
                    "completion_text": "<think>ok</think> \\boxed{1}",
                    "correctness_diagnostic_only": correctness,
                    # Deliberately wrong derived fields: selector must ignore them.
                    "terminated": not passed,
                    "clipped": passed,
                })
    metrics, gate = sft_gate_gpu.recompute_gate_from_rollouts(
        rollouts,
        num_prompts=32,
        samples_per_prompt=2,
        generation_seed=20260713,
        max_completion_tokens=32768,
        ordered_problem_ids=[row["problem_id"] for row in heldout],
        thresholds=select_sft_checkpoint.FROZEN_THRESHOLDS,
    )
    accepted_digest = accepted_digest or {
        "version": common.SFT_DATASET_DIGEST_VERSION,
        "row_count": 500,
        "sha256": "a" * 64,
    }
    trajectory_digest = trajectory_digest or {
        "version": common.TRAJECTORY_SOURCE_DIGEST_VERSION,
        "artifact_count": 1,
        "sha256": "b" * 64,
    }
    contract = (select_sft_checkpoint.DIAGNOSTIC_CONTRACT
                if selection_profile == select_sft_checkpoint.DIAGNOSTIC_PROFILE
                else fresh_c8_sft.CHECKPOINT_CONTRACT)
    metadata = {
        "checkpoint_kind": "post_optimizer_step_hf_model",
        "optimizer_step": step,
        **copy.deepcopy(contract),
        "accepted_library_digest": accepted_digest,
    }
    if selection_profile == fresh_c8_sft.PROFILE:
        training_evidence = copy.deepcopy(
            training_evidence or fresh_training_evidence())
        metadata["fresh_c8_training_evidence"] = training_evidence
        metadata["accepted_library_digest"] = training_evidence[
            "accepted_library"]["semantic_digest"]
        metadata["completed_run_evidence"] = fresh_c8_sft.build_completed_run_evidence(
            optimizer_steps=1000, micro_steps=1000,
            saved_steps=list(fresh_c8_sft.PREREGISTERED_STEPS),
            training_evidence_sha256=training_evidence["evidence_sha256"],
        )
        accepted_digest = metadata["accepted_library_digest"]
    else:
        metadata["trajectory_source"] = {
            "digest": trajectory_digest, "manifest": {}}
    if accepted_sources is None:
        accepted_sources = {
            "version": "wandb_accepted_sources_v2",
            "sources": [{
                "artifact_ref": "entity/project/accepted:v1",
                "artifact_digest": "accepted-digest",
                "file_manifest_sha256": "c" * 64,
            }],
        }
        accepted_sources["sha256"] = common.canonical_json_sha256(accepted_sources)
    model_file_manifest = model_file_manifest or _model_file_manifest(step)
    aggregate = {
        "metric_contract_version": sft_gate_gpu.METRIC_CONTRACT_VERSION,
        "model_source": model_file_manifest["source"],
        "model_file_manifest": model_file_manifest,
        "model_checkpoint": {
            "metadata_file": "sft_checkpoint.json", "metadata": metadata,
        },
        "accepted_library_digest": accepted_digest,
        "accepted_source_manifest": accepted_sources,
        "selection": {
            "uses_model_outputs": False,
            "uses_correctness": False,
            "selection_seed": 20260713,
            "heldout_selection_sha256": common.canonical_json_sha256(heldout_payload),
        },
        "sampling": {
            **select_sft_checkpoint.FROZEN_SAMPLING,
            "max_rendered_prompt_tokens": 100,
        },
        "prompt_modes": {mode: mode for mode in sft_gate_gpu.PROMPT_MODES},
        "metrics_by_prompt_mode": metrics,
        "health_gate": gate,
    }
    for name, rows in (("heldout_selection.jsonl", heldout),
                       ("rollouts.jsonl", rollouts)):
        (root / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    return root


def make_bundles(tmp_path: Path, passing=(16, 32)) -> dict[int, str]:
    return {
        step: str(make_gate_bundle(
            tmp_path / f"gate-{step}", step, passed=step in passing))
        for step in select_sft_checkpoint.PREREGISTERED_STEPS
    }


def sources() -> dict[int, str]:
    return {
        step: f"entity/project/gate-step-{step}:v0#digest=gate-digest-{step}"
        for step in select_sft_checkpoint.PREREGISTERED_STEPS
    }


def make_fresh_bundles(
    tmp_path: Path, passing=(125, 250), *, evidence: dict | None = None,
    accepted_sources: dict | None = None,
) -> dict[int, str]:
    evidence = evidence or fresh_training_evidence()
    return {
        step: str(make_gate_bundle(
            tmp_path / f"gate-{step}", step, passed=step in passing,
            selection_profile=fresh_c8_sft.PROFILE,
            training_evidence=evidence,
            accepted_sources=accepted_sources,
        ))
        for step in fresh_c8_sft.PREREGISTERED_STEPS
    }


def fresh_sources() -> dict[int, str]:
    return {
        step: f"entity/project/fresh-gate-{step}:v0#digest=fresh-digest-{step}"
        for step in fresh_c8_sft.PREREGISTERED_STEPS
    }


def make_fresh_accepted_artifact(
    tmp_path: Path,
) -> tuple[Path, dict, dict, dict]:
    root = tmp_path / "accepted-artifact"
    root.mkdir(parents=True)
    path = root / "accepted.jsonl"
    path.write_text("".join(
        json.dumps({
            "problem_id": f"accepted-{index}",
            "prompt_text": f"accepted training prompt {index}",
            "completion_text": "reason \\boxed{1}",
        }, sort_keys=True) + "\n"
        for index in range(500)
    ))
    source = {
        "artifact_ref": "entity/project/accepted:v1",
        "artifact_digest": "accepted-digest",
    }
    files = [{
        "relative_path": "accepted.jsonl",
        "sha256": common.file_sha256(str(path)),
        "size_bytes": path.stat().st_size,
    }]
    normalized = {
        "version": common.ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": source,
        "files": files,
    }
    identity = {
        **normalized,
        "file_manifest_sha256": common.canonical_json_sha256(normalized),
    }
    accepted_sources = {
        "version": "wandb_accepted_sources_v2",
        "sources": [{
            **source,
            "file_manifest_sha256": identity["file_manifest_sha256"],
        }],
    }
    accepted_sources["sha256"] = common.canonical_json_sha256(accepted_sources)
    evidence = fresh_training_evidence(path)
    return path, identity, accepted_sources, evidence


def make_fresh_selection_fixture(
    tmp_path: Path, passing=(125, 250),
) -> tuple[dict[int, str], Path, dict, dict]:
    accepted_path, identity, accepted_sources, evidence = \
        make_fresh_accepted_artifact(tmp_path)
    bundles = make_fresh_bundles(
        tmp_path / "gates", passing=passing, evidence=evidence,
        accepted_sources=accepted_sources)
    return bundles, accepted_path, identity, evidence


def test_latest_passing_is_rebuilt_from_all_raw_bundles(tmp_path):
    manifest = select_sft_checkpoint.build_selection_manifest_from_bundles(
        make_bundles(tmp_path, passing=(16, 32)), sources())
    assert manifest["selected_optimizer_step"] == 32
    assert manifest["selection_uses_correctness"] is False
    # The frozen c7-reuse profile remains byte-schema compatible with v25;
    # fresh-c8 fields are never injected into its signed manifest.
    assert "selection_profile" not in manifest
    assert all("fresh_c8_training_evidence" not in candidate
               for candidate in manifest["candidates"])
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    assert manifest["manifest_sha256"] == common.canonical_json_sha256(unsigned)


def test_correctness_diagnostic_cannot_change_selected_step(tmp_path):
    low = make_bundles(tmp_path / "low", passing=(16, 32, 64))
    high = make_bundles(tmp_path / "high", passing=(16, 32, 64))
    for bundle in high.values():
        path = Path(bundle) / "rollouts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["correctness_diagnostic_only"] = not row["correctness_diagnostic_only"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        aggregate_path = Path(bundle) / "aggregate_metrics.json"
        aggregate = json.loads(aggregate_path.read_text())
        for mode in sft_gate_gpu.PROMPT_MODES:
            aggregate["metrics_by_prompt_mode"][mode][
                "correctness_rate_diagnostic_only"] = 0.0
        aggregate_path.write_text(json.dumps(aggregate))
    assert select_sft_checkpoint.build_selection_manifest_from_bundles(
        low, sources())["selected_optimizer_step"] == 64
    assert select_sft_checkpoint.build_selection_manifest_from_bundles(
        high, sources())["selected_optimizer_step"] == 64


@pytest.mark.parametrize("forgery", ["summary_rate", "passed_flag", "raw_text"])
def test_raw_recomputation_rejects_forged_summary_or_rollout(tmp_path, forgery):
    bundle = make_gate_bundle(tmp_path / "gate", 16)
    aggregate_path = bundle / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text())
    if forgery == "summary_rate":
        aggregate["metrics_by_prompt_mode"]["training"]["termination_rate"] = 0.0
        aggregate_path.write_text(json.dumps(aggregate))
    elif forgery == "passed_flag":
        aggregate["health_gate"]["passed"] = False
        aggregate_path.write_text(json.dumps(aggregate))
    else:
        raw_path = bundle / "rollouts.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        rows[0]["completion_text"] = "no boxed answer"
        raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="differ.*raw-rollout recomputation"):
        select_sft_checkpoint.validate_gate_bundle(
            str(bundle), 16, sources()[16])


@pytest.mark.parametrize(
    "field,bad", [
        ("model_init", "other/model"),
        ("optimizer", "adafactor"),
        ("learning_rate", 1e-5),
        ("grad_accum", 1),
        ("accepted_examples", 499),
        ("optimizer_steps_executed", 125),
        ("saved_optimizer_steps", [16, 32, 64]),
    ],
)
def test_exact_ga8_diagnostic_metadata_is_required(tmp_path, field, bad):
    bundle = make_gate_bundle(tmp_path / "gate", 16)
    path = bundle / "aggregate_metrics.json"
    aggregate = json.loads(path.read_text())
    aggregate["model_checkpoint"]["metadata"][field] = bad
    path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="diagnostic contract"):
        select_sft_checkpoint.validate_gate_bundle(str(bundle), 16, sources()[16])


@pytest.mark.parametrize(
    "field,bad", [
        ("resolved_max_total_len", 32768),
        ("completion_budget_flag", 16384),
        ("seed", 1),
        ("parameter_dtype", "bfloat16"),
        ("adam_beta2", 0.999),
        ("adam_eps", 1e-6),
        ("weight_decay", 0.1),
        ("grad_clip", 0.5),
        ("config_tier", "gpu-max:completion_budget"),
    ],
)
def test_exact_ga8_sft_runtime_contract_drift_is_rejected(
        tmp_path, field, bad):
    bundle = make_gate_bundle(tmp_path / "gate", 16)
    path = bundle / "aggregate_metrics.json"
    aggregate = json.loads(path.read_text())
    runtime = aggregate["model_checkpoint"]["metadata"]["sft_runtime_contract"]
    runtime[field] = bad
    path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="diagnostic contract"):
        select_sft_checkpoint.validate_gate_bundle(
            str(bundle), 16, sources()[16])


def test_all_four_exact_sources_are_required(tmp_path):
    bundles = make_bundles(tmp_path)
    with pytest.raises(ValueError, match="exactly"):
        select_sft_checkpoint.build_selection_manifest_from_bundles(
            {step: path for step, path in bundles.items() if step != 126}, sources())
    bad = sources()
    bad[126] = "floating-latest"
    with pytest.raises(ValueError, match="exact artifact digest"):
        select_sft_checkpoint.build_selection_manifest_from_bundles(bundles, bad)


def test_missing_raw_rollouts_fails_closed(tmp_path):
    bundle = make_gate_bundle(tmp_path / "gate", 16)
    (bundle / "rollouts.jsonl").unlink()
    with pytest.raises(ValueError, match="raw rollouts missing"):
        select_sft_checkpoint.validate_gate_bundle(str(bundle), 16, sources()[16])


def test_fresh_c8_latest_passing_and_full_endpoint_labels(tmp_path):
    early_bundles, early_accepted, early_identity, _ = \
        make_fresh_selection_fixture(
            tmp_path / "early", passing=(125, 250, 500))
    early = select_sft_checkpoint.build_selection_manifest_from_bundles(
        early_bundles, fresh_sources(), fresh_c8_sft.PROFILE,
        accepted_jsonl=str(early_accepted),
        accepted_artifact_identity=early_identity)
    assert early["selection_profile"] == fresh_c8_sft.PROFILE
    assert early["preregistered_optimizer_steps"] == [125, 250, 500, 1000]
    assert early["selected_optimizer_step"] == 500
    assert early["selection_uses_correctness"] is False
    fresh_c8_sft.validate_training_evidence(early["fresh_c8_training_evidence"])

    endpoint_bundles, endpoint_accepted, endpoint_identity, _ = \
        make_fresh_selection_fixture(
            tmp_path / "endpoint", passing=(125, 250, 500, 1000))
    endpoint = select_sft_checkpoint.build_selection_manifest_from_bundles(
        endpoint_bundles, fresh_sources(), fresh_c8_sft.PROFILE,
        accepted_jsonl=str(endpoint_accepted),
        accepted_artifact_identity=endpoint_identity)
    assert endpoint["selected_optimizer_step"] == 1000


def test_fresh_c8_no_passing_snapshot_fails_closed(tmp_path):
    bundles, accepted, identity, _ = make_fresh_selection_fixture(
        tmp_path, passing=())
    with pytest.raises(ValueError, match="none.*passed"):
        select_sft_checkpoint.build_selection_manifest_from_bundles(
            bundles, fresh_sources(), fresh_c8_sft.PROFILE,
            accepted_jsonl=str(accepted),
            accepted_artifact_identity=identity)


def _rewrite_heldout_and_digest(bundle: str, rows: list[dict]) -> None:
    heldout_path = Path(bundle) / "heldout_selection.jsonl"
    heldout_path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))
    payload = [{
        "problem_id": row["problem_id"],
        "selection_digest": row["selection_digest"],
        "training_prompt_sha256": row["training_prompt_sha256"],
    } for row in rows]
    aggregate_path = Path(bundle) / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["selection"]["heldout_selection_sha256"] = \
        common.canonical_json_sha256(payload)
    aggregate_path.write_text(json.dumps(aggregate))


@pytest.mark.parametrize("overlap", ["problem_id", "training_prompt"])
def test_fresh_c8_rejects_either_heldout_training_overlap(tmp_path, overlap):
    bundles, accepted, identity, _ = make_fresh_selection_fixture(tmp_path)
    bundle = bundles[125]
    rows = [json.loads(line) for line in
            (Path(bundle) / "heldout_selection.jsonl").read_text().splitlines()]
    if overlap == "problem_id":
        rows[0]["problem_id"] = "accepted-0"
        rows[0]["selection_digest"] = sft_gate_gpu._sha256(
            f"20260713\0accepted-0\0{rows[0]['problem']}")
        expected = "problem-ID overlaps"
    else:
        rows[0]["training_prompt"] = "accepted training prompt 0"
        rows[0]["training_prompt_sha256"] = sft_gate_gpu._sha256(
            rows[0]["training_prompt"])
        expected = "training-prompt hash overlaps"
    _rewrite_heldout_and_digest(bundle, rows)
    with pytest.raises(ValueError, match=expected):
        select_sft_checkpoint.validate_gate_bundle(
            bundle, 125, fresh_sources()[125], fresh_c8_sft.PROFILE,
            accepted_jsonl=str(accepted), accepted_artifact_identity=identity)


@pytest.mark.parametrize(
    "field,match", [
        ("training_prompt_sha256", "persisted training-prompt hash mismatch"),
        ("selection_digest", "persisted seed-selection digest mismatch"),
    ],
)
def test_fresh_c8_recomputes_persisted_heldout_identity(
        tmp_path, field, match):
    bundles, accepted, identity, _ = make_fresh_selection_fixture(tmp_path)
    bundle = bundles[125]
    rows = [json.loads(line) for line in
            (Path(bundle) / "heldout_selection.jsonl").read_text().splitlines()]
    rows[0][field] = "0" * 64
    _rewrite_heldout_and_digest(bundle, rows)
    with pytest.raises(ValueError, match=match):
        select_sft_checkpoint.validate_gate_bundle(
            bundle, 125, fresh_sources()[125], fresh_c8_sft.PROFILE,
            accepted_jsonl=str(accepted), accepted_artifact_identity=identity)


@pytest.mark.parametrize(
    "field,bad", [
        ("grad_accum", 8),
        ("planned_optimizer_steps", 999),
        ("optimizer_steps_executed", 999),
        ("micro_steps_executed", 999),
        ("requested_save_optimizer_steps", [125, 250, 500]),
        ("training_completed", False),
    ],
)
def test_fresh_c8_checkpoint_contract_drift_is_rejected(tmp_path, field, bad):
    accepted, identity, accepted_sources, evidence = \
        make_fresh_accepted_artifact(tmp_path)
    bundle = make_gate_bundle(
        tmp_path / "gate", 125, selection_profile=fresh_c8_sft.PROFILE,
        training_evidence=evidence, accepted_sources=accepted_sources)
    path = bundle / "aggregate_metrics.json"
    aggregate = json.loads(path.read_text())
    aggregate["model_checkpoint"]["metadata"][field] = bad
    path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="fresh_c8_single_scout contract"):
        select_sft_checkpoint.validate_gate_bundle(
            str(bundle), 125, fresh_sources()[125], fresh_c8_sft.PROFILE,
            accepted_jsonl=str(accepted), accepted_artifact_identity=identity)


@pytest.mark.parametrize(
    "section,field,bad", [
        ("trajectories", "row_count", 12799),
        ("trajectories", "rows_per_step", 63),
        ("accepted_library", "unique_problem_count", 499),
        ("accepted_library", "policy", "naive_pool"),
        ("scout", "step", 199),
        ("scout", "config_tier", "gpu-max:completion_budget"),
        ("code_source", "artifact_digest", ""),
    ],
)
def test_fresh_c8_training_lineage_drift_is_rejected(
        tmp_path, section, field, bad):
    accepted, identity, accepted_sources, evidence = \
        make_fresh_accepted_artifact(tmp_path)
    evidence[section][field] = bad
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256")
    evidence["evidence_sha256"] = common.canonical_json_sha256(unsigned)
    bundle = make_gate_bundle(
        tmp_path / "gate", 125, selection_profile=fresh_c8_sft.PROFILE,
        training_evidence=evidence, accepted_sources=accepted_sources)
    with pytest.raises(ValueError, match="fresh-c8"):
        select_sft_checkpoint.validate_gate_bundle(
            str(bundle), 125, fresh_sources()[125], fresh_c8_sft.PROFILE,
            accepted_jsonl=str(accepted), accepted_artifact_identity=identity)


def test_fresh_c8_completed_run_evidence_is_mandatory_and_signed(tmp_path):
    accepted, identity, accepted_sources, evidence = \
        make_fresh_accepted_artifact(tmp_path)
    bundle = make_gate_bundle(
        tmp_path / "missing", 125, selection_profile=fresh_c8_sft.PROFILE,
        training_evidence=evidence, accepted_sources=accepted_sources)
    path = bundle / "aggregate_metrics.json"
    aggregate = json.loads(path.read_text())
    aggregate["model_checkpoint"]["metadata"].pop("completed_run_evidence")
    path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="completed-run evidence"):
        select_sft_checkpoint.validate_gate_bundle(
            str(bundle), 125, fresh_sources()[125], fresh_c8_sft.PROFILE,
            accepted_jsonl=str(accepted), accepted_artifact_identity=identity)

    forged = make_gate_bundle(
        tmp_path / "forged", 125, selection_profile=fresh_c8_sft.PROFILE,
        training_evidence=evidence, accepted_sources=accepted_sources)
    forged_path = forged / "aggregate_metrics.json"
    aggregate = json.loads(forged_path.read_text())
    aggregate["model_checkpoint"]["metadata"]["completed_run_evidence"][
        "optimizer_steps_executed"] = 999
    forged_path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="SHA256"):
        select_sft_checkpoint.validate_gate_bundle(
            str(forged), 125, fresh_sources()[125], fresh_c8_sft.PROFILE,
            accepted_jsonl=str(accepted), accepted_artifact_identity=identity)


def test_fresh_c8_candidates_must_share_identical_training_evidence(tmp_path):
    bundles, accepted, identity, _ = make_fresh_selection_fixture(tmp_path)
    path = Path(bundles[250]) / "aggregate_metrics.json"
    aggregate = json.loads(path.read_text())
    evidence = aggregate["model_checkpoint"]["metadata"][
        "fresh_c8_training_evidence"]
    evidence["filter_funnel"]["correct"] = 123
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256")
    evidence["evidence_sha256"] = common.canonical_json_sha256(unsigned)
    aggregate["model_checkpoint"]["metadata"]["completed_run_evidence"] = \
        fresh_c8_sft.build_completed_run_evidence(
            optimizer_steps=1000, micro_steps=1000,
            saved_steps=list(fresh_c8_sft.PREREGISTERED_STEPS),
            training_evidence_sha256=evidence["evidence_sha256"])
    aggregate["model_checkpoint"]["metadata"]["accepted_library_digest"] = \
        evidence["accepted_library"]["semantic_digest"]
    aggregate["accepted_library_digest"] = \
        evidence["accepted_library"]["semantic_digest"]
    path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="disagree.*fresh_c8_training_evidence"):
        select_sft_checkpoint.build_selection_manifest_from_bundles(
            bundles, fresh_sources(), fresh_c8_sft.PROFILE,
            accepted_jsonl=str(accepted), accepted_artifact_identity=identity)
