"""Focused invariants for the opt-in C9 ExpDis stabilization path."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tmx_gpu import common, grpo_gpu, orchestrator_receipt, pipeline_gpu
from tmx_gpu.tests import orchestrator_receipt_fixtures as receipt_fixtures
from tmx_gpu.rnd_torch import MultilayerRND
from tmx_gpu.vllm_pool import Completion


def grpo_args(*extra: str):
    return grpo_gpu.build_arg_parser().parse_args(
        ["--output-dir", "/tmp/grpo-c9-test", *extra])


def c9_args(*extra: str):
    return grpo_args(
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "--phase", "scout",
        "--rl-validity-mode", "c8",
        "--lambda-novelty", "0.5",
        "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
        "--novelty-normalization", "selected_batch_zscore",
        "--scientific-variant", grpo_gpu.C9_SCIENTIFIC_VARIANT,
        "--code-source-manifest", "/tmp/immutable-code-manifest.json",
        *extra,
    )


def full_c9_pipeline_args(tmp_path, *extra: str):
    manifest = tmp_path / "code.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return pipeline_gpu.build_arg_parser().parse_args([
        "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
        "--container-image-identity",
        grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
        "--output-dir", str(tmp_path / "out"),
        "--run-name", "c9-full-c5",
        "--c9-execution-profile", "full",
        "--stabilization-cell", "C5",
        "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
        "--novelty-normalization", "selected_batch_zscore",
        "--scientific-variant", grpo_gpu.C9_SCIENTIFIC_VARIANT,
        "--scout-lr", "1e-6",
        "--reload-every", "1",
        "--vllm-max-model-len", "36864",
        "--code-source-manifest", str(manifest),
        "--vllm-gpus", "0",
        "--wandb",
        "--ckpt-artifact-every", "25",
        *extra,
    ])


class TestC9ConfigurationBoundary:
    @pytest.fixture(autouse=True)
    def _isolate_profile_tests_from_remote_receipt_io(self, monkeypatch):
        """Supply a real validated receipt without reaching W&B in unit tests.

        Receipt parsing/spoof resistance has its own adversarial suite.  The
        tests in this class exercise C9 mechanism/profile invariants, so bind
        the same immutable code identity and synthesize the corresponding
        already-resolved receipt for whichever run name the fixture declares.
        """
        code = {
            "source": {
                "artifact_ref": "entity/project/code:v1",
                "artifact_digest": "code-digest",
            },
        }
        monkeypatch.setattr(
            grpo_gpu, "load_code_source_manifest",
            lambda _path: copy.deepcopy(code))
        real_validate_binding = orchestrator_receipt.validate_binding

        def validate_binding(binding, **expected):
            job = binding["receipt"]["job"]
            defaults = {
                "expected_run_name": job["run_name"],
                "expected_container_image":
                    job["container_image_identity"],
                "expected_code_artifact_ref":
                    job["code_artifact"]["artifact_ref"],
                "expected_code_artifact_digest":
                    job["code_artifact"]["artifact_digest"],
                "expected_command_sha256": job["command_sha256"],
                "expected_polling_artifact_ref": job["environment"][
                    "ORCHESTRATOR_RECEIPT_ARTIFACT"],
            }
            normalized = {
                key: defaults[key] if value in (None, "") else value
                for key, value in expected.items()
            }
            return real_validate_binding(binding, **normalized)

        monkeypatch.setattr(
            orchestrator_receipt, "validate_binding", validate_binding)

        def load_binding(
            _path, *, expected_run_name=None, expected_container_image=None,
            expected_code_artifact_ref=None,
            expected_code_artifact_digest=None,
            expected_command_sha256=None,
            expected_polling_artifact_ref=None,
        ):
            binding = receipt_fixtures.binding(
                run_name=str(expected_run_name),
                code_ref=code["source"]["artifact_ref"],
                code_digest=code["source"]["artifact_digest"],
            )
            return validate_binding(
                binding,
                expected_run_name=expected_run_name,
                expected_container_image=expected_container_image,
                expected_code_artifact_ref=expected_code_artifact_ref,
                expected_code_artifact_digest=expected_code_artifact_digest,
                expected_command_sha256=(
                    binding["receipt"]["job"]["command_sha256"]
                    if not expected_command_sha256 else
                    expected_command_sha256),
                expected_polling_artifact_ref=(
                    binding["receipt"]["job"]["environment"][
                        "ORCHESTRATOR_RECEIPT_ARTIFACT"]
                    if not expected_polling_artifact_ref else
                    expected_polling_artifact_ref),
            )

        monkeypatch.setattr(orchestrator_receipt, "load_binding", load_binding)

    def test_c8_defaults_remain_raw_and_implicit_scout_lr(self):
        args = grpo_args()
        assert args.expdis_stabilization_mode == grpo_gpu.C8_RND_LIFECYCLE
        assert args.novelty_normalization == "raw"
        assert args.lr is None
        resolved = float(args.lr) if args.lr is not None else 5e-6
        assert resolved == 5e-6
        assert grpo_gpu.resolved_config_tier(args) == "contract"

        runtime = grpo_gpu.build_runtime_contract(
            args, resolved_lr=resolved, resolved_steps=200)
        assert runtime["scout_novelty_contract"] == {
            "novelty_layers": "7,14,21",
            "rnd_learning_rate": 1e-4,
            "rnd_hidden_dim": 512,
            "novelty_feature_max_length": 16384,
            "incorrect_novelty_scale": 0.0,
        }

    def test_c9_requires_exact_label_validity_gate_and_code_binding(self):
        grpo_gpu.validate_expdis_stabilization(
            c9_args(), phase="scout", lambda_novelty=0.5)

        with pytest.raises(ValueError, match="mislabeled stabilization"):
            grpo_gpu.validate_expdis_stabilization(
                c9_args("--scientific-variant", "wrong"),
                phase="scout", lambda_novelty=0.5)
        with pytest.raises(ValueError, match="rl-validity-mode c8"):
            grpo_gpu.validate_expdis_stabilization(
                c9_args("--rl-validity-mode", "legacy"),
                phase="scout", lambda_novelty=0.5)
        with pytest.raises(ValueError, match="immutable --code-source-manifest"):
            grpo_gpu.validate_expdis_stabilization(
                c9_args("--code-source-manifest", ""),
                phase="scout", lambda_novelty=0.5)

    def test_c8_cannot_silently_take_c9_normalization_or_label(self):
        with pytest.raises(ValueError, match="C9-only"):
            grpo_gpu.validate_expdis_stabilization(
                grpo_args("--novelty-normalization", "selected_batch_zscore"),
                phase="scout", lambda_novelty=0.5)
        with pytest.raises(ValueError, match="cannot be attached"):
            grpo_gpu.validate_expdis_stabilization(
                grpo_args("--scientific-variant", grpo_gpu.C9_SCIENTIFIC_VARIANT),
                phase="scout", lambda_novelty=0.5)

    def test_c9_runtime_contract_is_explicit_and_resume_stable(self):
        args = c9_args("--lr", "1e-6")
        args._training_execution_environment = receipt_fixtures.environment(
            run_name=args.run_name)
        runtime = grpo_gpu.build_runtime_contract(
            args, resolved_lr=1e-6, resolved_steps=50)
        novelty = runtime["scout_novelty_contract"]
        assert runtime["scientific_variant"] == grpo_gpu.C9_SCIENTIFIC_VARIANT
        assert runtime["resolved_learning_rate"] == 1e-6
        assert novelty["rnd_lifecycle"] == grpo_gpu.C9_RND_LIFECYCLE
        assert novelty["rnd_updates_per_completed_step"] == 1
        assert novelty["novelty_normalization"] == "selected_batch_zscore"
        assert novelty["novelty_gate_order"] == "normalize_then_correct_only_gate"

    def test_pipeline_passes_c9_normalization_and_explicit_lr(self, tmp_path):
        args = pipeline_gpu.build_arg_parser().parse_args([
            "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
            "--container-image-identity",
            grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            "--output-dir", str(tmp_path / "out"),
            "--run-name", "c9-canary",
            "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
            "--novelty-normalization", "selected_batch_zscore",
            "--scientific-variant", grpo_gpu.C9_SCIENTIFIC_VARIANT,
            "--code-source-manifest", str(tmp_path / "code.json"),
            "--scout-lr", "1e-6",
            "--central-lr", "1e-6",
        ])
        scout = pipeline_gpu._grpo_args(
            args, phase="scout", model="m", output_dir=str(tmp_path / "scout"),
            lam=0.5, steps=50, round_idx=1, global_step_offset=0)
        central = pipeline_gpu._grpo_args(
            args, phase="central", model="m", output_dir=str(tmp_path / "central"),
            lam=0.0, steps=100, round_idx=1, global_step_offset=50)
        assert scout.lr == 1e-6 and central.lr == 1e-6
        assert scout.novelty_normalization == "selected_batch_zscore"
        assert scout.expdis_stabilization_mode == grpo_gpu.C9_RND_LIFECYCLE
        assert scout.scientific_variant == grpo_gpu.C9_SCIENTIFIC_VARIANT
        assert scout.code_source_manifest == str(tmp_path / "code.json")
        # Strict provenance follows the model through the deployed Central;
        # it must not stop at the Scout boundary.
        assert central.code_source_manifest == str(tmp_path / "code.json")

    def test_c9_central_rejects_base_init_and_has_no_rnd_lifecycle(self):
        args = c9_args("--phase", "central", "--lambda-novelty", "0")
        with pytest.raises(ValueError, match="local SFT checkpoint"):
            grpo_gpu.validate_expdis_stabilization(
                args, phase="central", lambda_novelty=0.0)
        metadata = grpo_gpu._c9_artifact_metadata(args)
        assert metadata["expdis_stabilization_mode"] == grpo_gpu.C9_RND_LIFECYCLE
        assert metadata["novelty_active"] is False
        assert "rnd_updates_per_completed_step" not in metadata

    @pytest.mark.parametrize(
        "cell,lifecycle,normalization,lr,variant",
        [
            ("C1", grpo_gpu.C8_RND_LIFECYCLE, "raw", "1e-6",
             grpo_gpu.C9_SCREEN_VARIANT),
            ("C2", grpo_gpu.C9_RND_LIFECYCLE, "raw", "5e-6",
             grpo_gpu.C9_SCREEN_VARIANT),
            ("C3", grpo_gpu.C9_RND_LIFECYCLE, "raw", "1e-6",
             grpo_gpu.C9_SCREEN_VARIANT),
            ("C4", grpo_gpu.C9_RND_LIFECYCLE, "selected_batch_zscore", "5e-6",
             grpo_gpu.C9_SCREEN_VARIANT),
            ("C5", grpo_gpu.C9_RND_LIFECYCLE, "selected_batch_zscore", "1e-6",
             grpo_gpu.C9_SCIENTIFIC_VARIANT),
        ],
    )
    def test_preregistered_canary_cells_bind_exact_mechanism(
            self, tmp_path, cell, lifecycle, normalization, lr, variant):
        manifest = tmp_path / "code.json"
        manifest.write_text("{}")
        args = pipeline_gpu.build_arg_parser().parse_args([
            "--model-revision", common.CANONICAL_TRAINING_MODEL_REVISION,
            "--container-image-identity",
            grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            "--output-dir", str(tmp_path / "out"),
            "--run-name", f"c9-{cell.lower()}",
            "--c9-execution-profile", "canary",
            "--stabilization-cell", cell,
            "--expdis-stabilization-mode", lifecycle,
            "--novelty-normalization", normalization,
            "--scientific-variant", variant,
            "--scout-lr", lr,
            "--scout-steps", "50",
            "--scout-only",
            "--probe-every", "0",
            "--reload-every", "1",
            "--vllm-max-model-len", "36864",
            "--code-source-manifest", str(manifest),
        ])
        # Isolate the fixed-profile contract from artifact-manifest parsing;
        # the latter has independent hardening tests with real manifests.
        original = pipeline_gpu._validated_executing_code_source_manifest
        try:
            pipeline_gpu._validated_executing_code_source_manifest = (
                lambda *_args, **_kwargs: {})
            pipeline_gpu.validate_args(args)
        finally:
            pipeline_gpu._validated_executing_code_source_manifest = original

    def test_canary_contract_rejects_probe_or_lr_drift(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline_gpu, "_validated_executing_code_source_manifest",
            lambda *_args, **_kwargs: {})
        base = [
            "--output-dir", str(tmp_path / "out"), "--run-name", "c9-c5",
            "--container-image-identity",
            grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE,
            "--c9-execution-profile", "canary", "--stabilization-cell", "C5",
            "--expdis-stabilization-mode", grpo_gpu.C9_RND_LIFECYCLE,
            "--novelty-normalization", "selected_batch_zscore",
            "--scientific-variant", grpo_gpu.C9_SCIENTIFIC_VARIANT,
            "--scout-steps", "50", "--scout-only", "--reload-every", "1",
            "--vllm-max-model-len", "36864", "--code-source-manifest",
            str(tmp_path / "code.json"),
        ]
        with pytest.raises(ValueError, match="probe_every=25"):
            pipeline_gpu.validate_args(
                pipeline_gpu.build_arg_parser().parse_args([
                    *base, "--scout-lr", "1e-6"]))
        with pytest.raises(ValueError, match="scout_lr=5e-06"):
            pipeline_gpu.validate_args(
                pipeline_gpu.build_arg_parser().parse_args([
                    *base, "--probe-every", "0", "--scout-lr", "5e-6"]))

    def test_full_c9_requires_online_wandb_and_positive_artifact_cadence(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            pipeline_gpu, "_validated_executing_code_source_manifest",
            lambda *_args, **_kwargs: {
                "source": {
                    "artifact_ref": "entity/project/code:v1",
                    "artifact_digest": "digest",
                },
            })
        monkeypatch.delenv("WANDB_MODE", raising=False)
        canonical = full_c9_pipeline_args(tmp_path)
        pipeline_gpu.validate_args(canonical)

        no_wandb = copy.deepcopy(canonical)
        no_wandb.wandb = False
        with pytest.raises(ValueError, match="online wandb"):
            pipeline_gpu.validate_args(no_wandb)

        monkeypatch.setenv("WANDB_MODE", "offline")
        with pytest.raises(ValueError, match="online wandb"):
            pipeline_gpu.validate_args(copy.deepcopy(canonical))
        monkeypatch.delenv("WANDB_MODE")

        no_cadence = copy.deepcopy(canonical)
        no_cadence.ckpt_artifact_every = 0
        with pytest.raises(ValueError, match="ckpt_artifact_every"):
            pipeline_gpu.validate_args(no_cadence)

    def test_full_c9_execution_contract_binds_durable_stage_artifacts(
        self, tmp_path, monkeypatch,
    ):
        code = {
            "source": {
                "artifact_ref": "entity/project/code:v1",
                "artifact_digest": "digest",
            },
        }
        monkeypatch.setattr(
            pipeline_gpu, "_validated_executing_code_source_manifest",
            lambda *_args, **_kwargs: code)
        monkeypatch.delenv("WANDB_MODE", raising=False)
        args = full_c9_pipeline_args(tmp_path)
        pipeline_gpu.validate_args(args)
        args._training_execution_environment = receipt_fixtures.environment(
            run_name=args.run_name, code_ref=code["source"]["artifact_ref"],
            code_digest=code["source"]["artifact_digest"])
        contract = pipeline_gpu._c9_pipeline_execution_contract(args)
        assert contract["artifact_durability_contract"] == {
            "online_wandb_required": True,
            "grpo_final_artifact_every_steps": 25,
            "durable_scout_sft_central_required": True,
        }
        assert contract["contract_sha256"] == common.canonical_json_sha256({
            key: value for key, value in contract.items()
            if key != "contract_sha256"
        })


class TestSeparatedRNDLifecycle:
    def test_score_is_read_only_and_selected_update_is_counted_and_persisted(self):
        rnd = MultilayerRND(
            input_dim=4, hidden_dim=8, layers=(0,), lr=1e-3,
            seed=7, device="cpu")
        features = {0: torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10}
        before = {
            name: value.detach().clone()
            for name, value in rnd.predictors[0].state_dict().items()
        }

        score1, layers1 = rnd.score(features)
        score2, layers2 = rnd.score(features)
        assert np.array_equal(score1, score2)
        assert np.array_equal(layers1[0], layers2[0])
        assert rnd.selected_batch_update_count == 0
        assert all(torch.equal(before[name], value)
                   for name, value in rnd.predictors[0].state_dict().items())

        loss = rnd.update_selected_batch(features)
        assert np.isfinite(loss)
        assert rnd.selected_batch_update_count == 1
        assert any(not torch.equal(before[name], value)
                   for name, value in rnd.predictors[0].state_dict().items())
        state = rnd.state_dict()
        assert state["meta"]["selected_batch_update_count"] == 1

        restored = MultilayerRND(
            input_dim=4, hidden_dim=8, layers=(0,), lr=1e-3,
            seed=7, device="cpu")
        restored.load_state_dict(copy.deepcopy(state))
        assert restored.selected_batch_update_count == 1

    def test_frozen_c8_score_and_update_schema_is_unchanged(self):
        rnd = MultilayerRND(
            input_dim=4, hidden_dim=8, layers=(0,), lr=1e-3,
            seed=3, device="cpu")
        rnd.score_and_update({0: torch.ones(2, 4)}, train=True)
        assert rnd.selected_batch_update_count == 0
        assert "selected_batch_update_count" not in rnd.state_dict()["meta"]


def _reward_row(correct: bool) -> dict:
    correctness = 1.0 if correct else -1.0
    return {
        "problem_id": "p",
        "prompt_text": "prompt",
        "completion_text": "answer",
        "ground_truth": "1",
        "finish_reason": "stop",
        "completion_token_length": 1,
        "clipped": False,
        "terminated": True,
        "valid_answer": True,
        "is_correct": correct,
        "correctness_raw": correctness,
        "correctness_reward": correctness,
        "overlong_penalty": 0.0,
        "blended_reward": correctness,
        "novelty_raw": 0.0,
        "novelty_used": 0.0,
        "novelty_reward": 0.0,
        "r_novel_layers": {},
        "_novelty_text": "answer",
        "_comp_ids": [1],
    }


class _Pool:
    def generate(self, prompts, n, **_kwargs):
        return [[Completion(text="answer", finish_reason="stop") for _ in range(n)]
                for _ in prompts]


class _FakeRND:
    layers = (0,)

    def __init__(self):
        self.score_calls = 0
        self.score_and_update_calls = 0
        self.update_calls = 0
        self.selected_batch_update_count = 0
        self.updated_features = None

    def score(self, features):
        self.score_calls += 1
        values = features[0][:, 0].detach().cpu().numpy().astype(np.float32)
        return values, {0: values.copy()}

    def score_and_update(self, features, *, train=True):
        assert train is True
        self.score_and_update_calls += 1
        values = features[0][:, 0].detach().cpu().numpy().astype(np.float32)
        return values, {0: values.copy()}

    def update_selected_batch(self, features):
        self.update_calls += 1
        self.selected_batch_update_count += 1
        self.updated_features = features[0].detach().clone()
        return 0.25


def _dynamic_args(*extra: str):
    return c9_args(
        "--prompts-per-step", "1",
        "--num-generations", "2",
        "--dynamic-max-attempts", "2",
        "--probe-every", "0",
        *extra,
    )


class TestC9DynamicSamplingLifecycle:
    def test_only_final_selected_rows_update_once_and_rewards_are_recomputed(
            self, monkeypatch):
        calls = {"score": 0, "features": 0}

        def fake_score_completions(*_args, **_kwargs):
            calls["score"] += 1
            correctness = ([False, False] if calls["score"] == 1
                           else [True, False])
            return [_reward_row(value) for value in correctness]

        def fake_features(*_args, **_kwargs):
            calls["features"] += 1
            base = 1 if calls["features"] == 1 else 3
            return {0: torch.tensor([[base], [base + 1]], dtype=torch.float32)}, {
                "count": 2.0, "truncated_count": 0.0}

        monkeypatch.setattr(grpo_gpu, "score_completions", fake_score_completions)
        monkeypatch.setattr(grpo_gpu, "extract_novelty_features", fake_features)
        rnd = _FakeRND()
        example = SimpleNamespace(problem_id="p", prompt_text="prompt", ground_truth="1")
        selected, all_rows, stats, _ = grpo_gpu._dynamic_sample_step(
            rng=np.random.default_rng(0), pool=_Pool(), tokenizer=None,
            model=None, rnd=rnd, examples=[example], args=_dynamic_args(),
            lam=0.5, step=1)

        assert rnd.score_calls == 2
        assert rnd.score_and_update_calls == 0
        assert rnd.update_calls == 1
        assert torch.equal(rnd.updated_features, torch.tensor([[3.0], [4.0]]))
        assert stats["rnd_candidate_rows_scored"] == 4.0
        assert stats["rnd_update_rows"] == 2.0
        assert stats["rnd_updates_this_step"] == 1.0
        assert stats["novelty_normalization_center"] == 3.5
        assert stats["novelty_normalization_scale"] == 0.5

        assert len(selected) == 2 and len(all_rows) == 4
        assert selected[0]["novelty_used"] == -1.0
        assert selected[1]["novelty_used"] == 0.0  # correct-only gate
        assert selected[0]["blended_reward"] == 0.5
        assert selected[1]["blended_reward"] == -1.0
        assert selected[0]["novelty_normalization"] == "selected_batch_zscore"
        assert all_rows[0]["novelty_normalization"] == \
            "raw_candidate_eligibility_only"

    def test_failed_dynamic_assembly_never_updates_predictor(self, monkeypatch):
        monkeypatch.setattr(
            grpo_gpu, "score_completions",
            lambda *_args, **_kwargs: [_reward_row(False), _reward_row(False)])
        feature_calls = {"n": 0}

        def fake_features(*_args, **_kwargs):
            feature_calls["n"] += 1
            return {0: torch.tensor([[1.0], [2.0]])}, {
                "count": 2.0, "truncated_count": 0.0}

        monkeypatch.setattr(grpo_gpu, "extract_novelty_features", fake_features)
        rnd = _FakeRND()
        example = SimpleNamespace(problem_id="p", prompt_text="prompt", ground_truth="1")
        with pytest.raises(RuntimeError, match="failed to assemble batch"):
            grpo_gpu._dynamic_sample_step(
                rng=np.random.default_rng(0), pool=_Pool(), tokenizer=None,
                model=None, rnd=rnd, examples=[example], args=_dynamic_args(),
                lam=0.5, step=1)
        assert rnd.score_calls == 2
        assert rnd.update_calls == 0
        assert rnd.selected_batch_update_count == 0

    def test_c8_still_updates_each_candidate_attempt(self, monkeypatch):
        calls = {"score": 0, "features": 0}

        def fake_score_completions(*_args, **_kwargs):
            calls["score"] += 1
            correctness = ([False, False] if calls["score"] == 1
                           else [True, False])
            return [_reward_row(value) for value in correctness]

        def fake_features(*_args, **_kwargs):
            calls["features"] += 1
            base = 1 if calls["features"] == 1 else 3
            return {0: torch.tensor([[base], [base + 1]], dtype=torch.float32)}, {
                "count": 2.0, "truncated_count": 0.0}

        monkeypatch.setattr(grpo_gpu, "score_completions", fake_score_completions)
        monkeypatch.setattr(grpo_gpu, "extract_novelty_features", fake_features)
        rnd = _FakeRND()
        args = grpo_args(
            "--phase", "scout", "--rl-validity-mode", "c8",
            "--prompts-per-step", "1", "--num-generations", "2",
            "--dynamic-max-attempts", "2", "--probe-every", "0")
        example = SimpleNamespace(problem_id="p", prompt_text="prompt", ground_truth="1")
        grpo_gpu._dynamic_sample_step(
            rng=np.random.default_rng(0), pool=_Pool(), tokenizer=None,
            model=None, rnd=rnd, examples=[example], args=args,
            lam=0.5, step=1)
        assert rnd.score_calls == 0
        assert rnd.score_and_update_calls == 2
        assert rnd.update_calls == 0

    def test_c9_labeled_central_uses_plain_correctness_sampling(self, monkeypatch):
        monkeypatch.setattr(
            grpo_gpu,
            "score_completions",
            lambda *_args, **_kwargs: [_reward_row(True), _reward_row(False)],
        )
        args = c9_args(
            "--phase", "central",
            "--lambda-novelty", "0",
            "--prompts-per-step", "1",
            "--num-generations", "2",
            "--dynamic-max-attempts", "1",
            "--probe-every", "0",
        )
        example = SimpleNamespace(
            problem_id="p", prompt_text="prompt", ground_truth="1")
        selected, _all_rows, stats, _ = grpo_gpu._dynamic_sample_step(
            rng=np.random.default_rng(0), pool=_Pool(), tokenizer=None,
            model=None, rnd=None, examples=[example], args=args,
            lam=0.0, step=1)
        assert len(selected) == 2
        assert "rnd_updates_this_step" not in stats
