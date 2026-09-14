"""Focused CPU tests for the explicit c8 GRPO validity contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tmx_gpu import common, grpo_gpu
from tmx_gpu.vllm_pool import Completion


def _args(*extra: str):
    return grpo_gpu.build_arg_parser().parse_args(
        ["--output-dir", "/tmp/grpo-c8-test", *extra])


class TestC8ModeValidation:
    def test_parser_preserves_legacy_default(self):
        assert _args().rl_validity_mode == "legacy"
        assert _args().mask_truncated is False

    def test_masking_is_explicit_ablation(self):
        assert _args("--mask-truncated").mask_truncated is True
        assert _args("--mask-truncated", "--no-mask-truncated").mask_truncated is False

    def test_c8_requires_managed_pool(self):
        args = _args("--rl-validity-mode", "c8")
        with pytest.raises(RuntimeError, match="managed vLLM pool"):
            grpo_gpu.validate_rl_validity(args, SimpleNamespace(managed=False))

    def test_c8_preserves_single_update_kl_zero_contract(self):
        pool = SimpleNamespace(managed=True)
        with pytest.raises(ValueError, match="updates-per-rollout 1"):
            grpo_gpu.validate_rl_validity(
                _args("--rl-validity-mode", "c8", "--updates-per-rollout", "2"),
                pool,
            )
        with pytest.raises(ValueError, match="KL=0 REINFORCE"):
            grpo_gpu.validate_rl_validity(
                _args("--rl-validity-mode", "c8", "--ppo-mode"), pool)

    def test_c8_syncs_after_every_update_including_final(self):
        args = _args(
            "--rl-validity-mode", "c8", "--reload-every", "1000")
        assert [grpo_gpu.sync_after_update_due(args, step, 3)
                for step in (1, 2, 3)] == [True, True, True]

    def test_legacy_sync_schedule_is_unchanged(self):
        args = _args("--reload-every", "2")
        assert [grpo_gpu.sync_after_update_due(args, step, 5)
                for step in (1, 2, 3, 4, 5)] == [False, True, False, True, False]


class TestC8DynamicEligibility:
    def test_overlong_only_variance_does_not_make_homogeneous_group_eligible(self):
        legacy = grpo_gpu.group_reason(16, 16, 0.25, 0.5)
        c8 = grpo_gpu.group_reason(
            16, 16, 0.25, 0.5,
            novelty_contribution_std=0.0, validity_mode="c8")
        assert legacy == ("reward_informative_all_correct", True)
        assert c8 == ("all_correct", False)

    def test_actual_novelty_variance_can_admit_homogeneous_correct_group(self):
        assert grpo_gpu.group_reason(
            16, 16, 0.25, 0.5,
            novelty_contribution_std=0.05, validity_mode="c8",
        ) == ("reward_informative_all_correct", True)

    def test_mixed_correctness_remains_eligible(self):
        assert grpo_gpu.group_reason(
            8, 16, 0.0, 0.5,
            novelty_contribution_std=0.0, validity_mode="c8",
        ) == ("informative", True)

    def test_dynamic_sampler_uses_novelty_signal_not_blended_signal(
            self, monkeypatch):
        rows = [
            {"is_correct": True, "blended_reward": reward,
             "novelty_used": 0.0, "finish_reason": "stop"}
            for reward in (1.0, 0.5)
        ]

        def fake_score(*_args, **_kwargs):
            return [dict(row) for row in rows]

        class Pool:
            def generate(self, prompts, n, **_kwargs):
                return [[Completion(text="x", finish_reason="stop") for _ in range(n)]
                        for _ in prompts]

        monkeypatch.setattr(grpo_gpu, "score_completions", fake_score)
        args = _args(
            "--rl-validity-mode", "c8", "--phase", "scout",
            "--prompts-per-step", "1", "--num-generations", "2",
            "--dynamic-max-attempts", "1", "--probe-every", "0",
        )
        example = SimpleNamespace(problem_id="p0", prompt_text="p", ground_truth="1")
        with pytest.raises(RuntimeError, match="dynamic sampling failed"):
            grpo_gpu._dynamic_sample_step(
                rng=np.random.default_rng(0), pool=Pool(), tokenizer=None,
                model=None, rnd=None, examples=[example], args=args, lam=0.5,
                step=1,
            )

    def test_legacy_dynamic_sampler_preserves_fixed_geometry_fallback(
            self, monkeypatch):
        rows = [
            {"is_correct": True, "blended_reward": reward,
             "novelty_used": 0.0, "finish_reason": "stop"}
            for reward in (1.0, 1.0)
        ]

        def fake_score(*_args, **_kwargs):
            return [dict(row) for row in rows]

        class Pool:
            def generate(self, prompts, n, **_kwargs):
                return [[Completion(text="x", finish_reason="stop") for _ in range(n)]
                        for _ in prompts]

        monkeypatch.setattr(grpo_gpu, "score_completions", fake_score)
        args = _args(
            "--rl-validity-mode", "legacy", "--phase", "scout",
            "--prompts-per-step", "1", "--num-generations", "2",
            "--dynamic-max-attempts", "1", "--probe-every", "0",
        )
        example = SimpleNamespace(problem_id="p0", prompt_text="p", ground_truth="1")
        selected, all_rows, stats, _ = grpo_gpu._dynamic_sample_step(
            rng=np.random.default_rng(0), pool=Pool(), tokenizer=None,
            model=None, rnd=None, examples=[example], args=args, lam=0.5,
            step=1,
        )

        assert len(selected) == 2
        assert stats["kept_group_count"] == 0.0
        assert stats["fallback_group_count"] == 1.0
        assert all(row["dynamic_sampling_reason"] == "fallback_all_correct"
                   for row in all_rows)


class TestC8ResumeContinuity:
    def test_fp32_loader_never_uses_bf16_fallback(self, monkeypatch):
        import transformers

        calls = []
        sentinel = torch.tensor([1.000000119], dtype=torch.float32)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(sentinel.clone())
                self.config = SimpleNamespace(use_cache=True)

            def gradient_checkpointing_enable(self, **_kwargs):
                return None

        class Auto:
            @staticmethod
            def from_pretrained(_path, *, torch_dtype, attn_implementation):
                calls.append((attn_implementation, torch_dtype))
                if attn_implementation == "flash_attention_2":
                    raise RuntimeError("synthetic FA2 FP32 rejection")
                return Model()

        monkeypatch.setattr(transformers, "AutoModelForCausalLM", Auto)
        model = grpo_gpu.load_policy(
            "checkpoint", device="cpu", param_dtype="float32")
        assert calls == [
            ("flash_attention_2", torch.float32),
            ("sdpa", torch.float32),
        ]
        assert model.weight.dtype == torch.float32
        assert torch.equal(model.weight.detach(), sentinel)
        assert model._tmx_attention_implementation == "sdpa"

        calls.clear()
        grpo_gpu.load_policy(
            "checkpoint", device="cpu", param_dtype="float32",
            expected_attention_implementation="sdpa")
        assert calls == [("sdpa", torch.float32)]

    def test_step_checkpoint_preserves_live_parameter_dtypes(self, tmp_path):
        class Model:
            def __init__(self):
                self.weights = {
                    "fp32": torch.tensor([1.000000119], dtype=torch.float32),
                    "bf16": torch.tensor([2.0], dtype=torch.bfloat16),
                }
                self.saved = None

            def state_dict(self):
                return self.weights

            def save_pretrained(self, output, *, state_dict, safe_serialization):
                Path(output).mkdir(parents=True, exist_ok=True)
                self.saved = state_dict
                assert safe_serialization is True
                (Path(output) / "config.json").write_text("{}")

        class Tokenizer:
            @staticmethod
            def save_pretrained(output):
                (Path(output) / "tokenizer.json").write_text("{}")

        model = Model()
        grpo_gpu.save_model_training_dtype(model, Tokenizer(), str(tmp_path))
        assert model.saved["fp32"].dtype == torch.float32
        assert torch.equal(model.saved["fp32"], model.weights["fp32"])
        assert model.saved["bf16"].dtype == torch.bfloat16

    def test_old_checkpoint_is_legacy_not_silently_c8(self):
        recorded = {"phase": "scout", "model": "m", "config_tier": "contract"}
        grpo_gpu.check_resume_compat(
            recorded, phase="scout", model="m", config_tier="contract")
        with pytest.raises(RuntimeError, match="rl_validity_mode"):
            grpo_gpu.check_resume_compat(
                recorded, phase="scout", model="m", config_tier="contract",
                rl_validity_mode_current="c8",
            )

    def test_matching_c8_checkpoint_certifies_optimizer_and_lr(self):
        recorded = {
            "phase": "scout", "model": "m", "config_tier": "contract",
            "rl_validity_mode": "c8", "optimizer_state_saved": True,
            "resolved_lr": 5e-6,
        }
        grpo_gpu.check_resume_compat(
            recorded, phase="scout", model="m", config_tier="contract",
            rl_validity_mode_current="c8", resolved_lr_current=5e-6,
        )
        with pytest.raises(RuntimeError, match="resolved_lr"):
            grpo_gpu.check_resume_compat(
                recorded, phase="scout", model="m", config_tier="contract",
                rl_validity_mode_current="c8", resolved_lr_current=1e-6,
            )

    def test_c8_checkpoint_without_optimizer_certification_is_rejected(self):
        with pytest.raises(RuntimeError, match="optimizer_state_saved"):
            grpo_gpu.check_resume_compat(
                {"phase": "scout", "model": "m", "config_tier": "contract",
                 "rl_validity_mode": "c8"},
                phase="scout", model="m", config_tier="contract",
                rl_validity_mode_current="c8",
            )

    def test_optimizer_state_round_trip(self, tmp_path):
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()

        path = grpo_gpu._save_optimizer_state(optimizer, str(tmp_path))
        assert path.endswith(grpo_gpu.OPTIMIZER_STATE_FILENAME)
        assert not (tmp_path / (grpo_gpu.OPTIMIZER_STATE_FILENAME + ".tmp")).exists()

        restored_model = torch.nn.Linear(3, 1)
        restored = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
        assert grpo_gpu._restore_optimizer_state(
            restored, str(tmp_path), device="cpu", required=True)

        before = optimizer.state_dict()
        after = restored.state_dict()
        assert before["param_groups"] == after["param_groups"]
        assert before["state"].keys() == after["state"].keys()
        for param_id in before["state"]:
            for key, value in before["state"][param_id].items():
                restored_value = after["state"][param_id][key]
                if torch.is_tensor(value):
                    assert torch.equal(value, restored_value)
                else:
                    assert value == restored_value

    def test_c8_missing_optimizer_state_fails_closed(self, tmp_path):
        optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())
        with pytest.raises(RuntimeError, match="requires optimizer state"):
            grpo_gpu._restore_optimizer_state(
                optimizer, str(tmp_path), device="cpu", required=True)
        assert not grpo_gpu._restore_optimizer_state(
            optimizer, str(tmp_path), device="cpu", required=False)

    @pytest.mark.parametrize("mode,expect_optimizer", [
        ("legacy", False), ("c8", True),
    ])
    def test_checkpoint_schema_is_explicit(
            self, tmp_path, monkeypatch, mode, expect_optimizer):
        output_dir = tmp_path / mode
        args = grpo_gpu.build_arg_parser().parse_args([
            "--output-dir", str(output_dir),
            "--rl-validity-mode", mode,
            "--keep-checkpoints", "1",
            "--ckpt-artifact-every", "0",
        ])
        optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())
        args._policy_attention_implementation = "sdpa"

        def fake_save(_model, _tokenizer, directory):
            # _save_checkpoint owns the directory contract in production.
            __import__("os").makedirs(directory, exist_ok=True)
            with open(__import__("os").path.join(directory, "config.json"), "w") as handle:
                handle.write("{}")

        monkeypatch.setattr(grpo_gpu, "save_model_training_dtype", fake_save)
        checkpoint = grpo_gpu._save_checkpoint(
            None, None, None, optimizer, args, 1, 0.5)
        state = json.loads((output_dir / "step_000001" /
                            "trainer_state.json").read_text())

        assert checkpoint == str(output_dir / "step_000001")
        assert state["rl_validity_mode"] == mode
        assert state["optimizer_state_saved"] is expect_optimizer
        assert state["checkpoint_parameter_dtype"] == "float32"
        assert state["checkpoint_file_manifest"]["sha256"]
        assert (output_dir / "step_000001" /
                grpo_gpu.OPTIMIZER_STATE_FILENAME).exists() is expect_optimizer

    def test_external_variant_step_checkpoint_and_resume_bind_full_lineage(
            self, tmp_path, monkeypatch):
        output_dir = tmp_path / "external"
        args = grpo_gpu.build_arg_parser().parse_args([
            "--output-dir", str(output_dir), "--phase", "central",
            "--scientific-variant", "external_sft_exposure_gated",
            "--rl-validity-mode", "c8", "--ckpt-artifact-every", "0",
        ])
        grpo_gpu.apply_completion_budget(args)
        grpo_gpu.enforce_probe_serving_len(args, pool=None)
        runtime_contract = grpo_gpu.build_runtime_contract(
            args, resolved_lr=1e-6, resolved_steps=100)
        lineage = {
            "lineage_contract_version": grpo_gpu.EXTERNAL_LINEAGE_CONTRACT_VERSION,
            "scientific_variant": "external_sft_exposure_gated",
            "model_source": {"artifact_ref": "model:v1", "artifact_digest": "d"},
            "central_runtime_contract": runtime_contract,
        }
        lineage["lineage_sha256"] = common.canonical_json_sha256(lineage)
        args._external_lineage = lineage
        args._runtime_contract = runtime_contract
        args._policy_attention_implementation = "sdpa"
        optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())

        def fake_save(_model, _tokenizer, directory):
            __import__("os").makedirs(directory, exist_ok=True)
            with open(__import__("os").path.join(directory, "config.json"), "w") as handle:
                handle.write("{}")

        monkeypatch.setattr(grpo_gpu, "save_model_training_dtype", fake_save)
        checkpoint = grpo_gpu._save_checkpoint(
            None, None, None, optimizer, args, 1, 0.0)
        state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        assert state["config_tier"] == \
            "contract+scientific-variant:external_sft_exposure_gated"
        assert state["external_lineage"] == lineage
        assert state["external_lineage_sha256"] == lineage["lineage_sha256"]
        grpo_gpu.check_resume_compat(
            state, phase="central", model=args.model,
            config_tier=state["config_tier"], rl_validity_mode_current="c8",
            resolved_lr_current=state["resolved_lr"],
            external_lineage_current=lineage,
            runtime_contract_current=runtime_contract)
        tampered = copy.deepcopy(lineage)
        tampered["model_source"]["artifact_digest"] = "other"
        tampered["lineage_sha256"] = common.canonical_json_sha256({
            key: value for key, value in tampered.items() if key != "lineage_sha256"})
        with pytest.raises(RuntimeError, match="external_lineage"):
            grpo_gpu.check_resume_compat(
                state, phase="central", model=args.model,
                config_tier=state["config_tier"], rl_validity_mode_current="c8",
                resolved_lr_current=state["resolved_lr"],
                external_lineage_current=tampered,
                runtime_contract_current=runtime_contract)
        drifted_runtime = copy.deepcopy(runtime_contract)
        drifted_runtime["prompts_per_step"] = 8
        with pytest.raises(RuntimeError, match="runtime_contract"):
            grpo_gpu.check_resume_compat(
                state, phase="central", model=args.model,
                config_tier=state["config_tier"], rl_validity_mode_current="c8",
                resolved_lr_current=state["resolved_lr"],
                external_lineage_current=lineage,
                runtime_contract_current=drifted_runtime)

    def test_external_lineage_rejects_resolved_budget_or_step_drift(self, tmp_path):
        args = grpo_gpu.build_arg_parser().parse_args([
            "--output-dir", str(tmp_path / "out"), "--phase", "central",
            "--scientific-variant", "external_sft_exposure_gated",
            "--rl-validity-mode", "c8", "--steps", "100",
            "--completion-budget", "16384", "--probe-every", "0",
        ])
        grpo_gpu.apply_completion_budget(args)
        runtime = grpo_gpu.build_runtime_contract(
            args, resolved_lr=1e-6, resolved_steps=100)
        lineage = {
            "lineage_contract_version": grpo_gpu.EXTERNAL_LINEAGE_CONTRACT_VERSION,
            "scientific_variant": "external_sft_exposure_gated",
            "central_runtime_contract": runtime,
        }
        lineage["lineage_sha256"] = common.canonical_json_sha256(lineage)
        lineage_path = tmp_path / "lineage.json"
        lineage_path.write_text(json.dumps(lineage))
        args.external_lineage_json = str(lineage_path)
        assert grpo_gpu.load_external_lineage(
            args, expected_runtime_contract=runtime) == lineage

        drifted = copy.deepcopy(runtime)
        drifted["resolved_max_completion_len"] = 32768
        with pytest.raises(ValueError, match="runtime contract"):
            grpo_gpu.load_external_lineage(
                args, expected_runtime_contract=drifted)
        one_step = copy.deepcopy(runtime)
        one_step["expected_steps"] = 1
        lineage["central_runtime_contract"] = one_step
        lineage["lineage_sha256"] = common.canonical_json_sha256({
            key: value for key, value in lineage.items()
            if key != "lineage_sha256"
        })
        lineage_path.write_text(json.dumps(lineage))
        with pytest.raises(ValueError, match="100 Central steps"):
            grpo_gpu.load_external_lineage(
                args, expected_runtime_contract=one_step)

    def test_deployable_final_schema_does_not_certify_optimizer(self):
        """Final/model-handoff artifacts are intentionally not resume points.

        Optimizer moments live only in step_* checkpoints, keeping evaluation
        artifacts lightweight while preserving complete crash recovery.
        """
        with pytest.raises(RuntimeError, match="optimizer_state_saved"):
            grpo_gpu.check_resume_compat(
                {
                    "phase": "central", "model": "m",
                    "config_tier": "contract", "rl_validity_mode": "c8",
                    "optimizer_state_saved": False,
                    "checkpoint_kind": "deployable_model",
                    "resolved_lr": 1e-6,
                },
                phase="central", model="m", config_tier="contract",
                rl_validity_mode_current="c8", resolved_lr_current=1e-6,
            )
