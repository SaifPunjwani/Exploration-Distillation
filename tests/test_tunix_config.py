from __future__ import annotations

from dataclasses import replace

from tmx.orchestration.spec import load_experiment_spec
from tmx.orchestration.tunix_config import (
    emit_command,
    emit_overrides,
    emit_programmatic_argv,
    emit_programmatic_overrides,
)


def _load_resume_spec():
    return load_experiment_spec(
        "configs/tunix/tmx_dapo_drgrpo_resume_step100.json"
    )


def _load_smoke_spec():
    return load_experiment_spec("configs/tunix/tmx_smoke_tiny.json")


def test_resume_spec_validates() -> None:
    spec = _load_resume_spec()
    result = spec.validate()
    assert result.ok, f"errors={list(result.errors)}"


def test_cli_rejects_novelty_specs() -> None:
    spec = _load_resume_spec()
    try:
        emit_overrides(spec)
    except ValueError as exc:
        assert "novelty/RND" in str(exc)
    else:
        raise AssertionError("novelty specs must not silently use the Tunix CLI path")


def test_overrides_lock_correctness_only_grpo_contract() -> None:
    spec = _load_smoke_spec()
    flat = " ".join(emit_overrides(spec))
    assert "grpo_config.epsilon=0.2" in flat
    assert "grpo_config.beta=0.0" in flat
    assert "grpo_config.num_generations=4" in flat
    assert "model_source=maxtext" in flat
    assert "rollout_config.total_generation_steps=512" in flat
    assert "rollout_config.max_prompt_length=256" in flat
    assert "actor_model_config.mesh.shape=(1,4)" in flat


def test_overrides_use_tmx_reward_module() -> None:
    spec = _load_smoke_spec()
    flat = " ".join(emit_overrides(spec))
    assert 'reward_functions=["tmx.orchestration.tunix_rewards"]' in flat


def test_overrides_set_resume_checkpoint_for_cli_baselines() -> None:
    base = _load_smoke_spec()
    spec = replace(
        base,
        resume=replace(
            base.resume,
            fresh_run=False,
            init_checkpoint="hf://SaifPunjwani/two-model-exploration-checkpoints/test/step_000001",
        ),
    )
    flat = " ".join(emit_overrides(spec))
    assert "model_path=hf://" in flat


def test_overrides_use_vllm_jax_rollout() -> None:
    spec = _load_smoke_spec()
    flat = " ".join(emit_overrides(spec))
    assert "rollout_engine=vllm" in flat
    assert "rollout_config.rollout_vllm_tpu_backend_type=jax" in flat


def test_overrides_no_gcs_paths() -> None:
    spec = _load_smoke_spec()
    for o in emit_overrides(spec):
        assert "gs://" not in o, o


def test_emit_command_starts_with_tunix_grpo_main() -> None:
    spec = _load_smoke_spec()
    cmd = emit_command(spec)
    # cmd[0] is sys.executable; -m tunix.cli.grpo_main follows
    assert cmd[1] == "-m"
    assert cmd[2] == "tunix.cli.grpo_main"


def test_programmatic_overrides_include_dapo_drgrpo_contract() -> None:
    spec = _load_resume_spec()
    flat = " ".join(emit_programmatic_overrides(spec))
    assert "grpo_config.advantage_estimator=drgrpo" in flat
    assert "grpo_config.loss_agg_mode=sequence-mean-token-scale" in flat
    assert "grpo_config.epsilon_high=0.28" in flat
    assert "grpo_config.dynamic_sampling=true" in flat
    assert "grpo_config.beta=null" in flat
    assert "grpo_config.overlong_buffer.overlong_buffer_length=3277" in flat
    assert "grpo_config.overlong_buffer.max_response_length=16384" in flat
    assert "actor_model_config.mesh.shape=(16,4)" in flat
    assert "rollout_model_config.mesh.shape=(48,4)" in flat
    assert "reference_model_config.same_mesh_as=actor" in flat
    assert "gs://" not in flat


def test_programmatic_argv_targets_tmx_pipeline_not_plain_cli() -> None:
    spec = _load_resume_spec()
    argv = emit_programmatic_argv(spec)
    assert argv[0] == "tmx-tunix-main"
    assert any(x == "grpo_config.advantage_estimator=drgrpo" for x in argv)
