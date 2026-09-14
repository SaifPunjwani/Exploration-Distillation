"""Offline end-to-end smoke: real tiny Flax policy, AdamW, RND, SFT, Orbax.

Only the external dataset, rollout server, and artifact transports are replaced.
This checks stage integration; it does not test TPU Pallas or model accuracy.
"""
from dataclasses import asdict, replace
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from expdis_jax import pipeline, train, distill
from expdis_jax.config import TrainConfig
from expdis_jax.data import DapoExample
from expdis_jax.generate import Completion
from expdis_jax.model import Qwen3Config, Qwen3Model
from expdis_jax.weights import flax_params_to_hf_tensors


@pytest.mark.parametrize("host_accum", [False, True])
def test_real_rl_filter_sft_rl_checkpoint_pipeline(monkeypatch, tmp_path, host_accum):
    run_tiny_pipeline(monkeypatch, tmp_path, host_accum=host_accum)


def run_tiny_pipeline(monkeypatch, tmp_path, *, host_accum=False, checkpoint_root="", rounds=1, explorers=1):
    for name, value in {
        "EXPDIS_REQUIRE_WANDB": "0", "WANDB_MODE": "disabled",
        "EXPDIS_HF_MIRROR_CHECKPOINTS": "0", "EXPDIS_HF_MIRROR_TRAJECTORIES": "0",
        "EXPDIS_HF_MIRROR_FINAL_EXPORTS": "0", "EXPDIS_HF_MIRROR_METRICS": "0",
        "EXPDIS_SKIP_FINAL_EVAL": "1", "EXPDIS_SAVE_OPT_STATE": "1",
        "EXPDIS_FORCE_FINAL_PHASE_CHECKPOINT": "1",
        "EXPDIS_DISABLE_ROLLOUT_DIVERSITY": "1",
        "EXPDIS_BASE_POLICY_SNAPSHOT": "0", "EXPDIS_DAPO_DRGRPO_FULL_CONTRACT": "0",
        "EXPDIS_HOST_GRAD_ACCUM": str(int(host_accum)),
        "EXPDIS_HOST_GRAD_COMPACT_NONZERO_ADV": str(int(host_accum)),
        "EXPDIS_ENABLE_EXPERIMENTAL_TPU_MRME": "1",
    }.items():
        monkeypatch.setenv(name, value)
    model_dir = tmp_path / "qwen3-tiny"
    model_dir.mkdir()
    mcfg = Qwen3Config(hidden_size=8, intermediate_size=16, num_hidden_layers=4,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                       vocab_size=32, dtype=jnp.bfloat16, tie_word_embeddings=False)
    params = Qwen3Model(mcfg).init(jax.random.PRNGKey(0), jnp.ones((1, 8), dtype=jnp.int32),
                                  jnp.ones((1, 8), dtype=jnp.int32))["params"]
    hf_cfg = {k: v for k, v in asdict(mcfg).items() if k not in ("dtype", "param_dtype")}
    hf_cfg.update(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    (model_dir / "config.json").write_text(json.dumps(hf_cfg))
    save_file(flax_params_to_hf_tensors(params, 4, False), str(model_dir / "model.safetensors"))
    tokenizer = Tokenizer(models.WordLevel({"<pad>": 0, "<unk>": 1, "prompt": 2,
                                            "one": 3, "two": 4, "reason": 5,
                                            "1": 6, "2": 7}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
                            unk_token="<unk>").save_pretrained(model_dir)
    examples = [DapoExample("one", "prompt one", "1", "", ""), DapoExample("two", "prompt two", "1", "", "")]
    monkeypatch.setattr(train, "load_examples", lambda *a, **kw: examples)
    # The real validator is independently pinned at full paper geometry. A
    # tiny geometry is used here to exercise the full implementation on CPU.
    monkeypatch.setattr(train, "validate_contract", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "validate_contract", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_wandb_log_pipeline", lambda *a, **kw: None)
    monkeypatch.setattr(train, "_checkpoint_space_janitor", lambda *a, **kw: None)
    monkeypatch.setattr(train, "_maybe_upload_checkpoint_to_gcs", lambda *a, **kw: False)
    monkeypatch.setattr(distill, "_maybe_upload_checkpoint_to_gcs", lambda *a, **kw: False)
    snapshots = []
    initial_heads = {}
    def reload(state, cfg, m_cfg, step, *, reason):
        snapshots.append((cfg.output_dir, step, reason, int(state.step)))
        if reason == "stage_start" and jax.process_count() == 1:
            initial_heads[cfg.output_dir] = np.asarray(state.params["lm_head"]["kernel"]).copy()
    monkeypatch.setattr(train, "_maybe_export_and_reload_vllm", reload)
    def generate(prompts, *a, **kw):
        assert snapshots, "sampler must be synchronized before its first batch"
        assert all(isinstance(prompt, list) and len(prompt) <= 4 for prompt in prompts)
        return [[Completion(text, [], [], "stop", tokenizer.encode(text).ids)
                 for text in (r"reason \boxed{1}", r"reason \boxed{2}")] for _ in prompts]
    monkeypatch.setattr(train, "batched_generate", generate)
    cfg = replace(TrainConfig(), model_name=str(model_dir), output_dir=str(tmp_path / "run"),
                  checkpoint_root=checkpoint_root, num_rounds=rounds, scouts_per_round=explorers,
                  pipeline_mode="multi_round" if rounds > 1 else "two_model",
                  vllm_server_urls="http://unused/v1", lambda_novelty=0.5,
                  max_prompt_len=4, max_completion_len=16, max_total_len=20,
                  novelty_feature_max_length=16, novelty_layers=[1, 2, 3], rnd_hidden=8,
                  grpo_batch_size=2, grpo_num_generations=2, trajectory_batch_size=2,
                  grpo_max_steps=2 * rounds * explorers, actual_online_grpo_max_steps=rounds,
                  soft_overlong_expected_len=12, soft_overlong_cache_len=4,
                  accepted_min_completion_tokens=1, accepted_max_completion_tokens=16,
                  distill_max_total_len=20, min_accepted_trajectories=1,
                  save_every_steps=50, logprob_chunk_tokens=8)
    if rounds > 1:
        monkeypatch.setattr(pipeline, "parse_args", lambda: cfg)
        pipeline.multi_round_main()
    else:
        pipeline.main(cfg)
    if jax.process_index() != 0:
        return
    run = tmp_path / "run"
    if rounds > 1:
        run = run / f"round_{rounds}"
    explorer_relative = "explorer" if explorers == 1 else "explorers/scout_01"
    explorer_dir = run / explorer_relative
    rows = [json.loads(line) for line in (explorer_dir / "explorer_trajectories.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert all(row["r_novel_reward"] == 0 for row in rows if not row["is_correct"])
    accepted = [json.loads(line) for line in (run / "trajectory_library.accepted.jsonl").read_text().splitlines()]
    assert len(accepted) == 2
    assert all(row["is_correct"] and row["terminated"] for row in accepted)
    actual = [json.loads(line) for line in (run / "actual/grpo/explorer_trajectories.jsonl").read_text().splitlines()]
    assert len(actual) == 4
    assert all(row["r_novel_reward"] == 0 for row in actual)
    stage_snapshots = [(step, reason) for path, step, reason, _ in snapshots if path == str(explorer_dir)]
    assert stage_snapshots == [(0, "stage_start"), (1, "behavior_policy_sync"), (2, "behavior_policy_sync")]
    assert all(step == state_step for _, step, _, state_step in snapshots)
    from pathlib import Path
    saved = Path(checkpoint_root) if checkpoint_root else tmp_path / "run"
    if rounds > 1:
        saved = saved / f"round_{rounds}"
    assert (saved / "actual/actual_sft_final").is_dir()
    assert (saved / "actual/grpo/step_000001").is_dir()
    assert (run / "actual/actual_hf_final/model.safetensors").is_file()
    if jax.process_count() == 1:
        import orbax.checkpoint as ocp
        explorer = ocp.PyTreeCheckpointer().restore(str(saved / explorer_relative / "step_000002"))
        assert int(explorer["rnd_state"]["layer_1"]["opt_state"][0]["count"]) == 2
        main = ocp.PyTreeCheckpointer().restore(str(saved / "actual/grpo/step_000001"))
        # Both RL stages must change weights, not merely advance a counter.
        for directory, checkpoint in [(explorer_dir, explorer), (run / "actual/grpo", main)]:
            head = np.asarray(checkpoint["params"]["lm_head"]["kernel"])
            assert np.isfinite(head).all()
            assert np.any(head != initial_heads[str(directory)])
