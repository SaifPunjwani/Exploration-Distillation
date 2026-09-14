#!/usr/bin/env python3
"""Debug old/new logprob path consistency on TPU without vLLM.

Builds a tiny synthetic rollout batch from DAPO prompts + fixed completions and
compares:
  1. host-local rescoring path
  2. sharded rescore_jit path
  3. exact no-grad train-path logprob path
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer

from expdis_jax.config import TrainConfig
from expdis_jax.data import load_dapo
from expdis_jax.generate import Completion
from expdis_jax.grpo import compute_group_advantages
from expdis_jax.mesh import build_mesh, named
from expdis_jax.train import (
    _build_rollout_batch,
    _init_state,
    _rescore_old_logps,
    _to_global_batch,
    init_distributed,
    make_rescore_step,
    make_trainpath_logprob_step,
)


def main() -> None:
    init_distributed()

    cfg = TrainConfig(
        model_name="Qwen/Qwen3-1.7B",
        dataset_name="dapo_math_17k",
        max_train_examples=2,
        max_prompt_len=2048,
        max_completion_len=8192,
        max_total_len=10240,
        grpo_batch_size=1,
        grpo_grad_accum=2,
        grpo_num_generations=8,
        grpo_max_steps=1,
        grpo_loss_type="dr_grpo",
        optimizer="adamw",
        grpo_lr=5e-6,
        trajectory_batch_size=2,
        lambda_novelty=0.0,
        rescore_chunk_size=1,
        logprob_chunk_tokens=64,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_dapo(tokenizer, max_examples=2, enable_thinking=cfg.enable_thinking)
    examples = examples[:2]

    completions = []
    for ex in examples:
        good = f"Let us solve it carefully. Final answer: \\boxed{{{ex.ground_truth}}}"
        bad = "I am unsure. Final answer: \\boxed{0}"
        group = []
        for i in range(cfg.grpo_num_generations):
            txt = good if (i % 2 == 0) else bad
            group.append(Completion(text=txt, token_ids=[], token_logprobs=[], finish_reason="stop"))
        completions.append(group)

    batch_local = _build_rollout_batch(tokenizer, examples, completions, cfg)
    # Ensure non-degenerate advantages.
    rewards = np.asarray(batch_local["rewards"])
    if np.allclose(rewards.std(), 0.0):
        rewards = np.asarray([1.0 if i % 2 == 0 else 0.0 for i in range(rewards.shape[0])], dtype=np.float32)
        batch_local["rewards"] = jnp.asarray(rewards)
        batch_local["advantages"] = compute_group_advantages(jnp.asarray(rewards), cfg.grpo_num_generations)

    mesh = build_mesh()
    print(f"[debug] devices={len(jax.devices())} process_count={jax.process_count()} process_index={jax.process_index()}", flush=True)

    with mesh:
        state, _model, m_cfg, _param_sh = _init_state(cfg, mesh)
        rows_per_microbatch = int(cfg.grpo_batch_size) * int(cfg.grpo_num_generations)
        step_batch_pspec = {
            "full_input_ids": P("fsdp", None),
            "full_attention_mask": P("fsdp", None),
            "completion_mask": P(),
            "old_per_token_logps": P(),
            "advantages": P(),
        }
        batch = _to_global_batch(
            batch_local,
            mesh,
            {
                "full_input_ids": step_batch_pspec["full_input_ids"],
                "full_attention_mask": step_batch_pspec["full_attention_mask"],
                "completion_mask": step_batch_pspec["completion_mask"],
                "advantages": step_batch_pspec["advantages"],
            },
        )

        rescore_fn = make_rescore_step(
            int(cfg.max_prompt_len),
            bool(m_cfg.tie_word_embeddings),
            int(cfg.logprob_chunk_tokens),
            rows_per_microbatch,
            int(cfg.grpo_grad_accum),
        )
        trainpath_fn = make_trainpath_logprob_step(
            int(cfg.max_prompt_len),
            bool(m_cfg.tie_word_embeddings),
            int(cfg.logprob_chunk_tokens),
            rows_per_microbatch,
            int(cfg.grpo_grad_accum),
        )
        rescore_jit = jax.jit(rescore_fn, out_shardings=named(mesh, step_batch_pspec["old_per_token_logps"]))
        trainpath_jit = jax.jit(trainpath_fn, out_shardings=named(mesh, step_batch_pspec["old_per_token_logps"]))

        lp_host = _rescore_old_logps(
            state,
            batch_local["full_input_ids"],
            batch_local["full_attention_mask"],
            prompt_len=int(cfg.max_prompt_len),
            tie_word_embeddings=bool(m_cfg.tie_word_embeddings),
            batch_chunk=1,
            token_chunk=int(cfg.logprob_chunk_tokens),
        )
        lp_rescore = rescore_jit(state, batch)
        lp_trainpath = trainpath_jit(state, batch)

        lp_host = np.asarray(lp_host)
        lp_rescore = np.asarray(lp_rescore)
        lp_trainpath = np.asarray(lp_trainpath)
        mask = np.asarray(batch_local["completion_mask"]).astype(np.float32)
        denom = max(mask.sum(), 1.0)

        def summarize(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
            gap = np.abs(a - b) * mask
            print(
                f"[debug] {name_a}_vs_{name_b} "
                f"gap_mean={gap.sum()/denom:.3e} gap_max={gap.max():.3e}",
                flush=True,
            )

        summarize("host", lp_host, "rescore_jit", lp_rescore)
        summarize("host", lp_host, "trainpath_jit", lp_trainpath)
        summarize("rescore_jit", lp_rescore, "trainpath_jit", lp_trainpath)
        print("[debug] sample host row0", lp_host[0, :8].tolist(), flush=True)
        print("[debug] sample rescore row0", lp_rescore[0, :8].tolist(), flush=True)
        print("[debug] sample trainpath row0", lp_trainpath[0, :8].tolist(), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
    os.environ.setdefault("TPU_VISIBLE_CHIPS", "0,1,2,3")
    main()
