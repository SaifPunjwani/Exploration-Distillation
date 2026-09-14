# c7 — Frozen Campaign Configuration & 32k Audit

Generation c7 (2026-07-08) is the FROZEN primary configuration. No parameter
changes without a hard failure (crash, non-finite training, sustained
collapse); any future change is a separately named generation.

## Configuration (identical geometry across every arm)

- 32,768-token completion budget — intentional GPU-optimized reproduction,
  NOT strict TPU parity. Prompt 2048; vLLM max-model-len 36,864.
- Reward: correctness +1/−1; λ·RND novelty on CORRECT completions only
  (incorrect_novelty_scale=0, measured-flagship value); soft-overlong
  −min(1,(len−26,214)/6,554); KL β=0.
- Loss: Dr.GRPO token-level, fixed denominator rows × **16,384**
  (--loss-denominator-len; deliberately NOT scaled with the budget — the
  normalizer is an arbitrary length-bias constant, pinning it preserves
  gradient scale parity with the TPU runs and across budget tiers).
- Advantages: group-mean only, 16 generations/prompt, 4 prompts/step
  (64 rollouts/step); REINFORCE semantics (1 update/rollout, clip inert).
- DAPO components: dynamic sampling (0<c<16 + λ-variance escape),
  Overlong Filtering ON (clipped rows masked from loss; at 32k this should
  bind rarely — expected ~2–4%), asymmetric clip constants 0.2/0.28
  (inert under REINFORCE, active only in --ppo-mode experiments).
- RND: layers 7/14/21, feature window = 32,768 (extended with the budget —
  full reasoning context per the paper; truncation fraction logged).
- Optimizers: AdamW 5e-6 scout / 1e-6 central / 5e-6 SFT; grad clip 1.0.
- Training sampling T=1.0/0.95/20; probes: AIME24 30×4 @ T=0.6, 32,768
  tokens, every 25 steps. Weight reload every 10 steps; checkpoints every 25.

## Token/compute comparability across arms

- DAPO: 300 scout-only steps. ExpDis SingleScout: 200 scout + 100 central
  (+ SFT, ~1–2k short supervised steps-equivalent). Both = 300 RL steps ×
  64 rollouts; rollout token budgets identical. MultiScout scouts (seeds
  1/2): 200 steps each, pooled with SingleScout's scout (seed 0) for K=3.

## Provenance

- git branch gpu-repro @ SHA recorded per commit (freeze tag: c7-freeze-20260708).
- Code artifact: <wandb-entity>/two-model-explore-gpu/tmx-gpu-code:v15.
- Container: vllm/vllm-openai:latest at launch time — runtime-verified in every
  job log: vllm 0.24.0, transformers 5.12/5.13, torch 2.9-2.12 (exact per-job
  versions printed in the [boot] line and preserved in joblogs artifacts).
- Jobs (A100-80GB; deployment hostname omitted): tmx-jrl-singlescout-lam05 (8×),
  tmx-dapo-baseline, tmx-jrl-scout-s1, tmx-jrl-scout-s2 (4× each).
- W&B runs: gpu-{jrl-singlescout-lam05|dapo-correctness|jrl-scout-lam05-seed1/2}-32k-c7-*;
  full config in each run's W&B config incl. config_tier.

## Reporting commitments

At step-25 probes and every 25 thereafter, and at each stage boundary,
report per arm: eval/AIME24_avg, clipped_rate, valid_answer_rate,
mean_completion_tokens, train/correctness_rate, grad_norm,
masked_truncated_rows/rows_backward, rollout+update time. Negative results
are preserved and reported — projections are sanity references, never
tuning targets.
