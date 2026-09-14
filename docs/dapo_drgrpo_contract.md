# DAPO / Dr.GRPO / Novelty Contract

This repo treats the following as the production RL contract for paper runs.
The validator lives in `tmx/orchestration/contract.py` and is called by
`ExperimentSpec.validate()`.

## DAPO Components

Primary source: <https://dapo-sia.github.io/>

DAPO contributes four constraints we enforce for RL stages:

- Clip-higher: `GRPO_CLIP_EPSILON=0.2` and `GRPO_CLIP_EPSILON_HIGH=0.28`.
- Dynamic sampling: grouped rollouts are sampled and non-informative groups are filtered.
- Token-level policy-gradient loss: the training backend must use token-level log-prob/loss accounting, not sequence-only scoring.
- Overlong reward shaping: the soft-overlong schedule must sum to the completion budget.

For the 16K production geometry:

```text
MAX_PROMPT_LEN=2048
MAX_COMPLETION_LEN=16384
MAX_TOTAL_LEN=18432
SOFT_OVERLONG_EXPECTED_LEN=13107
SOFT_OVERLONG_CACHE_LEN=3277
```

## Dr.GRPO

Primary source: <https://github.com/sail-sg/understand-r1-zero>

Dr.GRPO is used to avoid GRPO's length/std normalization bias. Production RL
stages therefore enforce:

```text
GRPO_ADVANTAGE_NORMALIZATION=none
GRPO_KL_BETA=0.0
```

## Novelty Explorer

Explorer novelty runs add RND-style novelty reward on top of correctness:

```text
LAMBDA_NOVELTY > 0
INCORRECT_NOVELTY_SCALE > 0
TMX_DYNAMIC_SAMPLING_ACCEPT_REWARD_VARIANCE=1
```

The last two are important. If every completion in a group is incorrect, a
correctness-only reward is constant and produces zero advantage. For novelty
Explorer, all-wrong groups can still be useful if novelty varies across the
group, so dynamic sampling may accept reward-variance-informative groups.

## Format Quality

Format quality is logged and used for SFT filtering, but it is not part of the
default production RL reward. The production reward is:

```text
blended_reward = correctness_reward + lambda_novelty * novelty_used
```

where incorrect completions may still receive `incorrect_novelty_scale *
novelty_reward` during Explorer novelty runs. Format penalties should only be
enabled intentionally for a separate guarded ablation, because they change the
reward contract and make the run harder to compare against the DAPO/Dr.GRPO
baseline.

## Infrastructure Boundary

Primary sources:

- Tunix: <https://tunix.readthedocs.io/>
- MaxText: <https://github.com/AI-Hypercomputer/maxtext>
- XPK: <https://github.com/AI-Hypercomputer/xpk>
- SGLang-JAX TPU serving: <https://docs.sglang.io/docs/hardware-platforms/tpu>

Target backend:

```text
XPK/GKE -> MaxText/Tunix -> vLLM-on-TPU or SGLang-JAX rollout -> HF artifacts
```

Current executable backend:

```text
raw TRC TPU VM -> tmx-jax -> vLLM-on-TPU -> HF artifacts
```

The MaxText/Tunix backend is scaffolded but blocked until a real container
adapter exists. The raw TPU fallback is the only launch path that has been
validated in this repo today.
