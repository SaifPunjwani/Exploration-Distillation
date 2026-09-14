# Exploration-Distillation (ExpDis)

Exploration-Distillation separates *exploration* from *deployment* in
reinforcement learning for language-model reasoning. Two models are trained:

- **Explorer.** A disposable model trained with **Explorer RL**: the shared
  update described below, on a reward of correctness plus a novelty bonus.
  Novelty is Random Network Distillation (RND) over the Explorer's own hidden
  states and is credited **only to verifier-correct completions**, so
  exploration reorders correct solutions instead of trading correctness for
  surprise.
- **QualityPool filter.** The Explorer's trajectories are reduced to a small,
  clean transfer set: verifier-correct, naturally terminated, a valid boxed
  answer that matches the gold answer, no 40-character span repeated four
  times, 128–4,500 completion tokens, one shortest completion per problem
  (deterministic ties, fixed problem order), capped at 500.
- **Main model.** The deployed model. It is first fine-tuned (SFT) on the
  filtered set, then trained with **Main-model RL**: the same shared update on
  correctness plus soft-overlong shaping, with no novelty term. The Explorer's
  discoveries reach the Main model only through data.

**The shared update.** Both RL stages use one DAPO update with Dr.GRPO
normalization. The behavior policy is a snapshot frozen for each rollout batch:
the vLLM sampler is synchronized to the trainer before every batch, and the old
log-probabilities are those of that snapshot. Each batch is B = 4 prompt groups
× G = 16 completions. DAPO dynamic sampling keeps a group when its total
per-trajectory reward has variance and resamples otherwise (up to 8 attempts,
after which the batch is filled with the remaining groups). The objective uses
asymmetric clipping 0.2/0.28 on π_θ/π_old, which starts at 1 and normally does
not bind because there is exactly one optimizer step per batch; soft-overlong
shaping (onset 26,214 tokens, ramp 6,554, ending at the 32,768-token completion
budget); group-mean advantages with no standard-deviation division; and a fixed
loss denominator of 32,768 per row. Rewards are +1/−1 from the exact-rational
verifier; the Explorer adds λ · novelty on correct completions. In the code this
update is `expdis_jax/grpo.py` together with the `grpo_*` configuration flags.

Two extensions reuse the same pieces: **Multi-Explorer** trains K Explorers from
the same start with different seeds and merges their pools before the filter;
**Multi-Round** repeats Explorer → filter → Main for R rounds, splitting the
same total update budget (200 Explorer + 100 Main-model updates) across rounds
and seeding round r+1 from the round-r Main model, with λ annealed across
rounds. **Base** is the untouched reference model.

## Math prompt

Training and evaluation use `qwen3_math_user_suffix_v1`: append this instruction
to the question in the **user message**, without adding a system message:

> Please reason step by step, and put your final answer within \boxed{}.

Use the checkpoint's chat template, with thinking enabled for Qwen3. The same
instruction applies to Ministral's text decoder; its native template defaults
are retained. Main-model SFT retains the exact
prompt of each accepted trajectory. Explicit answer-only or teacher ablations
have separate contracts.

This matches [Qwen's stated evaluation placement](https://github.com/QwenLM/Qwen3/discussions/1395)
and [math instruction](https://huggingface.co/Qwen/Qwen3-1.7B).
It replaces the earlier training/evaluation prompt split in this release on
2026-09-17. Existing checkpoints, generation pools, and paper scores are not
relabelled as runs of the new protocol; those require their original run records.
New evaluations must use new output directories.

Prompt consolidation does not change the paper's 32,768-token completion budget.
The [Qwen3 report](https://arxiv.org/html/2505.09388v1) used 38,912 for AIME24/25.


## Naming in the code

`expdis_jax`, `expdis_torch`, and the `EXPDIS_*` environment variables carry the
paper's name. Three legacy role names remain because checkpoints, artifact
paths, W&B runs, and configs use them: `Scout` is the **Explorer**, and
`Central` / `Actual` are the **Main model**.

Public training flags use `--explorers-per-round`, `--main-rl-max-steps`,
`--main-rl-lr`, and `--main-parent-checkpoint`. Parallel manifests accept the
same names with underscores. Legacy flags remain aliases; saved checkpoint
paths and serialized fields retain their existing names for compatibility.

| Paper | Code |
|---|---|
| Explorer | `Scout`, `explorer_*` (e.g. `explorer_trajectories.jsonl`, `scouts_per_round`) |
| Main model | `Actual`, `Central`, `actual_*`, `central_*` (e.g. `actual_sft_final`, `actual_grpo_lr`, `central_init_policy`) |
| Base | `base` |
| Explorer RL / Main-model RL (the shared update) | `grpo.py`, `train.py::run_training`, the `grpo_*` flags (`grpo_max_steps`, `actual_online_grpo_max_steps`, `grpo_clip_epsilon`, `dr_grpo_loss_normalizer`, ...) |
| novelty weight λ | `lambda_novelty`, `LAMBDA_NOVELTY`, `round_novelty_schedule` |

## Implementations

- **`expdis_jax/` — the main implementation (JAX / Flax on TPU).** Full
  pipeline: Explorer RL with RND novelty → QualityPool → Main-model SFT →
  Main-model RL → evaluation probe, plus the Multi-Explorer and Multi-Round
  drivers. Start with [`expdis_jax/README.md`](expdis_jax/README.md).
- `expdis_torch/` — the original torch_xla pilot, kept as the record of how the
  work was first done. Entry point: `run_expdis_torch.py`.
- `tmx_gpu/` on the **`gpu`** branch — the GPU (PyTorch + vLLM) campaign and
  measurement stack. Its historical contracts differ from the latest paper;
  see that branch's `GPU_AUDIT.md` before treating a run as a reproduction.

| Branch | Contents |
|---|---|
| **`main`** (default) | `expdis_jax/` main implementation; `expdis_torch/` pilot |
| **`gpu`** | `tmx_gpu/` GPU stack |

Historical experiment snapshots are not part of the public release.

### Verification status

The JAX implementation has offline tests for the loss and reward rules, sampling
boundaries, filtering, checkpoint handoffs, and a tiny real Explorer → SFT → Main
pipeline. Optional numerical tests compare Qwen3 and Ministral3 forwards with
Hugging Face. The actual Pallas attention kernel is also tested in CPU
interpretation mode, including its backward pass and padding alignment. See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the reviewed commits,
fixes, test commands, and unresolved reproduction requirements.

Two-process CPU tests exercise full stage handoffs and two-round, two-Explorer
execution with shared checkpoints. The native driver executes Explorers
sequentially; `expdis_jax.parallel_pipeline` launches independent groups
concurrently on an explicitly partitioned allocation. Passing CPU tests does
not establish TPU/vLLM integration, equal wall time, or reported benchmark gains.
Those require full training and evaluation with retained run artifacts.

## How the method maps to the code (`expdis_jax/`)

| Stage | Where |
|---|---|
| Behavior-policy snapshot: sampler synchronized to the trainer before every rollout batch | `train.py::_maybe_export_and_reload_vllm` (`EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS=1`), `scripts/external_vllm_reload_watcher.sh` |
| Rollouts from the vLLM serving copy | `generate.py` (`batched_generate`) |
| Verifier and reward (+1/−1, soft-overlong shaping) | `rewarding.py`, `train.py::_score_rollouts` |
| RND novelty on pooled hidden states, credited on correct completions | `novelty.py`, `model.py` (`return_pooled_layers`), `train.py::_score_rollouts` |
| The shared update: DAPO with Dr.GRPO normalization (group-mean advantages, fixed 32,768 denominator, dynamic sampling, clip 0.2/0.28, truncated completions masked) | `grpo.py`, `train.py::_dynamic_sample_rollouts`, `train.py::_build_rollout_batch`, `train.py::run_training` |
| Explorer RL stage (λ > 0) and Main-model RL stage (λ = 0) | `train.py::run_training`, driven by `pipeline.py::main` |
| QualityPool filter | `filtering.py` (`pool_trajectory_files`), `pipeline.py::collect_accepted` |
| Main-model SFT on the filtered set | `distill.py::run_sft` |
| Multi-Explorer (K Explorers, merged pools) and Multi-Round (R rounds, budget split, λ schedule) | `pipeline.py::main`, `pipeline.py::multi_round_main`, `lineage.py` |
| Run contract (geometry, lengths, loss constants, gated novelty), fail-closed at startup | `config.py::validate_contract` |
| P1 evaluation and in-training probes (avg@n, pass@k through 64, diversity) | `eval.py` |
| Model, weights, sharding | `model.py`, `weights.py`, `mesh.py` |

For Ministral, use the publisher's [BF16 instruct checkpoint](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
as `--model-name`; the unsuffixed FP8 checkpoint is unsupported for training.
The JAX environment requires Transformers 5.17+ for current tokenizer metadata.

## Try the pipeline on CPU

From the repository root, with Python 3.11 installed:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest tests/test_expdis_jax_pipeline_cpu.py -q
```

Expected result: **2 passed**. After package installation this test needs no
network, credentials, pretrained weights, or accelerator. It runs real Explorer
RL, filtering, Main SFT and Main RL on a tiny local Flax model, and checks that
both RL stages change their weights. Dataset input and vLLM serving use fixtures.

Run the full suite with `.venv/bin/python -m pytest tests -q`; tests needing an
optional PyTorch reference environment skip when it is absent. For four
simulated CPU devices, prefix the pipeline test command with
`XLA_FLAGS=--xla_force_host_platform_device_count=4`.

Full training requires TPU compute, compatible vLLM serving, and weight-reload
configuration. Follow [`expdis_jax/README.md`](expdis_jax/README.md#training-on-tpu)
for that setup. The CPU lock is a test environment; TPU setup installs its own
runtime. An accelerator run is still needed to validate the deployed stack.

The historical torch pilot's CLI is `python run_expdis_torch.py --help`;
its training commands download models and data and are not offline sanity tests.

## Repository layout (`main`)

| Path | Purpose |
|---|---|
| `expdis_jax/` | Main implementation (see its README for modules, contract, launchers) |
| `expdis_torch/`, `run_expdis_torch.py` | Original torch_xla pilot |
| `scripts/` | Pilot launchers and TPU utilities |
| `configs/` | Campaign/benchmark TSVs and TPU accelerate configs for the pilot |
| `tests/` | Unit tests for both implementations |

## Notes

- Runtime caches default to repo-local `.hf/` unless `HF_HOME`, `HF_HUB_CACHE`,
  or `HF_DATASETS_CACHE` are set.
- Tracked runs fail fast without W&B credentials (`wandb login` or
  `WANDB_API_KEY`); set `EXPDIS_REQUIRE_WANDB=0` to run without W&B.
- `analysis/`, paper artifacts, and experiment outputs are intentionally
  untracked (see `.gitignore`).
- Pre-rename external state keeps its historical names on purpose: GCS paths
  under `gs://two-model-exploration-checkpoints/` (and the older
  `gs://llmvae-checkpoints/two-model-exploration/`), the HF repo
  `SaifPunjwani/two-model-exploration-checkpoints`, and the W&B project
  `two-model-explore-gpu` for GPU-era receipts. Renaming live storage would
  orphan checkpoints, caches, and receipts. New runs log to the W&B project
  `expdis`.
