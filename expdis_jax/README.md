# expdis_jax — the main Exploration-Distillation implementation (JAX / Flax on TPU)

Native JAX implementation of the full Exploration-Distillation pipeline:
Explorer RL with RND novelty → QualityPool filter → Main-model SFT → Main-model
RL → evaluation probe, plus the Multi-Explorer and Multi-Round drivers. Built
for v4/v5e/v6e TPUs with 1-D FSDP sharding (GSPMD + NamedSharding) and Pallas
flash attention.

Naming: `expdis_jax` and the `EXPDIS_*` environment variables carry the paper's
name. Three legacy role names remain because checkpoints, artifact paths, W&B
runs, and configs use them: `Scout` / `explorer_*` is the paper's **Explorer**,
`Central` / `Actual` / `actual_*` is the **Main model**, and **Base** is the
untouched reference model.
Public flags use Explorer/Main names (`--explorers-per-round`,
`--main-rl-max-steps`, `--main-rl-lr`, `--main-parent-checkpoint`); the old
spellings remain aliases. Checkpoint paths and serialized fields are unchanged.

## The shared update

Both RL stages use one DAPO update with Dr.GRPO normalization, implemented in
`grpo.py` and driven by `train.py::run_training` through the `grpo_*` flags:

- **Frozen behavior policy per rollout batch.** Before every rollout batch the
  vLLM serving copy is synchronized to the trainer
  (`EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS=1`, `_maybe_export_and_reload_vllm`),
  so rollouts and their old log-probabilities come from that snapshot. With one
  optimizer step per batch the snapshot's log-probabilities equal the current
  policy's at the start of the step, so they are evaluated inline
  (`old_lp = stop_gradient(new_lp)`; `make_rescore_step` rescores them once per
  batch when `grpo_updates_per_rollout > 1`). vLLM log-probabilities are never
  used.
- **Geometry.** B = 4 prompt groups × G = 16 completions = 64 rows per update.
- **Dynamic sampling.** A group is kept when its total per-trajectory reward
  (correctness + gated novelty + soft-overlong shaping) has variance; otherwise
  fresh groups are sampled, up to 8 attempts, after which the remaining groups
  fill the batch (`_dynamic_sample_rollouts`; the update is never skipped).
- **Objective.** Asymmetric clipping 0.2/0.28 on π_θ/π_old — the ratio starts
  at 1, so the clip normally does not bind; KL β = 0; group-mean advantages with
  no standard-deviation division; fixed loss denominator 32,768 per row (the
  completion budget, independent of the realized length); truncated completions
  masked out of the loss (DAPO overlong filtering).
- **Rewards.** +1/−1 from the exact-rational verifier plus soft-overlong shaping
  (onset 26,214 tokens, ramp 6,554, ending at the 32,768-token cap). The
  Explorer adds λ · novelty, credited only on verifier-correct completions
  (`incorrect_novelty_scale=0`); the Main-model stage has no novelty term.
- **Optimizer.** AdamW β = 0.9/0.95, ε = 1e-8, no weight decay, gradient clip
  1.0; LR 5e-6 for Explorer RL, 1e-6 for Main-model RL, 5e-6 for SFT; fp32
  parameters, bf16 compute.

## What runs, in order

1. **Explorer RL** (`train.py::run_training`, λ > 0, 200 updates). Each update
   synchronizes the serving copy, samples 4 prompts × 16 completions from it
   (`generate.py`), scores them (`_score_rollouts`: verifier reward ±1,
   soft-overlong shaping, RND novelty on correct completions), applies dynamic
   sampling, and takes one step of the shared update. Checkpoints are exported
   every `SAVE_EVERY_STEPS` (50) updates; that cadence is independent of the
   per-batch synchronization.
2. **QualityPool filter** (`filtering.py`, `pipeline.py::collect_accepted`).
   Keeps completions that are verifier-correct, naturally terminated, carry a
   valid boxed answer matching the gold answer, contain no 40-character span
   repeated four times, and are 128–4,500 tokens long; keeps the shortest
   qualifying completion per problem (deterministic ties, fixed problem order)
   and caps the set at 500. Rewards play no role in selection.
3. **Main-model SFT** (`distill.py::run_sft`). Two epochs, batch 1, LR 5e-6,
   completion-only token-mean cross-entropy with prompt tokens masked, fresh
   optimizer, at most 500 examples.
4. **Main-model RL** (`train.py::run_training`, λ = 0, 100 updates, LR 1e-6).
   Correctness plus soft-overlong shaping, no novelty.
5. **Evaluation** (`eval.py`). The final AIME24 evaluation defaults to 64
   rollouts and 32,768 completion tokens under P1. Smaller in-training probes
   are distinct from full benchmark evaluation. Additional benchmarks accept
   local JSONL snapshots; see below.

`pipeline.py::main` runs stages 1–5 once. `pipeline.py::multi_round_main` runs
R rounds: the prompt set is split into R disjoint shards, the Explorer and Main
budgets are split evenly across rounds (`lineage.py`), round r+1 initializes
both models from the round-r Main model, RND and optimizer state are reset every
round, and λ follows `--round-novelty-schedule`. Multi-Explorer (`explorers_per_round`
= K) trains K Explorers from the same start with different seeds and merges their
pools before the filter. Every arm consumes the same total number of updates.

The single-model control is the Explorer RL stage alone with λ = 0 and 300
updates (`EXPDIS_PIPELINE_STOP_AFTER_EXPLORER=1 LAMBDA_NOVELTY=0 GRPO_MAX_STEPS=300`).

## Contract (enforced by `config.py::validate_contract`)

Two tiers, both fail-closed at startup:

- **Base contract** (always on; default model/data are Qwen/Qwen3-1.7B and DAPO-Math-17k):
  `MAX_PROMPT_LEN=2048`, `MAX_COMPLETION_LEN=32768`, `MAX_TOTAL_LEN=34816`,
  4 prompts × 16 generations = 64 rows per update, Dr.GRPO loss
  (`grpo_loss_type=dr_grpo`), AdamW only, z-scoring off,
  `format_penalty_weight=0`, novelty gate fields pinned, accepted/SFT cap 500,
  accepted completion window 128–4,500 tokens. Launcher geometry is matched
  against the `EXPDIS_EXPECTED_GRPO_*` env pins the launch scripts export.
- **Full DAPO/Dr.GRPO contract** (`EXPDIS_DAPO_DRGRPO_FULL_CONTRACT=1`, on by
  default in the CLI and launchers): 16 generations, clip 0.2/0.28, KL 0, reward +1/−1,
  `incorrect_novelty_scale=0` (novelty only on correct completions), dynamic
  sampling and overlong filtering on, `dr_grpo_loss_normalizer=32768`
  (= the completion budget), soft-overlong shaping (onset 26214, ramp 6554,
  ending at the 32,768 cap), one optimizer
  update per rollout batch.

`config.py` is the source of truth; `tests/test_expdis_jax_invariants.py` pins the
kernel math (mean-centered advantages with no std division, fixed 32,768
denominator) and the fail-closed validation; `tests/test_expdis_jax_novelty.py`
pins the RND update order and optimizer-state lifecycle.

## Training on TPU

Start with the [CPU pipeline test](../README.md#try-the-pipeline-on-cpu) to check
your installation. Full training needs the serving and storage configuration
below; its accelerator integration has not yet been validated by this audit.

Prerequisites: a Linux x86_64 TPU VM with glibc 2.31 or newer for training,
one or more vLLM servers (OpenAI-compatible
`/v1/completions`) serving the same base model, and either W&B credentials or
`EXPDIS_REQUIRE_WANDB=0`. The HF transport needs a dataset repo you can write in
`EXPDIS_HF_CHECKPOINT_REPO`, a write token on the trainer, and read access on the
serving hosts. Checkpoint and trajectory mirrors can be disabled separately;
disabling those does not remove the need to transfer serving weights.
The per-batch sampler synchronization ships a reload bundle
through the configured transport (`EXPDIS_VLLM_RELOAD_TRANSPORT`, HF mirror by
default via `EXPDIS_HF_MIRROR_VLLM_EXPORTS=1`) and blocks until the serving copy
acknowledges it (`EXPDIS_EXTERNAL_VLLM_RELOAD_WAIT=1`,
`EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED=1`); with no transport configured,
training stops instead of sampling from stale weights.

The vLLM server must support `return_token_ids` on `/v1/completions`. Training
sends prompt token IDs and retains sampled completion IDs, including terminal
tokens, through rewards, novelty features, policy loss, and SFT. Missing action
IDs fail the paper contract. The [vLLM API description](https://vllm.ai/blog/2025-10-22-agent-lightning)
explains this requirement. Real serving reloads and TPU kernels still need an
accelerator smoke test; the offline tests replace that transport.

```bash
# On the training TPU VM (single host):
bash expdis_jax/scripts/setup_jax_tpu.sh

# From a machine with gcloud/IAP access to the serving TPU slice:
TPU_NAME="<serving-slice>" ZONE="<serving-zone>" WORKERS="0 1 2 3" \
MAX_MODEL_LEN=40960 \
bash expdis_jax/scripts/bootstrap_vllm_slice.sh

# On the control machine, keep this watcher running in a separate terminal.
# REMOTE_HOST is an SSH alias that reaches the training VM.
REMOTE_HOST="<trainer-ssh-alias>" \
REMOTE_RUNS_ROOT="/home/<trainer-user>/expdis/runs" \
bash expdis_jax/scripts/external_vllm_reload_watcher.sh

# On the training VM, set these for both the pipeline and baseline commands:
export EXPDIS_VLLM_SERVER_URLS="http://<vllm_ip>:8000/v1"
export EXPDIS_HF_CHECKPOINT_REPO="<you>/<dataset-repo>"
export EXPDIS_VLLM_RELOAD_TPU_NAME="<serving-slice>"
export EXPDIS_VLLM_RELOAD_ZONE="<serving-zone>"
export EXPDIS_VLLM_RELOAD_WORKERS="0 1 2 3"
export EXPDIS_REQUIRE_WANDB=0   # optional; use wandb login to enable tracking

# Full pipeline (Explorer RL -> QualityPool -> SFT -> Main-model RL -> eval):
LAMBDA_NOVELTY=0.5 \
RUN_NAME=expdis_$(date +%Y%m%d_%H%M%S) \
bash expdis_jax/scripts/launch_jax_pipeline.sh

# MR-ME: the same total 200 Explorer + 100 Main RL updates across four rounds.
NUM_ROUNDS=4 EXPLORERS_PER_ROUND=3 ROUND_NOVELTY_SCHEDULE=0.75,0.50,0.35,0.25 \
bash expdis_jax/scripts/launch_jax_pipeline.sh

# Single-model control (Explorer RL only, no novelty, 300 updates):
EXPDIS_PIPELINE_STOP_AFTER_EXPLORER=1 LAMBDA_NOVELTY=0 GRPO_MAX_STEPS=300 \
bash expdis_jax/scripts/launch_jax_pipeline.sh
```

Replace the angle-bracket placeholders before running. Use only workers assigned
to serving and list their matching server URLs. The watcher needs `gcloud`/SSH
access to both slices; without it the default external reload mode waits for an
acknowledgment. The launcher accepts an activated virtualenv or Conda environment,
or an explicit `ENV_DIR`; install the TPU runtime in that environment first.
It does not select a personal HF destination or delete older runs by default.
For resumable within-stage checkpoints, set `EXPDIS_SAVE_OPT_STATE=1` and allow
space for Adam state; the default checkpoints support weights-only stage handoffs.

Multi-Round and Multi-Explorer runs are driven by `pipeline.py::multi_round_main`
through the config flags `--pipeline-mode multi_round --num-rounds R
--explorers-per-round K --round-novelty-schedule l1,...,lR`. Per-batch sampler
synchronization is required. The native driver runs K Explorers sequentially.
For concurrent Explorers see [parallel execution](PARALLEL.md).

Multi-host stage handoffs require `--checkpoint-root` on shared storage: a
filesystem mounted at the same location on every trainer host, or a `gs://`
prefix accessible to all of them. Use a unique prefix for each run. All hosts
participate in Orbax writes and parameter collection; process 0 writes HF
exports and broadcasts the accepted SFT library. Only Main weights carry to the
next round. The multi-host launcher requires `CHECKPOINT_ROOT` and forwards it.
Two-process CPU tests cover the single-round and R=2/K=2 paths. Neither those
tests nor local simulated-device sharding establish TPU performance or equal
wall-clock cost; real TPU/vLLM integration remains to be validated.

Outputs land in `$RUNS_ROOT/$RUN_NAME` (default `$HOME/expdis/runs`):
`console.log`, per-stage checkpoints (`step_XXXXXX/`), trajectory JSONL,
`trajectory_library*.jsonl` and its `.funnel.json` audit, and `run_summary.json`
with the eval block. W&B project: `expdis`.

Multi-host launches: `scripts/launch_jax_pipeline_multihost_v5lite.sh` (one
slice; by default workers 0–3 train and 4–15 serve) and `scripts/launch_dapo_drgrpo_tpu.sh` (two
slices: one trains, one serves; requires `REMOTE_REPO` and the
`TRAIN_*`/`SERVE_*` TPU coordinates; Explorer RL only, default λ = 0.5 and 200
updates). `scripts/launch_vanilla_grpo*.sh` are a separate vanilla-GRPO
ablation (sequence-mean-token-mean loss, std-normalized advantages, 16k
completions), not the paper's single-model control. Retired one-off launchers
tied to past runs live in `scripts/archive/` (see its README).

## Key environment variables

Defaults in parentheses. Round-count and seed variables below are forwarded by
`launch_jax_pipeline.sh` and `launch_jax_pipeline_multihost_v5lite.sh`; when
invoking Python directly use `--num-rounds`, `--explorers-per-round`,
`--round-novelty-schedule` and `--seed`.

| Variable | Effect |
|---|---|
| `EXPDIS_VLLM_SERVER_URLS` | comma-separated vLLM URLs (required) |
| `LAMBDA_NOVELTY` (0.5) | novelty weight λ in the single-host pipeline and DAPO launchers; pass it explicitly to other launchers |
| `GRPO_MAX_STEPS` (200), `MAIN_RL_MAX_STEPS` (100) | Explorer RL and Main-model RL update budgets; legacy `ACTUAL_ONLINE_GRPO_MAX_STEPS` is accepted |
| `MAIN_RL_LR` (1e-6) | Main-model RL learning rate; legacy `ACTUAL_GRPO_LR` is accepted |
| `NUM_ROUNDS` (1), `EXPLORERS_PER_ROUND` (1), `ROUND_NOVELTY_SCHEDULE` | Multi-Round / Multi-Explorer geometry and λ schedule; legacy `SCOUTS_PER_ROUND` is accepted |
| `SEED` (0) | run seed in `launch_jax_pipeline.sh` |
| `EXPDIS_PIPELINE_STOP_AFTER_EXPLORER` (0) | stop after the Explorer RL stage |
| `EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS` (1) | sampler synchronization cadence; 1 = before every rollout batch (paper protocol) |
| `SAVE_EVERY_STEPS` (50) | checkpoint/export cadence (independent of the synchronization cadence) |
| `EXPDIS_TRAIN_VLLM_RELOAD_MODE` (external) | `external` (control-plane watcher) or `direct` (ssh into the serving slice) |
| `EXPDIS_DYNAMIC_SAMPLING_ELIGIBILITY` (blended_reward) | group eligibility on total-reward variance; `novelty_contribution` counts only λ·novelty variance |
| `EXPDIS_REQUIRE_WANDB` (1) | fail fast without W&B credentials |
| `EXPDIS_HF_CHECKPOINT_REPO`, `EXPDIS_HF_MIRROR_*` | HF mirroring of checkpoints, trajectories, reload bundles, and metrics |
| `EXPDIS_ALLOW_GCS_ARTIFACTS` (0), `EXPDIS_GCS_CHECKPOINT_BASE` | optional GCS artifact path |

## Behavior worth knowing when reproducing

- Training and evaluation use `prompting.math_messages`: the question followed
  by Qwen's math instruction in one user message, with no system message.
  This is release protocol `qwen3_math_user_suffix_v1` (2026-09-17); historical
  results retain their original prompt provenance.
- Each update exports the trainer's weights (`vllm_reload_hf/step_XXXXXX`) and
  reloads the serving copy before the next rollout batch; old
  log-probabilities are evaluated inline as `stop_gradient` of the current
  policy, which equals the frozen snapshot's log-probabilities because there is
  one optimizer step per batch, so the ratio starts at 1 and the 0.2/0.28 clip
  normally does not bind.
- The reward verifier scores the complete response and uses its last boxed
  answer when present. Commas and dollar signs are removed before
  comparison; numbers compare within 1e-6 in exact rational arithmetic.
- Soft-overlong shaping uses the sampled completion's token count before reward
  and advantage computation. Budget-clipped completions receive the full
  penalty and are masked from policy loss while remaining in the selected
  batch's advantages and fixed denominator. Offline/legacy fixtures without
  action IDs are tokenized once; production paper calls require server IDs.
- Dynamic sampling admits a group when its total per-trajectory reward has
  variance, in both stages; a group whose members differ only in the
  soft-overlong penalty is therefore admitted, while all-correct or
  all-incorrect groups with identical rewards are resampled. After 8 attempts
  the batch is filled with the remaining (zero-advantage) groups.
- RND: novelty is `sqrt(mse + 1e-8)` between the frozen target and the
  predictor, computed on completion tokens only (mean-pooled) at the
  quarter-depth layers `[7, 14, 21]` of Qwen3-1.7B. The default `auto` resolves
  `[9, 18, 27]` for Qwen3-4B and `[6, 13, 19]` for Ministral-3-3B.
  Width is 512 and bonuses are raw (no z-scoring). Candidate attempts use the
  same predictor; it is updated once after the policy step on selected rows'
  retained pre-update features. The predictor's Adam state
  (LR 1e-4) persists across updates within an Explorer and is re-initialized,
  together with the predictor, for every new Explorer and every round
  (`init_rnd_map`; the pipeline sets `EXPDIS_MULTIROUND_RESET_RND=1`).

## Evaluation and offline tests

Standalone P1 evaluation uses the training verifier and reports avg@n, pass@k
for k ≤ n (through 64), answer distinctness, normalized-answer entropy in nats,
and whitespace-token InterDistinct-4.
Errors invalidate the pool by default. No model-card baseline is subtracted
from a locally measured result.

```bash
python -m expdis_jax.eval --model-name /path/to/model \
  --server-urls http://localhost:8000/v1 --which AIME_2024 \
  --output-path /path/to/aime24.json
```

AIME24 and AIME25 have built-in loaders. For AIME26, MATH500, AMC23,
Minerva-Math, or GSM8K, add `--dataset-jsonl /path/to/benchmark.jsonl`; each row
must have `problem` and `answer` (or `question` and `ground_truth`). The CLI
records the file digest. Defaults are n=64 for AIME; n=32 for MATH500/AMC23;
n=8 for GSM8K; n=64 for Minerva except Qwen3-4B (n=32). Local checkpoints are
identified from `config.json`; unknown model aliases require an explicit
`--num-rollouts` for Minerva instead of silently choosing the wrong count.
Evaluate Base and every trained arm with the same server build, dataset
snapshot and protocol. Published gains require those measured pools.

Aggregate one checkpoint's five primary benchmark files (AMC23/GSM8K, if
supplied, are reported separately and excluded from mean accuracy):

```bash
python -m expdis_jax.eval --summarize-results \
  aime24.json aime25.json aime26.json math500.json minerva.json \
  --output-path mean_accuracy.json
```

The aggregator rejects duplicate, failed, partial, or inconsistent-protocol
files. Supply a separate set per checkpoint and seed; a serving alias alone
cannot prove checkpoint identity.

The repository covers the RND pipeline, breadth/depth/λ sweeps, dataset inputs,
and the accuracy and lexical/answer metrics above. The separate appendix
harnesses for sampled-token entropy, DARLING semantic classification, and
kNN/elliptical novelty are not included. Answer entropy is not token entropy;
these CPU checks do not reproduce those appendix studies.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.lock
.venv/bin/pytest tests/ -q
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  .venv/bin/pytest tests/test_expdis_jax_pipeline_cpu.py -q
# Optional reference runtime: torch + transformers supporting both models.
EXPDIS_HF_REFERENCE_PYTHON=/path/to/reference/python \
  .venv/bin/pytest tests/test_expdis_jax_model_reference.py -q
```

The supplied manuscript has a dynamic-sampling inconsistency: Algorithm 1
admits total-reward variance, but Appendix B also says homogeneous-correctness
groups cannot be admitted at λ=0. Soft-overlong shaping can make those groups'
rewards differ. This implementation follows Algorithm 1; see the audit report
before describing a run as satisfying both statements.

## Modules

- `config.py` — `TrainConfig` (paper defaults), CLI parsing, `validate_contract`
- `pipeline.py` — `main` (one Explorer RL → filter → SFT → Main-model RL → eval pass) and `multi_round_main`
- `train.py` — RL training loop for both stages: sampler synchronization, rollouts, scoring, dynamic sampling, the shared update, checkpoints
- `grpo.py` — the shared update (DAPO with Dr.GRPO normalization: mean-centered advantages, fixed 32,768 denominator, clip 0.2/0.28)
- `rewarding.py` — verifier (boxed-answer extraction, normalization, exact rational comparison) and reward assembly
- `novelty.py` — multilayer RND (one frozen target and one trained predictor per layer)
- `filtering.py` — QualityPool filter
- `distill.py` — Main-model SFT
- `lineage.py` — budget partition across Explorers and rounds, λ schedule
- `data.py` — DAPO / DeepScaleR / JSONL loaders, Qwen3 chat-template rendering, deterministic round shards
- `generate.py` — vLLM HTTP client
- `eval.py` — P1 evaluation with built-in AIME24/AIME25 and JSONL benchmark loaders
- `model.py` — Qwen3-family decoder in Flax (RMSNorm, RoPE/YaRN, GQA, SwiGLU; Pallas flash attention on TPU)
- `weights.py` — HF safetensors ↔ Flax params (transposes, tied embeddings; refuses quantized checkpoints)
- `mesh.py` — 1-D FSDP mesh and sharding helpers
- `lr_schedules.py` — per-stage learning-rate schedules
- `scripts/` — TPU setup, launchers, vLLM bootstrap/reload utilities; `scripts/archive/` holds retired one-off launchers

## Padding and optional uploads

Novelty microbatches and SFT batches use a bounded set of sequence lengths,
removing trailing padding while preserving active tokens. The attention adapter
adds any padding required by Pallas' 128-token tiles. Completion limits, pooled
features, SFT supervision and the fixed RL denominator are unchanged.

`artifacts.py` handles optional HF mirroring for both the trainer and pipeline.
Uploads run in a separate process without importing JAX. The old text-hash
novelty fallback has been removed; RND uses live hidden-state features.

## Tests

`tests/test_expdis_jax_*.py` cover the kernel invariants, the RND lifecycle, the
QualityPool filter, the lineage budget arithmetic, checkpoint handoff, and the
LR schedules. Padding tests compare features, SFT losses and gradients; they also
interpret the real Pallas forward/backward on CPU. They run on CPU:

```bash
pip install -r requirements-dev.txt
pytest tests/test_expdis_jax_filtering.py tests/test_expdis_jax_lineage.py \
       tests/test_expdis_jax_lr_schedule.py tests/test_expdis_jax_invariants.py \
       tests/test_expdis_jax_novelty.py tests/test_expdis_jax_checkpoint_handoff.py \
       tests/test_expdis_jax_padding.py tests/test_expdis_jax_pipeline_cpu.py
```
