# Exploration-Distillation (ExpDis) — GPU stack

This is the `gpu` branch: the PyTorch + vLLM implementation of
Exploration-Distillation used for the GPU reproduction and measurement
campaign. The main implementation (JAX / Flax on TPU, `expdis_jax/`) is on the
`main` branch; its README describes the method and the paper's protocol
constants in full.

Exploration-Distillation separates *exploration* from *deployment*: a
disposable **Explorer** is trained with **Explorer RL** on correctness plus an
RND novelty bonus credited only to verifier-correct completions, its
trajectories are reduced by the **QualityPool filter** to a small clean transfer
set, and the deployed **Main model** is fine-tuned on that set and then trained
with **Main-model RL** on correctness plus soft-overlong shaping, with no
novelty term. **Multi-Explorer** merges the pools of K Explorers; **Multi-Round**
repeats Explorer → filter → Main for R rounds under one fixed update budget.

**The GPU campaign update differs from the supplied paper.** Both GPU RL stages
share group-mean advantages, a fixed loss denominator, and soft-overlong shaping;
the Explorer adds correctness-gated RND. The default objective is REINFORCE,
whose first-step gradient equals the clipped objective at ratio one. Actual
asymmetric clipping requires `--ppo-mode`, which the c8 campaign validator
rejects. Standalone defaults use 16,384-token completions and denominator,
reload every 10 updates, and no masking of clipped completions. The c8 pipeline
does synchronize after every update, but also has its own filtering, dynamic
sampling and predictor lifecycle. These campaign profiles are not an exact
implementation of the latest 32,768-token JAX paper contract.

See [GPU_AUDIT.md](GPU_AUDIT.md) for the exact deviations, regression fix and
test evidence. The campaign's c8/c9/c10 identities remain distinct; do not label
their results as reproducing the latest paper without aligning and rerunning
the protocol. CPU tests do not reproduce benchmark gains or GPU serving.

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


## What `tmx_gpu/` implements

| Stage | Where |
|---|---|
| Explorer RL: one training GPU plus a restart-reload vLLM pool on the remaining GPUs; the shared update (group-mean advantages, fixed loss denominator), DAPO dynamic sampling, soft-overlong shaping | `grpo_gpu.py`, `vllm_pool.py` |
| Multilayer RND novelty (one frozen target and one trained predictor per layer), credited on correct completions | `rnd_torch.py`, `grpo_gpu.py` |
| QualityPool filter over the pooled trajectories of K Explorers (correct, naturally terminated, boxed answer, one per problem, shortest first, cap 500) | `filter_pool.py` |
| Main-model SFT on the filtered set; behaviour gate and correctness-blind checkpoint selection for SFT checkpoints | `sft_gpu.py`, `sft_gate_gpu.py`, `select_sft_checkpoint.py` |
| Main-model RL (correctness plus soft-overlong shaping, no novelty) | `grpo_gpu.py` |
| One-round and Multi-Round / Multi-Explorer driver with fail-closed run-identity contracts | `pipeline_gpu.py`, `mrme_contract.py` |
| Benchmark evaluation (AIME24/25, MATH500, AMC23, Minerva-Math, GSM8K) under the P1/P2 protocols and provenance-bound result matrices | `eval_gpu.py`, `eval_matrix.py` |
| Evidence and receipt contracts for the campaign | `c9_evidence.py`, `c10_evidence.py`, `dapo_evidence.py`, `orchestrator_receipt.py`, `gold_syntax_audit.py` |
| Shared W&B / artifact conventions | `common.py` |

Also on this branch: `tmx/` (the original torch_xla pilot and the
config-driven TPU orchestration layer `tmx/orchestration/`), `tmx_jax/` (the
TPU/JAX copy that this branch's TPU launchers drive), `configs/`
(`production_*.json` run specs), `infra/xpk/` and `docker/` (TPU/XPK
bring-up), `patches/`, and `baseline_runs_gpu/` (protocol and contract
documents).

## Naming in the code

The `tmx`, `tmx_gpu`, and `tmx_jax` packages and the `TMX_*` environment
variables keep their historical names on this branch. Three legacy role names
remain because checkpoints, artifact paths, W&B runs, and configs use them:
`Scout` is the **Explorer**, and `Central` / `Actual` are the **Main model**.

The pipeline CLI accepts `--explorer-steps`, `--main-steps`,
`--explorers-per-round`, `--explorer-lr`, `--main-lr`, `--explorer-only`, and
`--skip-explorer`. Legacy spellings remain aliases so existing campaign commands,
checkpoints and receipts still load.

| Paper | Code |
|---|---|
| Explorer | `Scout`, `scout_*`, `explorer_*` (e.g. `--scout-steps`, `explorer_trajectories.jsonl`) |
| Main model | `Central`, `Actual`, `central_*`, `actual_*` |
| Base | `base` |
| Explorer RL / Main-model RL (the shared update) | `grpo_gpu.py`, the `grpo_*`/`--grpo-*` options |
| novelty weight λ | `lambda_novelty`, `LAMBDA_NOVELTY`, `--lambda-schedule` |
| ExpDis pipeline | `expdis_job.sh`, `pipeline_gpu.py` |

## Try the pipeline on CPU

From the repository root, with Python 3.13 installed:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.lock
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest tmx_gpu/tests/test_e2e_dry_run.py -o addopts='' -q
```

Expected result: **2 passed**. These tests build a tiny Llama and tokenizer
locally, run two training rounds, and check crash/resume with a stub vLLM server.
After package installation, they need no credentials, model downloads or GPU.
The test lock records the CPU audit environment. Run the full suite with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest tests tmx_gpu/tests -o addopts='' -q
```

Full GPU training uses a separate CUDA/vLLM runtime. The Lepton job entrypoints
pin the `vllm/vllm-openai` image used by the historical campaign. Configure their
required model, data, storage and device settings as described below. The older
`requirements.txt` is retained for that stack; its complete installation was not
validated by the CPU audit. A real accelerator/serving test is still required.

## Entry points

GPU (one node, env-driven Lepton job entrypoints; every required variable is
named in the script header):

- `tmx_gpu/launch/expdis_job.sh` — Explorer RL → pool → QualityPool → Main-model
  SFT → Main-model RL (one round or Multi-Round / Multi-Explorer).
- `tmx_gpu/launch/dapo_job.sh` — the single-model DAPO control (correctness
  plus soft-overlong shaping, no novelty, from Base).
- `tmx_gpu/launch/base_eval_job.sh` — benchmark evaluation of Base and trained
  models under `baseline_runs_gpu/EVAL_PROTOCOL.md`.
- `tmx_gpu/launch/sft_gate_job.sh` — the benchmark-independent SFT behaviour
  gate.
- The stages are also plain CLIs: `python -m tmx_gpu.pipeline_gpu --help`,
  `python -m tmx_gpu.grpo_gpu --help`, `python -m tmx_gpu.sft_gpu --help`,
  `python -m tmx_gpu.filter_pool --help`, `python -m tmx_gpu.eval_gpu --help`.

TPU (config-driven submission of `configs/production_*.json`):

- `scripts/tmx_paper.py` (`validate`, `preflight`, `plan`, `launch`,
  `status`) and `scripts/tmx_submit.py` (`list-presets`, `init-spec`,
  `validate`, `plan`, `submit`, `doctor`); see `PRODUCTION.md`,
  `docs/production_pipeline.md`, `configs/README.md`, and
  `infra/xpk/README.md`.

## Configuration

- W&B: `WANDB_API_KEY`; `WANDB_ENTITY` is optional (unset means your default
  entity). GPU-era runs log to the project `two-model-explore-gpu`.
- TPU specs ship without a GCP project id or TPU name. Set `TPU_PROJECT`,
  `TPU_ZONE`, and `TPU_NAME` (or edit the spec's `network` block) before
  `launch`; set `TMX_TRC_PROJECT` to your TPU Research Cloud project id to
  enable the TRC networking guards (`scripts/trc2_v5lite256_guard.sh`,
  `tmx/orchestration/safety.py`).
- The legacy campaign launchers under `scripts/` fail fast with a message that
  names any host alias, remote path, or project variable they need.

## Documents

- `baseline_runs_gpu/EVAL_PROTOCOL.md` — the P1 / P2 evaluation protocols.
- `baseline_runs_gpu/C7_PROVENANCE.md`, `C8_REPAIR_CONTRACT.md`, `C9_*.md`,
  `C10_*.md` — preregistered contracts of the GPU campaign.
- `docs/dapo_drgrpo_contract.md` — the enforced DAPO / Dr.GRPO / novelty
  contract.
- `docs/production_pipeline.md`, `docs/xpk_maxtext_tunix_backend.md`,
  `infra/xpk/README.md`, `scripts/README.md`, `configs/README.md`.

Results, experiment logs, and paper artifacts are not part of the public
release. Pre-rename external state keeps its historical names on purpose:
GCS paths under `gs://two-model-exploration-checkpoints/` and
`gs://llmvae-checkpoints/two-model-exploration/`, the HF repos
`SaifPunjwani/two-model-exploration-checkpoints` and
`SaifPunjwani/two-model-exploration-models`, and the W&B projects
`two-model-explore-gpu` and `two-model-explore`.

Historical `jrl` strings in campaign provenance, evidence artifact identifiers,
and tests referring to those artifacts are external names retained for lookup.
They do not name the current method.
