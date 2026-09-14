# Tunix / MaxText / vLLM-on-TPU path

Encapsulated launcher for the DAPO / Dr.GRPO / novelty paper run on the TRC2
v5litepod-256 (or any other TPU VM with miniforge already set up).

Correctness-only baselines can dispatch through `tunix.cli.grpo_main`.
Novelty runs dispatch through `tmx.orchestration.tunix_main`, a
programmatic Tunix/MaxText adapter that swaps in Tunix `DAPOConfig` with the
Dr.GRPO advantage estimator and injects TMX novelty into reward kwargs inside
the learner. MaxText is used as the model backend via
`model_config.model_source=maxtext`. vLLM-on-TPU is the rollout engine,
in-process under Tunix, so rollout decoding uses the production KV-cache path
without a separate HTTP server or invalid libtpu sub-slice fight.

## One-command launch

The fastest happy-path for the λ=0.5 paper run on TRC2:

```bash
HF_TOKEN=hf_... WANDB_API_KEY=... \
bash infra/xpk/v5lite256_resume.sh
```

This:
1. Runs `preflight_local.sh` — py_compile, ruff, vulture, pytest, spec validate.
2. Bootstraps Tunix + MaxText into the worker's miniforge env (idempotent).
3. Walks gates 1 → 3 → 4 (rollout up to 16K) → 6 (resume from step_000100).
4. Streams the run log back to `runs_logs/paper_pipeline_<stamp>/`.

Gate 6 launches inside remote `tmux` session `tmx_tunix_resume100`, so the
training process survives laptop sleep or SSH disconnect.

## Gate-by-gate walkthrough (manual control)

```bash
export TPU_PROJECT=<trc-project>
export TPU_ZONE=europe-west4-b
export TPU_NAME=trc2-v5lite256-ew4b-r1
export TPU_WORKER=0
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
```

```bash
# Local checks (no TPU touched).
bash infra/xpk/preflight_local.sh

# Install Tunix + MaxText into the TPU worker's miniforge env.
# Idempotent — safe to rerun. Use WORKERS="0 1 2 3" to bootstrap multiple.
bash infra/xpk/bootstrap_tunix_env.sh

# Gate 1: imports + JAX devices visible.
bash infra/xpk/smoke_gate1_imports.sh

# Gate 2: tiny GRPO step, 2 prompts × 2 generations, 256 tokens, 2 steps.
bash infra/xpk/smoke_gate2_training.sh

# Gate 3: same as 2 but at the DAPO contract values; verifies contract logged.
bash infra/xpk/smoke_gate3_contract.sh

# Gate 4: rollout scale-up. TIERS=1 quick / TIERS="1 2 3" full 16K.
TIERS="1 2 3" bash infra/xpk/smoke_gate4_rollout.sh

# Gate 6: launch the actual paper-run resume from step_000100.
bash infra/xpk/smoke_gate6_resume.sh
```

## Why this design

**TRC2 + XPK is incompatible** — `xpk workload create` only works on
clusters XPK provisioned itself; `xpk cluster adapt` doesn't support TPU.
TRC's "no TPU creation" rule kills the XPK path on TRC2. We bypass XPK
and run Tunix directly on the TPU VM via `gcloud ssh`. For non-TRC TPUs
(non-TRC projects) you can use XPK via `cluster_create.sh` + `submit.sh`
in this same directory.

**MaxText direct RL is not enough for TMX rewards.** The plain MaxText RL
entrypoint does not own our correctness/novelty contract. The Tunix path
loads MaxText models but lets Tunix own GRPO/DAPO orchestration and reward
plugins. For λ>0 novelty runs, `tunix_workload` dispatches to
`tmx.orchestration.tunix_main` because the plain Tunix CLI cannot inject RND
novelty at the correct learner point.

**Novelty is learner-injected, not rollout-metadata magic.** Tunix's reward
manager receives kwargs from the learner, not arbitrary `RolloutOutput`
attributes. `tmx.orchestration.tunix_learner` captures the vLLM
`RolloutOutput` after `rl_cluster.generate(...)`, computes per-sample
novelty, and injects `novelty=[...]` before reward computation.

**Hidden-state RND is a hard gate.** The paper contract requires real actor
hidden-state novelty. The adapter deliberately fails if the MaxText/Tunix
actor does not expose the requested features; it does not silently fall back
to token-hash novelty.

**The single libtpu fabric on v5lite256 forbids sub-pod splits without
Pathways.** Tunix runs vLLM in-process (sharing the JAX libtpu init), so
both training and rollout coexist without a worker split fight.

## Configs

| Config | Use |
|---|---|
| `configs/tunix/tmx_smoke_tiny.json` | 1-host, 256/512 tokens, 2 steps. Sanity. |
| `configs/tunix/tmx_dapo_drgrpo_fresh.json` | Fresh λ=0.5 run from base Qwen3-1.7B. |
| `configs/tunix/tmx_dapo_drgrpo_resume_step100.json` | Resume from HF step_000100. |

Each config is round-trip tested by `tests/test_tunix_configs.py` and
adapter tests against:
- `spec.validate()` (codex's safety + DAPO contract validator)
- `emit_overrides()` produces clean Tunix CLI args for λ=0
- `emit_programmatic_overrides()` preserves DAPO + Dr.GRPO for λ>0
- No GCS leakage
- DAPO contract invariants (clip 0.2/0.28, KL=0, +1/-1 rewards)
- HF-only storage, no NAT, no internal IPs

## Tunable env

All scripts honor the same env contract:

| Var | Default | Notes |
|---|---|---|
| `TPU_PROJECT` | required | GCP project |
| `TPU_ZONE` | required | TPU zone |
| `TPU_NAME` | required | existing TPU VM |
| `TPU_WORKER` | 0 | which VM to target |
| `ENV_NAME` | tmx_jax | miniforge env name |
| `HOME_DIR` | (required) | TPU VM home |
| `SPEC` | resume_step100.json | path to ExperimentSpec |
| `GATES` | 1 3 4 6 | which gates to run |
| `TIERS` | 1 | gate-4 rollout tiers (1=2K, 2=8K, 3=16K) |
| `SKIP_BOOTSTRAP` | 0 | set 1 if Tunix already installed |
| `SKIP_PREFLIGHT` | 0 | set 1 to skip local checks |
| `HF_TOKEN` | required | HuggingFace upload |
| `WANDB_API_KEY` | optional | live metrics |
| `TMX_AGGRESSIVE_XLA` | 0 | 1 to enable latency-hiding scheduler XLA flags |

## What got built (everything new under one roof)

```
docker/maxtext-tunix/
├── Dockerfile.tpu          ← jax-stable-stack base, source-installs Tunix + MaxText
├── .dockerignore
└── build_and_push.sh       ← Artifact Registry image push

infra/xpk/
├── bootstrap_tunix_env.sh  ← install Tunix+MaxText into existing miniforge env
├── cluster_create.sh       ← XPK cluster create (non-TRC TPUs only)
├── preflight_local.sh      ← py_compile + ruff + vulture + pytest + validate
├── run_paper_pipeline.sh   ← single driver for gates 1→6
├── smoke_gate1_imports.sh
├── smoke_gate2_training.sh
├── smoke_gate3_contract.sh
├── smoke_gate4_rollout.sh
├── smoke_gate6_resume.sh
├── submit.sh               ← XPK workload create wrapper (non-TRC)
├── v5lite256_resume.sh     ← ONE-COMMAND TRC2 RESUME LAUNCHER
└── README.md

tmx/orchestration/
├── optimization.py         ← XLA env tuning + mesh shape calc
├── topologies_extra.py     ← v6e/v5p network/topology presets
├── tunix_config.py         ← ExperimentSpec → Tunix CLI/programmatic overrides
├── tunix_learner.py        ← DAPOConfig + Dr.GRPO + learner novelty injection
├── tunix_main.py           ← programmatic Tunix/MaxText novelty entrypoint
├── tunix_rewards.py        ← Tunix-shape reward fns (+1/-1 + novelty)
└── tunix_workload.py       ← container/host entrypoint dispatcher

configs/tunix/
├── tmx_smoke_tiny.json
├── tmx_dapo_drgrpo_fresh.json
└── tmx_dapo_drgrpo_resume_step100.json

tests/
├── test_optimization.py        (7 tests)
├── test_topologies_extra.py    (4 tests)
├── test_tunix_rewards.py       (7 tests)
├── test_tunix_config.py        (7 tests)
├── test_tunix_workload.py      (6 tests)
├── test_tunix_learner.py       (3 tests)
└── test_tunix_configs.py       (5×3=15 parameterized)

pyproject.toml                  ← ruff + vulture + pytest config
scripts/lint_orchestration.sh   ← one-command lint
```

## Replicating for a different TPU / different run

1. Copy `configs/tunix/tmx_dapo_drgrpo_resume_step100.json` to a new file.
2. Edit `network.tpu_name` / `zone` / `project`, `algorithm.lambda_novelty`,
   `resume.init_checkpoint`, `storage.artifact_prefix`, `name`.
3. `python3 scripts/tmx_submit.py validate --spec configs/tunix/<your>.json`
4. `SPEC=configs/tunix/<your>.json bash infra/xpk/v5lite256_resume.sh`

The validator will refuse `gs://`, NAT, internal-IP, TRC TPU creation, or
DAPO contract drift before the launch ever touches the TPU.
