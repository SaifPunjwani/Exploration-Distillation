# TMX Production Pipeline Refactor

This repo now has a production orchestration boundary in `tmx/orchestration`.
The intent is to stop encoding TPU topology, research rewards, storage policy,
and emergency recovery in the same shell scripts.

The enforced DAPO/Dr.GRPO/novelty contract is documented in
`docs/dapo_drgrpo_contract.md`.

## Target Architecture

The preferred long-term backend is:

```text
XPK/GKE -> MaxText distributed TPU training -> Tunix RL layer
        -> official rollout engine (vLLM-on-TPU or SGLang-JAX where supported)
        -> TMX reward/novelty plugins
        -> Hugging Face artifacts + W&B live metrics
```

The raw TPU VM path remains a fallback/debug backend:

```text
existing TRC v5litepod-256 -> tmx-jax launcher -> vLLM-on-TPU endpoints
```

Why this direction:

- XPK separates provisioned clusters from queued workloads, which is the right
  fit for repeated paper experiments on a fixed TPU allocation:
  <https://github.com/AI-Hypercomputer/xpk>
- MaxText is the preferred TPU training engine because it supports scalable
  JAX LLM training and post-training including SFT and GRPO:
  <https://github.com/AI-Hypercomputer/maxtext>
- Tunix is the preferred RL layer because it is JAX-based, supports SFT/RL,
  includes GRPO/DAPO/Dr.GRPO in its stated RL scope, and has native vLLM and
  SGLang-JAX rollout integrations:
  <https://github.com/google/tunix>

## Hard Constraints

- TRC project: your TPU Research Cloud project id (`network.project` or
  `TPU_PROJECT`; also export it as `TMX_TRC_PROJECT` so the TRC guards apply)
- TPU shape: `v5litepod-256`
- TPU target: `network.tpu_name` or `TPU_NAME` (the paper runs used
  `trc2-v5lite256-uc1a-r1`)
- External IPs required.
- Do not use `--internal-ips`.
- Do not create NAT gateways or routers.
- Hugging Face is canonical storage.
- Local disk is staging/cache only.
- Only process 0 should upload shared artifacts or initialize global W&B logs.

## Commands

List the supported presets:

```bash
python3 scripts/tmx_submit.py list-presets
```

For paper-stage workflows, prefer the short wrapper:

```bash
scripts/tmx_paper.py validate all
scripts/tmx_paper.py preflight explorer-novelty
scripts/tmx_paper.py plan explorer-novelty --backend raw-tpu-tmx-jax
scripts/tmx_paper.py launch explorer-novelty --backend raw-tpu-tmx-jax
scripts/tmx_paper.py status explorer-novelty
```

Available paper stages:

```text
explorer-correctness
explorer-novelty
actual-sft
actual-grpo
eval-aime24
```

Ready 64-chip targets can be used by creating a target-specific spec with
`init-spec`. Presets currently include:

```text
v6e-64-ew4a        -> v6e-64-16train-48rollout
v6e-64-use1d       -> v6e-64-16train-48rollout
v5litepod-64-ew4b  -> v5litepod-64-16train-48rollout
v5litepod-64-uc1a  -> v5litepod-64-16train-48rollout
```

These presets carry only the zone and accelerator type. The GCP project and
TPU name come from `TPU_PROJECT` / `TPU_NAME` in the environment (or
`--tpu-name`) when the spec is created, so no allocation is baked into the
repository.

Validate the production novelty spec:

```bash
python3 scripts/tmx_submit.py validate \
  --spec configs/production_trc2_v5lite256_explorer_novelty075.json
```

Show the preferred XPK/GKE/MaxText/Tunix submission plan:

```bash
TMX_XPK_CONTAINER_IMAGE=europe-west4-docker.pkg.dev/<gcp-project>/tmx/maxtext-tunix:smoke \
TMX_MAXTEXT_MODEL_NAME=qwen3-1.7b \
TMX_MAXTEXT_LOAD_PARAMETERS_PATH=hf://SaifPunjwani/two-model-exploration-models/base/qwen3-1.7b \
TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1 \
python3 scripts/tmx_submit.py plan \
  --spec configs/production_trc2_v5lite256_explorer_novelty075.json \
  --backend xpk-gke-maxtext-tunix
```

The XPK backend is documented in `docs/xpk_maxtext_tunix_backend.md`. It emits
a real `xpk workload create-pathways` command, but do not use it for the full
paper run until the container image has passed the documented smoke test.

Run through the current raw TPU fallback:

```bash
python3 scripts/tmx_submit.py submit \
  --spec configs/production_trc2_v5lite256_explorer_novelty075.json \
  --backend raw-tpu-tmx-jax \
  --execute \
  --wait-ready
```

Use `--dry-run` instead of `--execute` only when inspecting the emitted plan.

Actual stages are HF-trajectory driven on the raw TPU fallback:

```bash
python3 scripts/tmx_submit.py submit \
  --spec configs/production_trc2_v5lite256_actual_grpo.json \
  --backend raw-tpu-tmx-jax \
  --execute \
  --wait-ready
```

For `actual-sft` and `actual-grpo`, set `resume.existing_trajectories` to an
`hf://.../explorer/trajectories/*.jsonl` file, or set `resume.source_run` so the
launcher derives the latest Explorer trajectory path. The remote job sets
`TMX_PIPELINE_SKIP_EXPLORER=1`; it must never rerun Explorer by accident.
`actual-sft` also sets `TMX_PIPELINE_STOP_AFTER_SFT=1`, while `actual-grpo`
runs SFT and then correctness-only Actual GRPO from that SFT checkpoint.

```bash
python3 scripts/tmx_submit.py init-spec \
  --name dapo16k_drgrpo_actual_grpo_from_lambda05 \
  --stage actual_grpo_16k \
  --existing-trajectories hf://SaifPunjwani/two-model-exploration-checkpoints/<run>/explorer/trajectories/explorer_trajectories_latest.jsonl \
  --output configs/production_actual_grpo_from_lambda05.json
```

The paper pipeline specs are:

```text
configs/production_trc2_v5lite256_explorer_correctness.json
configs/production_trc2_v5lite256_explorer_novelty075.json
configs/production_trc2_v5lite256_actual_sft.json
configs/production_trc2_v5lite256_actual_grpo.json
configs/production_trc2_v5lite256_eval_aime24.json
```

To create a new lambda run without editing shell scripts:

```bash
python3 scripts/tmx_submit.py init-spec \
  --name dapo16k_drgrpo_novelty05_trc256_prod \
  --stage explorer_novelty05_16k \
  --topology v5litepod-256-64train-192rollout \
  --network trc2-v5litepod-256-uc1a \
  --output configs/production_trc2_v5lite256_explorer_novelty05.json
```

To resume explicitly from a Hugging Face run checkpoint, make that part of the
spec instead of relying on launcher defaults:

```bash
python3 scripts/tmx_submit.py init-spec \
  --name dapo16k_drgrpo_novelty075_resume60_trc256_prod \
  --stage explorer_novelty075_16k \
  --topology v5litepod-256-64train-192rollout \
  --network trc2-v5litepod-256-uc1a \
  --resume-source-run dapo16k_drgrpo_novelty075_trc256_prod \
  --resume-source-step 000060 \
  --output configs/production_trc2_v5lite256_explorer_novelty075_resume60.json
```

Safety scan before pushing or launching:

```bash
python3 scripts/tmx_submit.py doctor configs tmx/orchestration scripts/trc2_v5lite256_guard.sh
```

## Research Logic Boundary

Custom TMX code should own:

- correctness reward
- novelty/RND reward
- Explorer trajectory collection
- Actual SFT/GRPO stages
- eval metrics and plots

Infrastructure code should not know about reward internals beyond passing a
versioned spec into the training container/job.

## Hugging Face Layout

Use two repos:

- model repo: checkpoints, final model exports, eval plots
- dataset repo: rollout shards, trajectory JSONL, configs, metrics

Recommended paths:

```text
<run>/configs/experiment_spec.json
<run>/explorer/checkpoints/step_000025/
<run>/explorer/trajectories/step_000025/shard_00000.jsonl
<run>/evals/aime24_avg32.json
<run>/plots/reward_curve.png
```

Do not treat Hugging Face as a high-frequency distributed filesystem. Upload
milestones and uniquely named shards from process 0, or use per-rank unique
paths if parallel upload becomes necessary.

Production milestone uploads are required (`TMX_REQUIRE_HF_ARTIFACT_UPLOAD=1`):
if a checkpoint cannot be uploaded to Hugging Face, the run should stop visibly
rather than silently losing the only durable artifact copy.
