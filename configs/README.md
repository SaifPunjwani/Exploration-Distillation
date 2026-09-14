# Configs

Production run specs are JSON files named `production_*.json`.

Current paper pipeline:

```text
production_trc2_v5lite256_explorer_correctness.json
production_trc2_v5lite256_explorer_novelty075.json
production_trc2_v5lite256_explorer_novelty05_extend.json
production_trc2_v5lite256_actual_sft.json
production_trc2_v5lite256_actual_grpo.json
production_trc2_v5lite256_eval_aime24.json
```

Create variants without editing shell scripts:

```bash
scripts/tmx_submit.py init-spec \
  --name dapo16k_drgrpo_novelty05_trc256_prod \
  --network trc2-v5litepod-256-uc1a \
  --topology v5litepod-256-64train-192rollout \
  --stage explorer_novelty05_16k \
  --output configs/production_trc2_v5lite256_explorer_novelty05.json
```

Create a resume variant explicitly:

```bash
scripts/tmx_submit.py init-spec \
  --name dapo16k_drgrpo_novelty075_resume60_trc256_prod \
  --stage explorer_novelty075_16k \
  --resume-source-run dapo16k_drgrpo_novelty075_trc256_prod \
  --resume-source-step 000060 \
  --output configs/production_trc2_v5lite256_explorer_novelty075_resume60.json
```

Validate all production specs:

```bash
for f in configs/production_*.json; do
  scripts/tmx_submit.py validate --spec "$f"
done
```

The paper-stage wrapper uses these specs directly:

```bash
scripts/tmx_paper.py validate all
scripts/tmx_paper.py plan explorer-novelty --backend raw-tpu-tmx-jax
```

Continue the λ=0.5 Explorer past its completed checkpoint:

```bash
scripts/tmx_paper.py plan explorer-novelty05-extend --backend raw-tpu-tmx-jax
scripts/tmx_paper.py launch explorer-novelty05-extend --backend raw-tpu-tmx-jax
```

Run Actual from existing HF Explorer trajectories:

```bash
scripts/tmx_paper.py plan actual-sft --backend raw-tpu-tmx-jax
scripts/tmx_paper.py plan actual-grpo --backend raw-tpu-tmx-jax
```

Actual specs must set either `resume.existing_trajectories` to an `hf://...jsonl`
artifact or `resume.source_run` so the launcher can derive
`<source_run>/explorer/trajectories/explorer_trajectories_latest.jsonl`. The
launcher clears `SOURCE_RUN` before remote execution so the raw TPU script does
not reinterpret the source as an Explorer checkpoint resume.

Create an Actual variant directly:

```bash
scripts/tmx_submit.py init-spec \
  --name dapo16k_drgrpo_actual_grpo_from_lambda05 \
  --stage actual_grpo_16k \
  --existing-trajectories hf://SaifPunjwani/two-model-exploration-checkpoints/<run>/explorer/trajectories/explorer_trajectories_latest.jsonl \
  --output configs/production_actual_grpo_from_lambda05.json
```

## Ready 64-chip TPU targets

Use these `--network` / `--topology` pairs to create a target-specific spec.
All use HF-only artifacts and the same DAPO/Dr.GRPO contract.

```text
v6e-64-ew4a        + v6e-64-16train-48rollout
v6e-64-use1d       + v6e-64-16train-48rollout
v5litepod-64-ew4b  + v5litepod-64-16train-48rollout
v5litepod-64-uc1a  + v5litepod-64-16train-48rollout
```

These presets carry only the zone and accelerator type. The GCP project and
TPU name come from `TPU_PROJECT` / `TPU_NAME` in the environment (or
`--tpu-name`) when the spec is created, so no allocation is baked into the
repository.

Example:

```bash
TPU_PROJECT=<gcp-project> TPU_NAME=<tpu-name> \
scripts/tmx_submit.py init-spec \
  --name dapo16k_drgrpo_novelty075_v6e64_use1d \
  --network v6e-64-use1d \
  --topology v6e-64-16train-48rollout \
  --stage explorer_novelty075_16k \
  --output configs/production_v6e64_use1d_explorer_novelty075.json

scripts/tmx_submit.py status \
  --spec configs/production_v6e64_use1d_explorer_novelty075.json

scripts/tmx_submit.py submit \
  --spec configs/production_v6e64_use1d_explorer_novelty075.json \
  --backend raw-tpu-tmx-jax \
  --execute \
  --wait-ready
```

The XPK/MaxText/Tunix backend needs a container image and MaxText checkpoint
path before it is executable:

```bash
TMX_XPK_CONTAINER_IMAGE=europe-west4-docker.pkg.dev/<gcp-project>/tmx/maxtext-tunix:smoke \
TMX_MAXTEXT_MODEL_NAME=qwen3-1.7b \
TMX_MAXTEXT_LOAD_PARAMETERS_PATH=hf://SaifPunjwani/two-model-exploration-models/base/qwen3-1.7b \
TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1 \
scripts/tmx_paper.py plan explorer-novelty --backend xpk-gke-maxtext-tunix
```
