# Production Entry Point

Use the config-driven runner for new paper runs:

```bash
scripts/tmx_paper.py validate all
scripts/tmx_paper.py preflight explorer-novelty
scripts/tmx_paper.py plan explorer-novelty --backend raw-tpu-tmx-jax
scripts/tmx_paper.py launch explorer-novelty --backend raw-tpu-tmx-jax
```

Continue the λ=0.5 Explorer run past its completed checkpoint:

```bash
scripts/tmx_paper.py validate explorer-novelty05-extend
scripts/tmx_paper.py plan explorer-novelty05-extend --backend raw-tpu-tmx-jax
scripts/tmx_paper.py launch explorer-novelty05-extend --backend raw-tpu-tmx-jax
```

Run Actual from a completed Explorer trajectory bundle:

```bash
scripts/tmx_paper.py validate actual-grpo
scripts/tmx_paper.py plan actual-grpo --backend raw-tpu-tmx-jax
scripts/tmx_paper.py launch actual-grpo --backend raw-tpu-tmx-jax
```

The raw TPU Actual path is intentionally trajectory-driven. It sets
`TMX_PIPELINE_SKIP_EXPLORER=1`, downloads `resume.existing_trajectories` from
Hugging Face, filters accepted rollouts, trains SFT, then runs Actual GRPO. For
SFT-only distillation, use `actual-sft`; it sets `TMX_PIPELINE_STOP_AFTER_SFT=1`
and will not enter Actual GRPO.

Create a target-specific run for any known TPU preset without editing shell
scripts:

```bash
scripts/tmx_submit.py list-presets
TPU_PROJECT=<gcp-project> TPU_NAME=<tpu-name> \
scripts/tmx_submit.py init-spec \
  --name dapo16k_drgrpo_novelty075_v6e64_use1d \
  --network v6e-64-use1d \
  --topology v6e-64-16train-48rollout \
  --stage explorer_novelty075_16k \
  --output configs/production_v6e64_use1d_explorer_novelty075.json
scripts/tmx_submit.py validate --spec configs/production_v6e64_use1d_explorer_novelty075.json
scripts/tmx_submit.py submit \
  --spec configs/production_v6e64_use1d_explorer_novelty075.json \
  --backend raw-tpu-tmx-jax \
  --execute \
  --wait-ready
```

Detailed architecture and safety rules are in `docs/production_pipeline.md`.
The preferred XPK/MaxText/Tunix target backend is documented in
`docs/xpk_maxtext_tunix_backend.md`, but the raw TPU backend remains the proven
launch path until the MaxText/Tunix container smoke test passes.

Build the XPK smoke image only when intentionally testing that backend:

```bash
export TMX_XPK_CONTAINER_IMAGE=europe-west4-docker.pkg.dev/<gcp-project>/tmx/maxtext-tunix:smoke
scripts/build_maxtext_tunix_image.sh
```

Hard TRC rule: use only nanny-created external-IP `v5litepod-256` resources.
Do not create NAT/routers and do not use internal-IP TPU access.

Check readiness without launching:

```bash
scripts/tmx_paper.py status explorer-novelty
```

`launch` waits for the TPU to become `READY/HEALTHY`, validates the full
algorithm/storage/network contract, then starts the remote TPU workers.
