# Scripts Directory

<!-- TMX_DOCTOR_ALLOW_FORBIDDEN_NETWORK_TOKENS: this README names forbidden
networking flags only to document that they must not be used for TRC runs. -->

This directory is intentionally small. New paper runs should go through the
config-driven production entrypoints, not copied one-off launch scripts.

The primary entrypoint is:

```bash
scripts/tmx_submit.py
```

Use it for new runs:

```bash
scripts/tmx_submit.py validate --spec configs/production_trc2_v5lite256_explorer_novelty075.json
scripts/tmx_submit.py plan --spec configs/production_trc2_v5lite256_explorer_novelty075.json --backend raw-tpu-tmx-jax
scripts/tmx_submit.py submit --spec configs/production_trc2_v5lite256_explorer_novelty075.json --backend raw-tpu-tmx-jax --dry-run
```

For the standard paper stages, use the wrapper:

```bash
scripts/tmx_paper.py validate all
scripts/tmx_paper.py preflight explorer-novelty
scripts/tmx_paper.py plan explorer-novelty --backend raw-tpu-tmx-jax
scripts/tmx_paper.py launch explorer-novelty --backend raw-tpu-tmx-jax
scripts/tmx_paper.py status explorer-novelty
```

The XPK/MaxText/Tunix backend is scaffolded under the same entrypoints, but it
requires a built container image and MaxText checkpoint path before launch. See
`docs/xpk_maxtext_tunix_backend.md`.

Supported production scripts:

- `tmx_submit.py`: generic config validation/planning/submission.
- `tmx_paper.py`: named paper-stage wrapper with `preflight`, `status`, and `launch`.
- `build_maxtext_tunix_image.sh`: optional XPK/MaxText/Tunix smoke-image builder.
- `trc2_v5lite256_guard.sh`: TRC external-IP/no-NAT safety check.
- `launch_trc2_v5lite256_tmx_jax_resume.sh`: v5litepod-256 raw-TPU fallback launcher.
- `restart_dapo_drgrpo_twoslice.sh`: JAX/vLLM restart wrapper used by the raw-TPU fallback.
- `reload_vllm_with_model.sh` and `reload_vllm_slice_with_model.sh`: HF-first vLLM reload helpers.
- `bootstrap_vllm_tpu_env.sh`: vLLM TPU environment helper.
- `tpu_target_lib.sh`: shared TPU targeting/safety helpers.

Older `launch_*`, `watch_*`, `recover_*`, plotting, migration, and local
experiment scripts are intentionally not part of the production branch. Restore
from git history or local scratch only when debugging an old run.

TRC safety rules are enforced by:

```text
scripts/trc2_v5lite256_guard.sh
tmx/orchestration/safety.py
```

For the TRC project, never add `--internal-ips`, Cloud NAT, router setup, or TPU
creation logic. The TPU must be created by tpunanny with external IPs enabled.
