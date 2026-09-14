# XPK / MaxText / Tunix Backend

This is the target production backend for large TRC runs. It is intentionally
separate from the raw TPU VM fallback.

## Source-Backed Shape

Official MaxText multi-host RL documentation submits GRPO through:

```text
xpk workload create-pathways ... --command "python3 -m maxtext.trainers.post_train.rl.train_rl ..."
```

It also states that the RL algorithms are implemented through Tunix and that
vLLM is used for response generation:
<https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/rl_on_multi_host.html>

Official MaxText Docker documentation says `maxtext[runner]` installs the
runner dependencies and XPK support needed to build/upload MaxText Docker
images:
<https://maxtext.readthedocs.io/en/latest/build_maxtext.html>

Tunix documents GRPO, DAPO, Dr.GRPO, MaxText integration, vLLM rollout, and
SGLang-JAX rollout as first-class pieces of its JAX post-training stack:
<https://tunix.readthedocs.io/en/latest/>

## Current Repo Status

Implemented now:

- Config/schema support for `xpk-gke-maxtext-tunix`.
- XPK `create-pathways` command generation.
- Full DAPO/Dr.GRPO/novelty/HF-only environment propagation.
- TRC network guard: no internal IP flags, no NAT/router creation.
- Container entrypoint module: `python -m tmx.orchestration.workload run`.
- Reward plugin module: `tmx.orchestration.reward_plugins`.

Still required before making this the default executable path:

- Build a Docker image containing this repo, MaxText, Tunix, vLLM/SGLang-JAX
  rollout dependencies, and the Qwen/tokenizer dependencies.
- Convert/download the base or resume checkpoint into a MaxText-compatible
  `load_parameters_path`.
- Smoke-test `tmx.orchestration.workload run` with
  `TMX_DRY_RUN_WORKLOAD=1`.
- Smoke-test one small GRPO step on the target TPU before launching the full
  256-chip paper run.

Until those pass, use `raw-tpu-tmx-jax` for the real run.

## Safe XPK Plan

Build/push the scaffold image:

```bash
export TMX_XPK_CONTAINER_IMAGE=europe-west4-docker.pkg.dev/<gcp-project>/tmx/maxtext-tunix:smoke
scripts/build_maxtext_tunix_image.sh
```

The image scaffold lives at `docker/maxtext-tunix/Dockerfile`. It is intended
for smoke testing the adapter path; performance tuning should happen only after
the one-step GRPO smoke passes.

Dry-run the XPK command:

```bash
TMX_XPK_CONTAINER_IMAGE=europe-west4-docker.pkg.dev/<gcp-project>/tmx/maxtext-tunix:smoke \
TMX_MAXTEXT_MODEL_NAME=qwen3-1.7b \
TMX_MAXTEXT_LOAD_PARAMETERS_PATH=hf://SaifPunjwani/two-model-exploration-models/base/qwen3-1.7b \
TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1 \
scripts/tmx_paper.py plan explorer-novelty --backend xpk-gke-maxtext-tunix
```

The generated command must contain:

```text
xpk workload create-pathways
--tpu-type v5litepod-256
--num-slices 1
TMX_ALLOW_GCS_ARTIFACTS=0
TMX_REWARD_PLUGIN=tmx.orchestration.reward_plugins
```

It must not contain:

```text
--internal-ip
--internal-ips
cloud nat
compute routers
routers nats
```

## Container Entrypoint Contract

The production image should execute:

```bash
python -m tmx.orchestration.workload run
```

The entrypoint reads `TMX_EXPERIMENT_SPEC_JSON_B64`, validates the experiment
contract, verifies MaxText and Tunix are importable, then dispatches:

```bash
python -m maxtext.trainers.post_train.rl.train_rl \
  model_name=<TMX_MAXTEXT_MODEL_NAME> \
  load_parameters_path=<TMX_MAXTEXT_LOAD_PARAMETERS_PATH> \
  run_name=<RUN_NAME> \
  base_output_directory=<TMX_MAXTEXT_BASE_OUTPUT_DIRECTORY> \
  hf_access_token=<HF_TOKEN>
```

The reward plugin path is passed through `TMX_REWARD_PLUGIN`. Any final Tunix
adapter should import that plugin rather than duplicating correctness/novelty
logic inside infrastructure code.
