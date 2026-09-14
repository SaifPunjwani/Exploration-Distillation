#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-qwen3_1p7b_one_round_$(date +%Y%m%d_%H%M%S)}"
NOVELTY_RUN_NAME="${NOVELTY_RUN_NAME:-qwen3_1p7b_explorer_novelty_$(date +%Y%m%d_%H%M%S)}"
TRAIN_HOST="${TRAIN_HOST:?set TRAIN_HOST to the ssh host alias of the training TPU VM}"
INFER_TPU_NAME="${INFER_TPU_NAME:?set INFER_TPU_NAME to the inference (vLLM) TPU VM (slice) name}"
INFER_ZONE="${INFER_ZONE:-europe-west4-b}"
GCS_ACTUAL_PATH="${GCS_ACTUAL_PATH:-gs://llmvae-checkpoints/two-model-exploration/checkpoints/${RUN_NAME}/actual}"

cat <<EOF
Serious Qwen3 run commands

0. Validate the serious-run path locally
bash scripts/validate_qwen3_serious_run.sh

0.5 Check TPU remote state before launch
# The helper now inspects the full ew4b inference slice by worker.
bash scripts/qwen3_serious_remote_state.sh check

0.6 If the hosts are polluted, clean them explicitly
# The helper now cleans all ew4b workers, not just worker 0 / the host alias.
CONFIRM_REMOTE_CLEANUP=1 bash scripts/qwen3_serious_remote_state.sh clean

1. Explorer novelty saturation run
# Full-slice repo sync now uses one ingress copy plus intra-slice fanout.
# Do not fall back to manual 16-worker IAP scp unless this path fails.
ALLOW_DIRTY=1 RUN_NAME=${NOVELTY_RUN_NAME} WANDB_RUN_NAME=${NOVELTY_RUN_NAME} \\
  bash scripts/launch_qwen3_explorer_novelty_checked.sh

2. Full one-round run
ALLOW_DIRTY=1 RUN_NAME=${RUN_NAME} WANDB_RUN_NAME=${RUN_NAME} \\
  bash scripts/launch_qwen3_one_round_checked.sh

3. Monitor train tmux on ew4a
ssh ${TRAIN_HOST} 'tmux capture-pane -t training_${RUN_NAME} -p -S -120'

4. Monitor inference slice state on ew4b
TARGET=infer INFER_TPU_NAME=${INFER_TPU_NAME} INFER_ZONE=${INFER_ZONE} \\
  bash scripts/qwen3_serious_remote_state.sh check

5. Fast sharded AIME24 avg@32 after Actual is saved
GCS_MODEL_PATH=${GCS_ACTUAL_PATH} USE_WANDB=1 \\
  bash scripts/run_qwen3_saved_aime24_avg32_sharded.sh
EOF
