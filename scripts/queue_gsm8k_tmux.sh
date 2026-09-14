#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

SESSION_NAME="${SESSION_NAME:-gsm8k_queue}"
QUEUE_TAG="${QUEUE_TAG:-gsm8k_queue_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/${QUEUE_TAG}}"
MODEL_FAMILY="${MODEL_FAMILY:-mixed}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv_tpu/bin/python" ]; then
    PYTHON_BIN=".venv_tpu/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export TOKENIZERS_PARALLELISM=false

probe_tpu() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v4-8")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
os.environ.setdefault("TPU_WORKER_ID", "0")
os.environ.setdefault("TPU_WORKER_HOSTNAMES", "localhost")
import torch_xla.core.xla_model as xm
xm.xla_device()
PY
}

DEVICE="${DEVICE:-auto}"
if [ "$DEVICE" = "auto" ]; then
  if probe_tpu; then
    DEVICE="tpu"
  else
    DEVICE="cpu"
  fi
fi

mkdir -p "$RUN_ROOT"

queue_body() {
  local device="$1"
  local model_family="$2"
  cat <<EOF
set -euo pipefail
cd "$ROOT_DIR"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_ENTITY="$WANDB_ENTITY"
export TOKENIZERS_PARALLELISM=false
echo "[queue] starting at \$(date -Iseconds)"
echo "[queue] device=$device model_family=$model_family run_root=$RUN_ROOT"
EOF

  if [ "$model_family" = "mixed" ] || [ "$model_family" = "small" ]; then
    cat <<EOF
"$PYTHON_BIN" main.py --device "$device" --use-wandb \
  --model-name xsanskarx/qwen2-0.5b_numina_math-instruct \
  --dataset-name gsm8k \
  --dataset-split-policy auto \
  --topic-split-mode heuristic_math_topics \
  --topic-partition-policy explicit_or_heuristic \
  --decoupling-mode decoupled \
  --actual-train-mode distill \
  --base-eval-mode grpo \
  --max-train-examples 32 \
  --max-eval-examples 32 \
  --grpo-max-steps 8 \
  --grpo-num-generations 2 \
  --grpo-generation-batch-size 4 \
  --num-samples-per-prompt 2 \
  --distill-epochs 2 \
  --distill-batch-size 2 \
  --max-distill-examples 64 \
  --trajectory-batch-size 4 \
  --eval-batch-size 4 \
  --lambda-novelty 0.1 \
  --novelty-feature-source multilayer \
  --novelty-metric sqrt_mse \
  --format-penalty-weight 0.2 \
  --max-prompt-len 192 \
  --max-completion-len 96 \
  --max-total-len 288 \
  --wandb-run-name "${QUEUE_TAG}_numina05_len96" \
  --plots-dir "${RUN_ROOT}/numina05_len96/plots" \
  --explorer-traj-path "${RUN_ROOT}/numina05_len96/explorer_trajectories.jsonl"
EOF
  fi

  if [ "$model_family" = "mixed" ] || [ "$model_family" = "large" ]; then
    cat <<EOF
"$PYTHON_BIN" main.py --device "$device" --use-wandb \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --dataset-name gsm8k \
  --dataset-split-policy auto \
  --topic-split-mode heuristic_math_topics \
  --topic-partition-policy explicit_or_heuristic \
  --decoupling-mode decoupled \
  --actual-train-mode distill \
  --base-eval-mode grpo \
  --max-train-examples 32 \
  --max-eval-examples 32 \
  --grpo-max-steps 8 \
  --grpo-num-generations 2 \
  --grpo-generation-batch-size 4 \
  --num-samples-per-prompt 2 \
  --distill-epochs 2 \
  --distill-batch-size 2 \
  --max-distill-examples 64 \
  --trajectory-batch-size 2 \
  --eval-batch-size 2 \
  --lambda-novelty 0.1 \
  --novelty-feature-source multilayer \
  --novelty-metric sqrt_mse \
  --format-penalty-weight 0.2 \
  --max-prompt-len 192 \
  --max-completion-len 160 \
  --max-total-len 352 \
  --wandb-run-name "${QUEUE_TAG}_qwen15_len160" \
  --plots-dir "${RUN_ROOT}/qwen15_len160/plots" \
  --explorer-traj-path "${RUN_ROOT}/qwen15_len160/explorer_trajectories.jsonl"

"$PYTHON_BIN" main.py --device "$device" --use-wandb \
  --model-name Qwen/Qwen2.5-Math-1.5B-Instruct \
  --dataset-name gsm8k \
  --dataset-split-policy auto \
  --topic-split-mode heuristic_math_topics \
  --topic-partition-policy explicit_or_heuristic \
  --decoupling-mode decoupled \
  --actual-train-mode distill \
  --base-eval-mode grpo \
  --max-train-examples 32 \
  --max-eval-examples 32 \
  --grpo-max-steps 8 \
  --grpo-num-generations 2 \
  --grpo-generation-batch-size 4 \
  --num-samples-per-prompt 2 \
  --distill-epochs 2 \
  --distill-batch-size 2 \
  --max-distill-examples 64 \
  --trajectory-batch-size 2 \
  --eval-batch-size 2 \
  --lambda-novelty 0.1 \
  --novelty-feature-source multilayer \
  --novelty-metric sqrt_mse \
  --format-penalty-weight 0.2 \
  --max-prompt-len 192 \
  --max-completion-len 224 \
  --max-total-len 416 \
  --wandb-run-name "${QUEUE_TAG}_qwen15_len224" \
  --plots-dir "${RUN_ROOT}/qwen15_len224/plots" \
  --explorer-traj-path "${RUN_ROOT}/qwen15_len224/explorer_trajectories.jsonl"
EOF
  fi

  cat <<EOF
echo "[queue] finished at \$(date -Iseconds)"
EOF
}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists"
  exit 1
fi

QUEUE_SCRIPT="$(mktemp)"
queue_body "$DEVICE" "$MODEL_FAMILY" >"$QUEUE_SCRIPT"
chmod +x "$QUEUE_SCRIPT"

tmux new-session -d -s "$SESSION_NAME" "bash '$QUEUE_SCRIPT'; rm -f '$QUEUE_SCRIPT'"
echo "Started tmux session: $SESSION_NAME"
echo "Device: $DEVICE"
echo "Run root: $RUN_ROOT"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Logs: tmux capture-pane -pt $SESSION_NAME"
