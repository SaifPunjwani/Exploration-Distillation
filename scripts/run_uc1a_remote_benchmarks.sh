#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-64}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_PASS_K="${BENCHMARK_ROLLOUT_PASS_K:-1,4,8,16,32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-16384}"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
USE_WANDB="${USE_WANDB:-1}"
RUN_TAG="${RUN_TAG:-uc1a_bench_$(date +%Y%m%d_%H%M%S)}"
SKIP_CHIPS="${SKIP_CHIPS:-}"

should_skip_chip() {
  local chip="$1"
  case ",${SKIP_CHIPS}," in
    *,"${chip}",*) return 0 ;;
    *) return 1 ;;
  esac
}

launch_manifest() {
  local chip="$1"
  local session="$2"
  local manifest_path="$3"
  local run_group="$4"
  local remote_script="/tmp/${session}.sh"

  if should_skip_chip "$chip"; then
    echo "[bench] skipping chip=$chip session=$session"
    return 0
  fi

  tmux kill-session -t "$session" 2>/dev/null || true
  cat > "$remote_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
. .venv_tpu_slice/bin/activate
export PYTHONUNBUFFERED=1
export PYTHON_BIN=".venv_tpu_slice/bin/python"
export DEVICE="tpu"
export PJRT_DEVICE="TPU"
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_REQUIRE_TPU=1
export TMX_ALLOW_SINGLE_WORKER_TPU=1
export TMX_XLA_EVAL_NATIVE_GENERATE=1
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE="$TPU_ACCELERATOR_TYPE"
export TPU_VISIBLE_CHIPS="$chip"
export TPU_PROCESS_BOUNDS="1,1,1"
export TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1"
export TPU_CHIPS_PER_HOST_BOUNDS="1,1,1"
export TPU_HOST_BOUNDS="1,1,1"
export TPU_WORKER_ID="0"
export TPU_WORKER_HOSTNAMES="localhost"
export MANIFEST_PATH="$manifest_path"
export RUN_GROUP="$run_group"
export USE_WANDB="$USE_WANDB"
export BENCHMARK_DATASET_NAME="$BENCHMARK_DATASET_NAME"
export MAX_BENCHMARK="$MAX_BENCHMARK"
export BENCHMARK_NUM_ROLLOUTS="$BENCHMARK_NUM_ROLLOUTS"
export BENCHMARK_ROLLOUT_TEMPERATURE="$BENCHMARK_ROLLOUT_TEMPERATURE"
export BENCHMARK_ROLLOUT_TOP_P="$BENCHMARK_ROLLOUT_TOP_P"
export BENCHMARK_ROLLOUT_PASS_K="$BENCHMARK_ROLLOUT_PASS_K"
export BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN"
export BENCHMARK_EVAL_BATCH_SIZE="$BENCHMARK_EVAL_BATCH_SIZE"
export MAX_PROMPT_LEN="$MAX_PROMPT_LEN"
export MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN"
mkdir -p "runs/$run_group"
bash scripts/run_aime_model_herd.sh > "runs/$run_group/console.log" 2>&1
EOF
  chmod +x "$remote_script"
  tmux new-session -d -s "$session" "$remote_script"
}

launch_manifest 0 "uc1a_bench_actual_c0" "configs/aime_uc1a_core_actual.tsv" "${RUN_TAG}_actual"
launch_manifest 1 "uc1a_bench_raw_a_c1" "configs/aime_uc1a_core_raw_shard_a.tsv" "${RUN_TAG}_raw_a"
launch_manifest 2 "uc1a_bench_raw_b_c2" "configs/aime_uc1a_core_raw_shard_b.tsv" "${RUN_TAG}_raw_b"
launch_manifest 3 "uc1a_bench_raw_c_c3" "configs/aime_uc1a_core_raw_shard_c.tsv" "${RUN_TAG}_raw_c"

tmux ls | grep 'uc1a_bench_'
