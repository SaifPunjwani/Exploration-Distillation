#!/usr/bin/env bash
set -euo pipefail

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-b}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3}"
SERVE_WORKERS="${SERVE_WORKERS:-4 5 6 7 8 9 10 11 12 13 14 15}"
TMX_USE_MANUAL_TPU_TOPOLOGY="${TMX_USE_MANUAL_TPU_TOPOLOGY:-0}"
TMX_JAX_CLUSTER_DETECTION_METHOD="${TMX_JAX_CLUSTER_DETECTION_METHOD-}"
JAX_INITIALIZATION_TIMEOUT="${JAX_INITIALIZATION_TIMEOUT:-600}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATASET_NAME="${DATASET_NAME:-${DATASET:-dapo_math_17k}}"
TMX_TRAIN_DATASET_JSONL="${TMX_TRAIN_DATASET_JSONL:-}"
RUN_NAME="${RUN_NAME:-jax_pipeline_lambda0_multihost_$(date +%Y%m%d_%H%M%S)}"
COORD_PORT="${COORD_PORT:-12345}"
VENV_NAME="${VENV_NAME:-tmx_jax}"
ENV_DIR="${ENV_DIR:-}"
REMOTE_REPO="${REMOTE_REPO:?set REMOTE_REPO to the absolute repo path on every TPU worker}"
SYNC_LOCAL_PACKAGE="${SYNC_LOCAL_PACKAGE:-1}"
TMX_FAST_PACKAGE_FANOUT="${TMX_FAST_PACKAGE_FANOUT:-1}"
TMX_PACKAGE_FANOUT_PORT="${TMX_PACKAGE_FANOUT_PORT:-18765}"
TMX_FAST_WORKER_SCRIPT_FANOUT="${TMX_FAST_WORKER_SCRIPT_FANOUT:-1}"
TMX_WORKER_SCRIPT_FANOUT_PORT="${TMX_WORKER_SCRIPT_FANOUT_PORT:-18766}"
TMX_GCLOUD_TPU_USE_IAP="${TMX_GCLOUD_TPU_USE_IAP:-1}"
TMX_VALIDATE_TRAIN_LIBTPU="${TMX_VALIDATE_TRAIN_LIBTPU:-1}"
TMX_REPAIR_TRAIN_LIBTPU="${TMX_REPAIR_TRAIN_LIBTPU:-0}"
TMX_LIBTPU_VERSION="${TMX_LIBTPU_VERSION:-0.0.20}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-500}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-8}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-2}"
TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-2}"
ACTUAL_ONLINE_GRPO_MAX_STEPS="${ACTUAL_ONLINE_GRPO_MAX_STEPS:-250}"
ACTUAL_GRPO_LR="${ACTUAL_GRPO_LR:-1e-6}"
ACTUAL_GRPO_KL_COEF="${ACTUAL_GRPO_KL_COEF:-0.10}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.0}"
INCORRECT_NOVELTY_SCALE="${INCORRECT_NOVELTY_SCALE:-0.0}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-8192}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-$((MAX_PROMPT_LEN + MAX_COMPLETION_LEN))}"
NOVELTY_FEATURE_MAX_LENGTH="${NOVELTY_FEATURE_MAX_LENGTH:-$MAX_TOTAL_LEN}"
LOGPROB_CHUNK_TOKENS="${LOGPROB_CHUNK_TOKENS:-256}"
GRPO_LR="${GRPO_LR:-5e-6}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-16}"
VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-8}"
TMX_VLLM_FANOUT_DESYNC="${TMX_VLLM_FANOUT_DESYNC:-0}"
TMX_VLLM_FANOUT_NONCE="${TMX_VLLM_FANOUT_NONCE:-0}"
TMX_VLLM_FILTER_DEAD_SERVERS="${TMX_VLLM_FILTER_DEAD_SERVERS:-1}"
TMX_VLLM_HEALTH_FILTER_TIMEOUT="${TMX_VLLM_HEALTH_FILTER_TIMEOUT:-2}"
TMX_VLLM_HEALTH_FILTER_TTL_SECONDS="${TMX_VLLM_HEALTH_FILTER_TTL_SECONDS:-30}"
TMX_VLLM_HEALTH_FILTER_CONCURRENCY="${TMX_VLLM_HEALTH_FILTER_CONCURRENCY:-64}"
TMX_VLLM_FAILOVER_ROUNDS="${TMX_VLLM_FAILOVER_ROUNDS:-2}"
TMX_VLLM_FAILOVER_ROUND_SLEEP="${TMX_VLLM_FAILOVER_ROUND_SLEEP:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
DYNAMIC_SAMPLING="${DYNAMIC_SAMPLING:-false}"
DYNAMIC_SAMPLING_MAX_ATTEMPTS="${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-8}"
DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-1}"
TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="${TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT}}"
MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-false}"
GRPO_KL_BETA="${GRPO_KL_BETA:-0.0}"
GRPO_CLIP_EPSILON="${GRPO_CLIP_EPSILON:-0.2}"
GRPO_CLIP_EPSILON_HIGH="${GRPO_CLIP_EPSILON_HIGH:-$GRPO_CLIP_EPSILON}"
GRPO_UPDATES_PER_ROLLOUT="${GRPO_UPDATES_PER_ROLLOUT:-1}"
GRPO_ADVANTAGE_NORMALIZATION="${GRPO_ADVANTAGE_NORMALIZATION:-none}"
TMX_ALLOW_NOVELTY_ADVANTAGE_NORMALIZATION="${TMX_ALLOW_NOVELTY_ADVANTAGE_NORMALIZATION:-0}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:-0.0}"
NONTERMINATION_PENALTY="${NONTERMINATION_PENALTY:-0.0}"
INVALID_ANSWER_PENALTY="${INVALID_ANSWER_PENALTY:-0.0}"
SOFT_OVERLONG_EXPECTED_LEN="${SOFT_OVERLONG_EXPECTED_LEN:-0}"
SOFT_OVERLONG_CACHE_LEN="${SOFT_OVERLONG_CACHE_LEN:-0}"
NOVELTY_ZSCORE="${NOVELTY_ZSCORE:-off}"
CORRECTNESS_ZSCORE="${CORRECTNESS_ZSCORE:-off}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-2}"
DISTILL_BATCH_SIZE="${DISTILL_BATCH_SIZE:-1}"
MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-20000}"
TMX_ALLOW_LONG_COMPLETION_TRAIN="${TMX_ALLOW_LONG_COMPLETION_TRAIN:-0}"
TMX_SKIP_FINAL_EVAL="${TMX_SKIP_FINAL_EVAL:-0}"
TMX_AIME_PROBE_EVERY_STEPS="${TMX_AIME_PROBE_EVERY_STEPS:-$SAVE_EVERY_STEPS}"
TMX_AIME_PROBE_NUM_PROBLEMS="${TMX_AIME_PROBE_NUM_PROBLEMS:-6}"
TMX_AIME_PROBE_NUM_ROLLOUTS="${TMX_AIME_PROBE_NUM_ROLLOUTS:-4}"
TMX_AIME_PROBE_MAX_TOKENS="${TMX_AIME_PROBE_MAX_TOKENS:-$MAX_COMPLETION_LEN}"
TMX_AIME_PROBE_CONCURRENCY="${TMX_AIME_PROBE_CONCURRENCY:-8}"
TMX_SAVE_BASE_POLICY_TRAJECTORIES="${TMX_SAVE_BASE_POLICY_TRAJECTORIES:-1}"
TMX_BASE_POLICY_MAX_COMPLETION_LEN="${TMX_BASE_POLICY_MAX_COMPLETION_LEN:-$MAX_COMPLETION_LEN}"
TMX_EFFECTIVE_PROMPTS_PER_STEP="${TMX_EFFECTIVE_PROMPTS_PER_STEP:-}"
TMX_BATCH_SHARD_SOURCE_HOST="${TMX_BATCH_SHARD_SOURCE_HOST:-}"
TMX_BATCH_SHARD_HTTP_PORT="${TMX_BATCH_SHARD_HTTP_PORT:-18777}"
TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS="${TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS:-90}"
TMX_BATCH_SHARD_HTTP_ATTEMPTS="${TMX_BATCH_SHARD_HTTP_ATTEMPTS:-240}"
TMX_BATCH_SHARD_HTTP_TIMEOUT="${TMX_BATCH_SHARD_HTTP_TIMEOUT:-120}"
MIN_ACCEPTED_TRAJECTORIES="${MIN_ACCEPTED_TRAJECTORIES:-2048}"
MAX_ACCEPTED_TRAJECTORIES="${MAX_ACCEPTED_TRAJECTORIES:-0}"
ACCEPTED_SELECTION_POLICY="${ACCEPTED_SELECTION_POLICY:-quality_first}"
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-64}"
TMX_ARTIFACT_BACKEND="${TMX_ARTIFACT_BACKEND:-hf}"
TMX_ALLOW_GCS_ARTIFACTS="${TMX_ALLOW_GCS_ARTIFACTS:-0}"
TMX_GCS_CHECKPOINT_BASE="${TMX_GCS_CHECKPOINT_BASE:-}"
VLLM_URLS_OVERRIDE="${TMX_VLLM_SERVER_URLS:-${VLLM_SERVER_URLS:-}}"
VLLM_RELOAD_TPU_NAME="${VLLM_RELOAD_TPU_NAME:-$TPU_NAME}"
VLLM_RELOAD_TPU_PROJECT="${VLLM_RELOAD_TPU_PROJECT:-${TMX_VLLM_RELOAD_TPU_PROJECT:-${TPU_PROJECT:-${CLOUDSDK_CORE_PROJECT:-}}}}"
VLLM_RELOAD_ZONE="${VLLM_RELOAD_ZONE:-$ZONE}"
VLLM_RELOAD_WORKERS="${VLLM_RELOAD_WORKERS:-}"
TMX_VLLM_MULTI_SERVER="${TMX_VLLM_MULTI_SERVER:-0}"
TMX_VLLM_MULTI_SERVER_CHIPS="${TMX_VLLM_MULTI_SERVER_CHIPS:-0 1 2 3}"
TMX_VLLM_MULTI_SERVER_PORT_BASE="${TMX_VLLM_MULTI_SERVER_PORT_BASE:-8000}"
WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
if [ -z "$WANDB_API_KEY_VALUE" ] && [ -f "$HOME/.netrc" ]; then
  WANDB_API_KEY_VALUE="$(
    awk '
      $1 == "machine" && $2 == "api.wandb.ai" { in_wandb = 1; next }
      $1 == "machine" { in_wandb = 0 }
      in_wandb && $1 == "password" { print $2; exit }
      in_wandb {
        for (i = 1; i <= NF; i++) {
          if ($i == "password" && (i + 1) <= NF) { print $(i + 1); exit }
        }
      }
    ' "$HOME/.netrc"
  )"
fi
HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "$HF_TOKEN_VALUE" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN_VALUE="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
fi

GCLOUD_TPU_IAP_FLAGS=""
if [ "$TMX_GCLOUD_TPU_USE_IAP" = "1" ]; then
  GCLOUD_TPU_IAP_FLAGS="--tunnel-through-iap"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKER_IPS_RAW="$(
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --zone "$ZONE" \
    --format='get(networkEndpoints[].ipAddress)'
)"
IFS=';' read -r -a WORKER_IPS <<< "$WORKER_IPS_RAW"

train_workers=($TRAIN_WORKERS)
serve_workers=($SERVE_WORKERS)

max_train_worker=-1
for w in "${train_workers[@]}"; do
  if [ "$w" -gt "$max_train_worker" ]; then
    max_train_worker="$w"
  fi
done
if [ "${#WORKER_IPS[@]}" -le "$max_train_worker" ]; then
  echo "ERROR: train worker list [$TRAIN_WORKERS] requires worker index $max_train_worker, got ${#WORKER_IPS[@]} TPU worker IPs from: $WORKER_IPS_RAW" >&2
  exit 1
fi

default_host_bounds=""
case "${#train_workers[@]}" in
  1) default_host_bounds="1,1,1" ;;
  4) default_host_bounds="2,2,1" ;;
  16) default_host_bounds="4,4,1" ;;
  *)
    if [ "$TMX_USE_MANUAL_TPU_TOPOLOGY" = "1" ] && [ -z "${TMX_TPU_HOST_BOUNDS:-}" ]; then
      echo "ERROR: set TMX_TPU_HOST_BOUNDS for ${#train_workers[@]} train workers" >&2
      exit 1
    fi
    ;;
esac
TPU_HOST_BOUNDS_VALUE="${TMX_TPU_HOST_BOUNDS:-$default_host_bounds}"
TPU_VISIBLE_CHIPS_VALUE="${TMX_TPU_VISIBLE_CHIPS:-0,1,2,3}"
TPU_CHIPS_PER_HOST_BOUNDS_VALUE="${TMX_TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE="${TMX_TPU_CHIPS_PER_PROCESS_BOUNDS:-$TPU_CHIPS_PER_HOST_BOUNDS_VALUE}"

coord_worker="${train_workers[0]}"
coord_ip="${WORKER_IPS[$coord_worker]}"
train_worker_hosts=""
for w in "${train_workers[@]}"; do
  if [ -n "$train_worker_hosts" ]; then
    train_worker_hosts+=","
  fi
  train_worker_hosts+="${WORKER_IPS[$w]}"
done

vllm_urls="$VLLM_URLS_OVERRIDE"
reload_workers="$VLLM_RELOAD_WORKERS"
if [ -z "$vllm_urls" ]; then
  for w in "${serve_workers[@]}"; do
    ip="${WORKER_IPS[$w]}"
    if gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$w" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
      --command='curl -sf --max-time 2 http://localhost:8000/health >/dev/null && echo OK || echo DOWN' 2>/dev/null | grep -q OK; then
      if [ -n "$vllm_urls" ]; then
        vllm_urls+=","
      fi
      vllm_urls+="http://${ip}:8000/v1"
      if [ -z "$reload_workers" ]; then
        reload_workers="$w"
      else
        reload_workers+=" $w"
      fi
    fi
  done
fi

if [ -z "$vllm_urls" ]; then
  echo "ERROR: no healthy vLLM workers found in SERVE_WORKERS=$SERVE_WORKERS" >&2
  exit 1
fi

validate_train_libtpu() {
  if [ "$TMX_VALIDATE_TRAIN_LIBTPU" != "1" ]; then
    return 0
  fi
  echo "[preflight] validating libtpu on train workers=[$TRAIN_WORKERS]"
  local failed=0
  for w in "${train_workers[@]}"; do
    if ! gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$w" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
      --ssh-flag="-o ConnectTimeout=${TMX_PREFLIGHT_CONNECT_TIMEOUT:-20}" \
      --command="
        set -euo pipefail
        if [ -n '$ENV_DIR' ] && [ -f '$ENV_DIR/bin/activate' ]; then
          source '$ENV_DIR/bin/activate'
        elif [ -f \"\$HOME/miniforge3/bin/activate\" ]; then
          source \"\$HOME/miniforge3/bin/activate\" '$VENV_NAME' 2>/dev/null || true
        elif [ -f \"\$HOME/vllm_tpu_env/bin/activate\" ]; then
          source \"\$HOME/vllm_tpu_env/bin/activate\"
        elif [ -x \"\$HOME/vllm_tpu_env/bin/python\" ]; then
          export PATH=\"\$HOME/vllm_tpu_env/bin:\$PATH\"
        fi
        if python - <<'PY'
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec('libtpu')
if spec is None or not spec.submodule_search_locations:
    sys.exit(1)
path = pathlib.Path(list(spec.submodule_search_locations)[0]) / 'libtpu.so'
sys.exit(0 if path.exists() else 1)
PY
        then
          exit 0
        fi
        if [ '$TMX_REPAIR_TRAIN_LIBTPU' = '1' ]; then
          python -m pip install --quiet --disable-pip-version-check 'libtpu==$TMX_LIBTPU_VERSION'
          python - <<'PY'
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec('libtpu')
if spec is None or not spec.submodule_search_locations:
    sys.exit(1)
path = pathlib.Path(list(spec.submodule_search_locations)[0]) / 'libtpu.so'
sys.exit(0 if path.exists() else 1)
PY
          exit 0
        fi
        echo 'missing libtpu.so in selected Python env; set ENV_DIR or TMX_REPAIR_TRAIN_LIBTPU=1' >&2
        exit 1
      " >/dev/null; then
      echo "[preflight] ERROR: train worker $w missing usable libtpu.so" >&2
      failed=1
    fi
  done
  if [ "$failed" != "0" ]; then
    exit 1
  fi
}

validate_train_libtpu

LOCAL_TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$LOCAL_TMP_DIR"
}
trap cleanup EXIT

PACKAGE_ARCHIVE=""
PACKAGE_SYNC_DONE=0
if [ "$SYNC_LOCAL_PACKAGE" = "1" ]; then
  PACKAGE_ARCHIVE="$LOCAL_TMP_DIR/tmx_jax_payload.tar.gz"
  COPYFILE_DISABLE=1 tar -czf "$PACKAGE_ARCHIVE" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -C "$LOCAL_REPO_ROOT" \
    tmx_jax

  if [ "$TMX_FAST_PACKAGE_FANOUT" = "1" ]; then
    package_name="${RUN_NAME}_tmx_jax_payload.tar.gz"
    remote_archive="/tmp/${package_name}"
    echo "[sync] uploading tmx_jax package once to worker ${coord_worker}"
    gcloud alpha compute tpus tpu-vm scp "$PACKAGE_ARCHIVE" "$TPU_NAME:${remote_archive}" \
      --worker="$coord_worker" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS >/dev/null
    echo "[sync] starting package fanout server on worker ${coord_worker}:${TMX_PACKAGE_FANOUT_PORT}"
    gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$coord_worker" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
      --command="
        set -euo pipefail
        pkill -f '[h]ttp.server ${TMX_PACKAGE_FANOUT_PORT}' 2>/dev/null || true
        cd /tmp
        nohup python3 -m http.server '${TMX_PACKAGE_FANOUT_PORT}' --bind 0.0.0.0 \
          >/tmp/tmx_pkg_http_${TMX_PACKAGE_FANOUT_PORT}.log 2>&1 &
      " >/dev/null
    echo "[sync] fanning out package internally to ${#train_workers[@]} train workers"
    gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker=all --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
      --command="
        set -euo pipefail
        worker_id=\"\$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number 2>/dev/null || echo \${TPU_WORKER_ID:-0})\"
        case ' ${TRAIN_WORKERS} ' in
          *' '\${worker_id}' '*) ;;
          *) exit 0 ;;
        esac
        url='http://${coord_ip}:${TMX_PACKAGE_FANOUT_PORT}/${package_name}'
        dst='/tmp/${package_name}'
        python3 - \"\$url\" \"\$dst\" <<'PY'
import sys
import urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
        rm -rf '${REMOTE_REPO}/tmx_jax'
        mkdir -p '${REMOTE_REPO}'
        tar -xzf \"\$dst\" -C '${REMOTE_REPO}'
      " >/dev/null
    PACKAGE_SYNC_DONE=1
  fi
fi

for idx in "${!train_workers[@]}"; do
  w="${train_workers[$idx]}"
  worker_script="$LOCAL_TMP_DIR/${RUN_NAME}_w${w}.sh"
  cat > "$worker_script" <<INNER
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_REPO'
if [ -n '$ENV_DIR' ] && [ -f '$ENV_DIR/bin/activate' ]; then
  source '$ENV_DIR/bin/activate'
elif [ -f "\$HOME/miniforge3/bin/activate" ]; then
  source "\$HOME/miniforge3/bin/activate" '$VENV_NAME'
elif [ -f "\$HOME/vllm_tpu_env/bin/activate" ]; then
  source "\$HOME/vllm_tpu_env/bin/activate"
elif [ -x "\$HOME/vllm_tpu_env/bin/python" ]; then
  export PATH="\$HOME/vllm_tpu_env/bin:\$PATH"
else
  echo "ERROR: no usable Python env found; expected ~/miniforge3 or ~/vllm_tpu_env" >&2
  exit 1
fi
python - <<'PY' >/dev/null 2>&1 || python -m pip install --quiet jinja2
import jinja2  # noqa: F401
PY
export WANDB_PROJECT='two-model-explore'
export WANDB_ENTITY='${WANDB_ENTITY:-}'
$(if [ -n "$WANDB_API_KEY_VALUE" ]; then printf "export WANDB_API_KEY='%s'\n" "$WANDB_API_KEY_VALUE"; fi)
$(if [ -n "$HF_TOKEN_VALUE" ]; then printf "export HF_TOKEN='%s'\nexport HUGGING_FACE_HUB_TOKEN='%s'\n" "$HF_TOKEN_VALUE" "$HF_TOKEN_VALUE"; fi)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PJRT_DEVICE=TPU
export TPU_ACCELERATOR_TYPE='${TPU_ACCELERATOR_TYPE}'
if [ '${TMX_USE_MANUAL_TPU_TOPOLOGY}' = '1' ]; then
  export TPU_SKIP_MDS_QUERY="\${TPU_SKIP_MDS_QUERY:-1}"
  export TPU_VISIBLE_CHIPS='${TPU_VISIBLE_CHIPS_VALUE}'
  export TPU_WORKER_ID='${idx}'
  export TMX_TPU_WORKER_IP='${WORKER_IPS[$w]}'
  export TPU_WORKER_HOSTNAMES='${train_worker_hosts}'
  export TPU_HOST_BOUNDS='${TPU_HOST_BOUNDS_VALUE}'
  export TPU_PROCESS_BOUNDS='${TPU_HOST_BOUNDS_VALUE}'
  export TPU_CHIPS_PER_HOST_BOUNDS='${TPU_CHIPS_PER_HOST_BOUNDS_VALUE}'
  export TPU_CHIPS_PER_PROCESS_BOUNDS='${TPU_CHIPS_PER_PROCESS_BOUNDS_VALUE}'
  export JAX_LOCAL_DEVICE_IDS='${TPU_VISIBLE_CHIPS_VALUE}'
  if [ -n '${TMX_JAX_CLUSTER_DETECTION_METHOD}' ]; then
    export TMX_JAX_CLUSTER_DETECTION_METHOD='${TMX_JAX_CLUSTER_DETECTION_METHOD}'
  else
    unset TMX_JAX_CLUSTER_DETECTION_METHOD
  fi
else
  unset TPU_SKIP_MDS_QUERY TPU_VISIBLE_CHIPS TPU_WORKER_ID TPU_WORKER_HOSTNAMES TPU_HOST_BOUNDS TPU_PROCESS_BOUNDS
  unset TPU_CHIPS_PER_HOST_BOUNDS TPU_CHIPS_PER_PROCESS_BOUNDS JAX_LOCAL_DEVICE_IDS
  if [ -n '${TMX_JAX_CLUSTER_DETECTION_METHOD}' ]; then
    export TMX_JAX_CLUSTER_DETECTION_METHOD='${TMX_JAX_CLUSTER_DETECTION_METHOD}'
  else
    unset TMX_JAX_CLUSTER_DETECTION_METHOD
  fi
fi
export JAX_COORDINATOR_ADDRESS='${coord_ip}:${COORD_PORT}'
export JAX_NUM_PROCESSES='${#train_workers[@]}'
export JAX_PROCESS_ID='${idx}'
export JAX_INITIALIZATION_TIMEOUT='${JAX_INITIALIZATION_TIMEOUT}'
export JAX_HEARTBEAT_TIMEOUT_SECONDS='${JAX_HEARTBEAT_TIMEOUT_SECONDS:-900}'
$(if [ "$idx" = "0" ]; then printf "export JAX_COORDINATOR_BIND_ADDRESS='0.0.0.0:%s'\n" "$COORD_PORT"; fi)
export TMX_DISABLE_WANDB_NONZERO="\${TMX_DISABLE_WANDB_NONZERO:-1}"
export TMX_USE_INLINE_OLD_LOGPROBS=1
export RUNS_ROOT="\${RUNS_ROOT:-\$HOME/two-model-exploration/runs}"
export TMX_PREFETCH_ROLLOUTS="\${TMX_PREFETCH_ROLLOUTS:-1}"
export TMX_SAVE_OPT_STATE="\${TMX_SAVE_OPT_STATE:-0}"
export TMX_RESTORE_OPT_STATE="\${TMX_RESTORE_OPT_STATE:-1}"
export TMX_INIT_EMPTY_FOR_RESTORE="\${TMX_INIT_EMPTY_FOR_RESTORE:-0}"
export TMX_CKPT_DELETE_OLD_RUNS="\${TMX_CKPT_DELETE_OLD_RUNS:-1}"
export TMX_LOCAL_CHECKPOINT_EVERY_STEPS="\${TMX_LOCAL_CHECKPOINT_EVERY_STEPS:-0}"
export TMX_HF_CHECKPOINT_EVERY_STEPS='${TMX_HF_CHECKPOINT_EVERY_STEPS:-${TMX_GCS_CHECKPOINT_EVERY_STEPS:-${SAVE_EVERY_STEPS}}}'
export TMX_HF_CHECKPOINT_FIRST_STEP='${TMX_HF_CHECKPOINT_FIRST_STEP:-${TMX_GCS_CHECKPOINT_FIRST_STEP:-0}}'
export TMX_GCS_CHECKPOINT_EVERY_STEPS="\${TMX_GCS_CHECKPOINT_EVERY_STEPS:-\$TMX_HF_CHECKPOINT_EVERY_STEPS}"
export TMX_GCS_CHECKPOINT_FIRST_STEP="\${TMX_GCS_CHECKPOINT_FIRST_STEP:-\$TMX_HF_CHECKPOINT_FIRST_STEP}"
export TMX_ARTIFACT_BACKEND='${TMX_ARTIFACT_BACKEND}'
export TMX_ALLOW_GCS_ARTIFACTS='${TMX_ALLOW_GCS_ARTIFACTS}'
export TMX_ORBAX_DIRECT_GCS_CHECKPOINT="\${TMX_ORBAX_DIRECT_GCS_CHECKPOINT:-0}"
export TMX_SKIP_MULTIHOST_LOCAL_ONLY_CHECKPOINT="\${TMX_SKIP_MULTIHOST_LOCAL_ONLY_CHECKPOINT:-1}"
export TMX_REQUIRE_GCS_CHECKPOINT_UPLOAD="\${TMX_REQUIRE_GCS_CHECKPOINT_UPLOAD:-0}"
export TMX_LOCAL_CHECKPOINT_KEEP_LAST="\${TMX_LOCAL_CHECKPOINT_KEEP_LAST:-1}"
export TMX_LOCAL_VLLM_RELOAD_KEEP_LAST="\${TMX_LOCAL_VLLM_RELOAD_KEEP_LAST:-0}"
export TMX_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD="\${TMX_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD:-1}"
export TMX_CKPT_MIN_FREE_GB="\${TMX_CKPT_MIN_FREE_GB:-28}"
export TMX_VLLM_EXPORT_MIN_FREE_GB="\${TMX_VLLM_EXPORT_MIN_FREE_GB:-28}"
export TMX_WANDB_SINGLE_RUN="\${TMX_WANDB_SINGLE_RUN:-1}"
export TMX_WANDB_COMPACT="\${TMX_WANDB_COMPACT:-1}"
export TMX_WANDB_MINIMAL="\${TMX_WANDB_MINIMAL:-1}"
export TMX_WANDB_EVAL_PROGRESS="\${TMX_WANDB_EVAL_PROGRESS:-0}"
export TMX_SAVE_BASE_POLICY_TRAJECTORIES='${TMX_SAVE_BASE_POLICY_TRAJECTORIES:-1}'
export TMX_BASE_POLICY_TRAJECTORY_BATCHES='${TMX_BASE_POLICY_TRAJECTORY_BATCHES:-1}'
export TMX_BASE_POLICY_MAX_COMPLETION_LEN='${TMX_BASE_POLICY_MAX_COMPLETION_LEN:-${MAX_COMPLETION_LEN}}'
export TMX_HF_CHECKPOINT_REPO="\${TMX_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}"
export TMX_HF_ARTIFACT_PREFIX="\${TMX_HF_ARTIFACT_PREFIX:-${RUN_NAME}}"
export TMX_HF_MIRROR_CHECKPOINTS="\${TMX_HF_MIRROR_CHECKPOINTS:-1}"
export TMX_HF_MIRROR_TRAJECTORIES="\${TMX_HF_MIRROR_TRAJECTORIES:-1}"
export TMX_HF_MIRROR_METRICS="\${TMX_HF_MIRROR_METRICS:-1}"
export TMX_HF_MIRROR_VLLM_EXPORTS='${TMX_HF_MIRROR_VLLM_EXPORTS:-1}'
export TMX_HF_MIRROR_EVAL_PROBES="\${TMX_HF_MIRROR_EVAL_PROBES:-1}"
export TMX_HF_MIRROR_RUN_SUMMARY="\${TMX_HF_MIRROR_RUN_SUMMARY:-1}"
export TMX_HF_MIRROR_FINAL_EXPORTS="\${TMX_HF_MIRROR_FINAL_EXPORTS:-1}"
export TMX_HF_MIRROR_FINAL_EVAL="\${TMX_HF_MIRROR_FINAL_EVAL:-1}"
export TMX_HF_MIRROR_ASYNC="\${TMX_HF_MIRROR_ASYNC:-1}"
export TMX_REQUIRE_HF_ARTIFACT_UPLOAD="\${TMX_REQUIRE_HF_ARTIFACT_UPLOAD:-0}"
export TMX_VLLM_RETURN_LOGPROBS="\${TMX_VLLM_RETURN_LOGPROBS:-0}"
export TMX_TRACK_EXTERNAL_OLD_GAP="\${TMX_TRACK_EXTERNAL_OLD_GAP:-0}"
export TMX_GRPO_ROW_VECTOR_CHUNK='${TMX_GRPO_ROW_VECTOR_CHUNK:-1}'
export TMX_VECTORIZE_GRPO_ROWS='${TMX_VECTORIZE_GRPO_ROWS:-0}'
export TMX_DISABLE_ROLLOUT_DIVERSITY='${TMX_DISABLE_ROLLOUT_DIVERSITY:-0}'
export TMX_DAPO_DRGRPO_FULL_CONTRACT='${TMX_DAPO_DRGRPO_FULL_CONTRACT:-0}'
export TMX_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN='${TMX_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN:-20480}'
export TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN='${TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN:-16384}'
export TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN='${TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN:-4096}'
export TMX_EXPECTED_GRPO_BATCH_SIZE='${TMX_EXPECTED_GRPO_BATCH_SIZE:-${GRPO_BATCH_SIZE:-1}}'
export TMX_EXPECTED_GRPO_GRAD_ACCUM='${TMX_EXPECTED_GRPO_GRAD_ACCUM:-${GRPO_GRAD_ACCUM:-2}}'
export TMX_EXPECTED_GRPO_UPDATES_PER_ROLLOUT='${TMX_EXPECTED_GRPO_UPDATES_PER_ROLLOUT:-${GRPO_UPDATES_PER_ROLLOUT:-1}}'
export TMX_EFFECTIVE_PROMPTS_PER_STEP='${TMX_EFFECTIVE_PROMPTS_PER_STEP:-}'
export TMX_REPLICATE_ROLLOUT_BATCH='${TMX_REPLICATE_ROLLOUT_BATCH:-0}'
export TMX_BATCH_SHARD_SOURCE_HOST='${TMX_BATCH_SHARD_SOURCE_HOST:-}'
export TMX_BATCH_SHARD_HTTP='${TMX_BATCH_SHARD_HTTP:-1}'
export TMX_BATCH_SHARD_HTTP_PORT='${TMX_BATCH_SHARD_HTTP_PORT:-18777}'
export TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS='${TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS:-90}'
export TMX_BATCH_SHARD_HTTP_ATTEMPTS='${TMX_BATCH_SHARD_HTTP_ATTEMPTS:-240}'
export TMX_BATCH_SHARD_HTTP_TIMEOUT='${TMX_BATCH_SHARD_HTTP_TIMEOUT:-120}'
export TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS='${TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS:-${SAVE_EVERY_STEPS}}'
export TMX_TRAIN_VLLM_RELOAD_FIRST_STEP='${TMX_TRAIN_VLLM_RELOAD_FIRST_STEP:-0}'
export TMX_TRAIN_VLLM_RELOAD_ON_RESUME='${TMX_TRAIN_VLLM_RELOAD_ON_RESUME:-1}'
export TMX_TRAIN_VLLM_RELOAD_MODE='${TMX_TRAIN_VLLM_RELOAD_MODE:-external}'
export TMX_VLLM_RELOAD_TRANSPORT='${TMX_VLLM_RELOAD_TRANSPORT:-hf}'
export TMX_VLLM_INTERNAL_RELOAD_PORT='${TMX_VLLM_INTERNAL_RELOAD_PORT:-18765}'
export TMX_VLLM_INTERNAL_RELOAD_HOST='${TMX_VLLM_INTERNAL_RELOAD_HOST:-}'
export TMX_EXTERNAL_VLLM_RELOAD_WAIT='${TMX_EXTERNAL_VLLM_RELOAD_WAIT:-1}'
export TMX_EXTERNAL_VLLM_RELOAD_REQUIRED='${TMX_EXTERNAL_VLLM_RELOAD_REQUIRED:-1}'
export TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL='${TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL:-1}'
export TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT='${TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT:-7200}'
export TMX_VLLM_RELOAD_ALLOW_PARALLEL='${TMX_VLLM_RELOAD_ALLOW_PARALLEL:-0}'
export TMX_VLLM_RELOAD_PARALLELISM='${TMX_VLLM_RELOAD_PARALLELISM:-8}'
if [ -z "${TPU_TRANSPORT_FLAG:-}" ]; then
  if [ "${TMX_GCLOUD_TPU_USE_IAP:-1}" = "1" ]; then
    export TPU_TRANSPORT_FLAG='--tunnel-through-iap'
  else
    export TPU_TRANSPORT_FLAG='direct'
  fi
else
  export TPU_TRANSPORT_FLAG='${TPU_TRANSPORT_FLAG}'
fi
export TMX_VLLM_EXPORT_FROM_LIVE_ALLGATHER='${TMX_VLLM_EXPORT_FROM_LIVE_ALLGATHER:-1}'
export TMX_VLLM_EXPORT_FROM_GCS_CHECKPOINT='${TMX_VLLM_EXPORT_FROM_GCS_CHECKPOINT:-0}'
export TMX_TRAIN_RELOAD_VLLM_MAX_NUM_SEQS='${TMX_TRAIN_RELOAD_VLLM_MAX_NUM_SEQS:-32}'
export TMX_ACTUAL_GRPO_VLLM_MAX_NUM_SEQS='${TMX_ACTUAL_GRPO_VLLM_MAX_NUM_SEQS:-32}'
export TMX_VLLM_RELOAD_MAX_BATCHED_TOKENS='${TMX_VLLM_RELOAD_MAX_BATCHED_TOKENS:-65536}'
export TMX_VLLM_SERVER_URLS='${vllm_urls}'
export TMX_VLLM_RELOAD_SERVER_URLS='${vllm_urls}'
export TMX_VLLM_RELOAD_TPU_NAME='${VLLM_RELOAD_TPU_NAME}'
export TMX_VLLM_RELOAD_TPU_PROJECT='${VLLM_RELOAD_TPU_PROJECT}'
export TMX_VLLM_RELOAD_ZONE='${VLLM_RELOAD_ZONE}'
export TMX_VLLM_RELOAD_WORKERS='${reload_workers}'
export TMX_VLLM_MULTI_SERVER='${TMX_VLLM_MULTI_SERVER}'
export TMX_VLLM_MULTI_SERVER_CHIPS='${TMX_VLLM_MULTI_SERVER_CHIPS}'
export TMX_VLLM_MULTI_SERVER_PORT_BASE='${TMX_VLLM_MULTI_SERVER_PORT_BASE}'
export VLLM_ENABLE_CHUNKED_PREFILL='${VLLM_ENABLE_CHUNKED_PREFILL:-1}'
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION='${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-1}'
export TMX_VLLM_V1_USE_PREFILL_DECODE_ATTENTION='${TMX_VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}}'
export VLLM_XLA_CACHE_TAG='${VLLM_XLA_CACHE_TAG:-tp${TP_SIZE:-4}_pda1_mlen${MAX_TOTAL_LEN}_mbt${TMX_VLLM_RELOAD_MAX_BATCHED_TOKENS:-${MAX_TOTAL_LEN}}}'
export TMX_VLLM_XLA_CACHE_TAG='${TMX_VLLM_XLA_CACHE_TAG:-${VLLM_XLA_CACHE_TAG:-}}'
export TMX_GCS_CHECKPOINT_BASE='${TMX_GCS_CHECKPOINT_BASE}'
export RUN_NAME='${RUN_NAME}'
export MODEL_NAME='${MODEL_NAME}'
export DATASET_NAME='${DATASET_NAME}'
export TMX_TRAIN_DATASET_JSONL='${TMX_TRAIN_DATASET_JSONL}'
export MAX_TRAIN_EXAMPLES='${MAX_TRAIN_EXAMPLES}'
export INIT_CHECKPOINT='${INIT_CHECKPOINT:-}'
export TMX_INIT_CHECKPOINT_LOCAL_PATH='${TMX_INIT_CHECKPOINT_LOCAL_PATH:-}'
export GRPO_MAX_STEPS='${GRPO_MAX_STEPS}'
export GRPO_NUM_GENERATIONS='${GRPO_NUM_GENERATIONS}'
export GRPO_BATCH_SIZE='${GRPO_BATCH_SIZE:-1}'
export GRPO_GRAD_ACCUM='${GRPO_GRAD_ACCUM:-2}'
export TRAJECTORY_BATCH_SIZE='${TRAJECTORY_BATCH_SIZE}'
export ACTUAL_ONLINE_GRPO_MAX_STEPS='${ACTUAL_ONLINE_GRPO_MAX_STEPS}'
export ACTUAL_GRPO_LR='${ACTUAL_GRPO_LR}'
export ACTUAL_GRPO_KL_COEF='${ACTUAL_GRPO_KL_COEF}'
export SAVE_EVERY_STEPS='${SAVE_EVERY_STEPS}'
export LAMBDA_NOVELTY='${LAMBDA_NOVELTY}'
export INCORRECT_NOVELTY_SCALE='${INCORRECT_NOVELTY_SCALE}'
export MAX_PROMPT_LEN='${MAX_PROMPT_LEN}'
export MAX_COMPLETION_LEN='${MAX_COMPLETION_LEN}'
export MAX_TOTAL_LEN='${MAX_TOTAL_LEN}'
export NOVELTY_FEATURE_MAX_LENGTH='${NOVELTY_FEATURE_MAX_LENGTH}'
export TMX_NOVELTY_TRUNCATION_MODE='${TMX_NOVELTY_TRUNCATION_MODE:-tokenizer}'
export TMX_NOVELTY_TEXT_HASH_FALLBACK='${TMX_NOVELTY_TEXT_HASH_FALLBACK:-0}'
export TMX_NOVELTY_BROADCAST_DEBUG='${TMX_NOVELTY_BROADCAST_DEBUG:-0}'
export TMX_NOVELTY_BATCH_SIZE='${TMX_NOVELTY_BATCH_SIZE:-1}'
export TMX_NOVELTY_PROGRESS_EVERY_ROWS='${TMX_NOVELTY_PROGRESS_EVERY_ROWS:-32}'
export LOGPROB_CHUNK_TOKENS='${LOGPROB_CHUNK_TOKENS}'
export GRPO_LR='${GRPO_LR}'
export VLLM_CONCURRENCY='${VLLM_CONCURRENCY}'
export VLLM_FANOUT_PER_PROMPT='${VLLM_FANOUT_PER_PROMPT}'
export TMX_HTTP_POOL_MAXSIZE='${TMX_HTTP_POOL_MAXSIZE:-512}'
export TMX_VLLM_ENDPOINT_MAX_INFLIGHT='${TMX_VLLM_ENDPOINT_MAX_INFLIGHT:-1}'
export TMX_VLLM_ALLOW_ERROR_COMPLETIONS='${TMX_VLLM_ALLOW_ERROR_COMPLETIONS:-0}'
export TMX_VLLM_FANOUT_DESYNC='${TMX_VLLM_FANOUT_DESYNC}'
export TMX_VLLM_FANOUT_NONCE='${TMX_VLLM_FANOUT_NONCE}'
export TEMPERATURE='${TEMPERATURE}'
export TOP_P='${TOP_P}'
export TOP_K='${TOP_K}'
export DYNAMIC_SAMPLING='${DYNAMIC_SAMPLING}'
export DYNAMIC_SAMPLING_MAX_ATTEMPTS='${DYNAMIC_SAMPLING_MAX_ATTEMPTS}'
export DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT='${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT}'
export TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT='${TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT}'
export MASK_TRUNCATED_COMPLETIONS='${MASK_TRUNCATED_COMPLETIONS}'
export GRPO_KL_BETA='${GRPO_KL_BETA}'
export GRPO_CLIP_EPSILON='${GRPO_CLIP_EPSILON}'
export GRPO_CLIP_EPSILON_HIGH='${GRPO_CLIP_EPSILON_HIGH}'
export GRPO_UPDATES_PER_ROLLOUT='${GRPO_UPDATES_PER_ROLLOUT}'
export GRPO_ADVANTAGE_NORMALIZATION='${GRPO_ADVANTAGE_NORMALIZATION}'
export TMX_ALLOW_NOVELTY_ADVANTAGE_NORMALIZATION='${TMX_ALLOW_NOVELTY_ADVANTAGE_NORMALIZATION}'
$(if [ -n "${TMX_WANDB_RUN_ID:-}" ]; then printf "export TMX_WANDB_RUN_ID='%s'\nexport TMX_WANDB_EXPLORER_RUN_ID='%s'\nexport TMX_WANDB_DISTILL_RUN_ID='%s'\nexport TMX_WANDB_ACTUAL_RUN_ID='%s'\nexport TMX_WANDB_ACTUAL_GRPO_RUN_ID='%s'\n" "${TMX_WANDB_RUN_ID}" "${TMX_WANDB_EXPLORER_RUN_ID:-${TMX_WANDB_RUN_ID}}" "${TMX_WANDB_DISTILL_RUN_ID:-${TMX_WANDB_RUN_ID}}" "${TMX_WANDB_ACTUAL_RUN_ID:-${TMX_WANDB_RUN_ID}}" "${TMX_WANDB_ACTUAL_GRPO_RUN_ID:-${TMX_WANDB_RUN_ID}}"; fi)
export REWARD_CORRECT='${REWARD_CORRECT}'
export REWARD_INCORRECT='${REWARD_INCORRECT}'
export NONTERMINATION_PENALTY='${NONTERMINATION_PENALTY}'
export INVALID_ANSWER_PENALTY='${INVALID_ANSWER_PENALTY}'
export SOFT_OVERLONG_EXPECTED_LEN='${SOFT_OVERLONG_EXPECTED_LEN}'
export SOFT_OVERLONG_CACHE_LEN='${SOFT_OVERLONG_CACHE_LEN}'
export NOVELTY_ZSCORE='${NOVELTY_ZSCORE}'
export CORRECTNESS_ZSCORE='${CORRECTNESS_ZSCORE}'
export DISTILL_EPOCHS='${DISTILL_EPOCHS}'
export DISTILL_BATCH_SIZE='${DISTILL_BATCH_SIZE}'
export MAX_DISTILL_EXAMPLES='${MAX_DISTILL_EXAMPLES}'
export TMX_ALLOW_LONG_COMPLETION_TRAIN='${TMX_ALLOW_LONG_COMPLETION_TRAIN}'
export TMX_SKIP_FINAL_EVAL='${TMX_SKIP_FINAL_EVAL}'
export TMX_AIME_PROBE_EVERY_STEPS='${TMX_AIME_PROBE_EVERY_STEPS}'
export TMX_AIME_PROBE_NUM_PROBLEMS='${TMX_AIME_PROBE_NUM_PROBLEMS}'
export TMX_AIME_PROBE_NUM_ROLLOUTS='${TMX_AIME_PROBE_NUM_ROLLOUTS}'
export TMX_AIME_PROBE_MAX_TOKENS='${TMX_AIME_PROBE_MAX_TOKENS}'
export TMX_AIME_PROBE_CONCURRENCY='${TMX_AIME_PROBE_CONCURRENCY}'
export TMX_SAVE_BASE_POLICY_TRAJECTORIES='${TMX_SAVE_BASE_POLICY_TRAJECTORIES}'
export TMX_BASE_POLICY_MAX_COMPLETION_LEN='${TMX_BASE_POLICY_MAX_COMPLETION_LEN}'
export TMX_EFFECTIVE_PROMPTS_PER_STEP='${TMX_EFFECTIVE_PROMPTS_PER_STEP}'
export TMX_REPLICATE_ROLLOUT_BATCH='${TMX_REPLICATE_ROLLOUT_BATCH:-0}'
export TMX_BATCH_SHARD_SOURCE_HOST='${TMX_BATCH_SHARD_SOURCE_HOST}'
export TMX_BATCH_SHARD_HTTP_PORT='${TMX_BATCH_SHARD_HTTP_PORT}'
export TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS='${TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS}'
export TMX_BATCH_SHARD_HTTP_ATTEMPTS='${TMX_BATCH_SHARD_HTTP_ATTEMPTS}'
export TMX_BATCH_SHARD_HTTP_TIMEOUT='${TMX_BATCH_SHARD_HTTP_TIMEOUT}'
export MIN_ACCEPTED_TRAJECTORIES='${MIN_ACCEPTED_TRAJECTORIES}'
export MAX_ACCEPTED_TRAJECTORIES='${MAX_ACCEPTED_TRAJECTORIES}'
export ACCEPTED_SELECTION_POLICY='${ACCEPTED_SELECTION_POLICY}'
export TMX_PIPELINE_STAGE='${TMX_PIPELINE_STAGE:-explorer_novelty}'
export TMX_PIPELINE_STOP_AFTER_EXPLORER='${TMX_PIPELINE_STOP_AFTER_EXPLORER:-0}'
export TMX_PIPELINE_STOP_AFTER_SFT='${TMX_PIPELINE_STOP_AFTER_SFT:-0}'
export TMX_PIPELINE_SKIP_EXPLORER='${TMX_PIPELINE_SKIP_EXPLORER:-0}'
export TMX_EXISTING_TRAJECTORIES='${TMX_EXISTING_TRAJECTORIES:-}'
export TMX_VLLM_RANDOMIZE_SEEDS='${TMX_VLLM_RANDOMIZE_SEEDS:-0}'
export TMX_VLLM_SEED_MODE='${TMX_VLLM_SEED_MODE:-api}'
export TMX_VLLM_SEED_API_FALLBACK='${TMX_VLLM_SEED_API_FALLBACK:-1}'
export TMX_VLLM_SEED_BASE='${TMX_VLLM_SEED_BASE:-}'
export TMX_VLLM_FILTER_DEAD_SERVERS='${TMX_VLLM_FILTER_DEAD_SERVERS:-1}'
export TMX_VLLM_HEALTH_FILTER_TIMEOUT='${TMX_VLLM_HEALTH_FILTER_TIMEOUT:-2}'
export TMX_VLLM_HEALTH_FILTER_TTL_SECONDS='${TMX_VLLM_HEALTH_FILTER_TTL_SECONDS:-30}'
export TMX_VLLM_HEALTH_FILTER_CONCURRENCY='${TMX_VLLM_HEALTH_FILTER_CONCURRENCY:-64}'
export TMX_VLLM_FAILOVER_ROUNDS='${TMX_VLLM_FAILOVER_ROUNDS:-2}'
export TMX_VLLM_FAILOVER_ROUND_SLEEP='${TMX_VLLM_FAILOVER_ROUND_SLEEP:-10}'
export TMX_VLLM_FANOUT_DESYNC='${TMX_VLLM_FANOUT_DESYNC:-0}'
export TMX_VLLM_FANOUT_NONCE='${TMX_VLLM_FANOUT_NONCE:-0}'
export TMX_VLLM_FANOUT_NONCE_MODE='${TMX_VLLM_FANOUT_NONCE_MODE:-instruction}'
export TMX_VLLM_FANOUT_NONCE_TEMPLATE='${TMX_VLLM_FANOUT_NONCE_TEMPLATE:-}'
bash tmx_jax/scripts/launch_jax_pipeline.sh
INNER
  chmod +x "$worker_script"
  if [ "$TMX_FAST_WORKER_SCRIPT_FANOUT" != "1" ]; then
    gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$w" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
      --command="mkdir -p '$REMOTE_REPO'" >/dev/null
    if [ "$SYNC_LOCAL_PACKAGE" = "1" ] && [ "$PACKAGE_SYNC_DONE" != "1" ]; then
      remote_archive="/tmp/${RUN_NAME}_tmx_jax_payload.tar.gz"
      gcloud alpha compute tpus tpu-vm scp "$PACKAGE_ARCHIVE" "$TPU_NAME:${remote_archive}" \
        --worker="$w" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS >/dev/null
      gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$w" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
        --command="rm -rf '$REMOTE_REPO/tmx_jax' && mkdir -p '$REMOTE_REPO' && tar -xzf '$remote_archive' -C '$REMOTE_REPO'" >/dev/null
    fi
    gcloud alpha compute tpus tpu-vm scp "$worker_script" "$TPU_NAME:/tmp/${RUN_NAME}_w${w}.sh" \
      --worker="$w" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS >/dev/null
  fi
done

if [ "$TMX_FAST_WORKER_SCRIPT_FANOUT" = "1" ]; then
  script_bundle="$LOCAL_TMP_DIR/${RUN_NAME}_worker_scripts.tar.gz"
  COPYFILE_DISABLE=1 tar -czf "$script_bundle" -C "$LOCAL_TMP_DIR" $(printf "${RUN_NAME}_w%s.sh " "${train_workers[@]}")
  script_bundle_name="${RUN_NAME}_worker_scripts.tar.gz"
  remote_script_bundle="/tmp/${script_bundle_name}"
  echo "[sync] uploading worker-script bundle once to worker ${coord_worker}"
  gcloud alpha compute tpus tpu-vm scp "$script_bundle" "$TPU_NAME:${remote_script_bundle}" \
    --worker="$coord_worker" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS >/dev/null
  echo "[sync] starting worker-script fanout server on worker ${coord_worker}:${TMX_WORKER_SCRIPT_FANOUT_PORT}"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$coord_worker" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
    --command="
      set -euo pipefail
      pkill -f '[h]ttp.server ${TMX_WORKER_SCRIPT_FANOUT_PORT}' 2>/dev/null || true
      cd /tmp
      nohup python3 -m http.server '${TMX_WORKER_SCRIPT_FANOUT_PORT}' --bind 0.0.0.0 \
        >/tmp/tmx_worker_script_http_${TMX_WORKER_SCRIPT_FANOUT_PORT}.log 2>&1 &
    " >/dev/null
  echo "[sync] fanning out worker scripts internally to ${#train_workers[@]} train workers"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker=all --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
    --command="
      set -euo pipefail
      worker_id=\"\$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number 2>/dev/null || echo \${TPU_WORKER_ID:-0})\"
      case ' ${TRAIN_WORKERS} ' in
        *' '\${worker_id}' '*) ;;
        *) exit 0 ;;
      esac
      mkdir -p '${REMOTE_REPO}'
      url='http://${coord_ip}:${TMX_WORKER_SCRIPT_FANOUT_PORT}/${script_bundle_name}'
      dst='/tmp/${script_bundle_name}'
      python3 - \"\$url\" \"\$dst\" <<'PY'
import sys
import urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
      tar -xzf \"\$dst\" -C /tmp
      chmod +x /tmp/${RUN_NAME}_w*.sh
    " >/dev/null
fi

launch_train_worker() {
  local idx="$1"
  local w="$2"
  launched=0
  for attempt in $(seq 1 "${TMX_TRAIN_LAUNCH_RETRIES:-5}"); do
    if gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" --worker="$w" --zone="$ZONE" $GCLOUD_TPU_IAP_FLAGS \
      --ssh-flag="-o ConnectTimeout=${TMX_TRAIN_LAUNCH_CONNECT_TIMEOUT:-30}" \
      --ssh-flag="-o ServerAliveInterval=${TMX_TRAIN_LAUNCH_SERVER_ALIVE_INTERVAL:-10}" \
      --ssh-flag="-o ServerAliveCountMax=${TMX_TRAIN_LAUNCH_SERVER_ALIVE_COUNT_MAX:-3}" \
      --command="
    set -euo pipefail
    RUN='$RUN_NAME'
    pkill -9 -f '[p]ython -m tmx_jax' 2>/dev/null || true
    pkill -9 -f '[V]LLM::EngineCore|[v]llm.entrypoints.openai.api_server|[v]llm serve' 2>/dev/null || true
    (tmux ls 2>/dev/null || true) | while IFS=: read -r s rest; do
      case \"\$s\" in *\"\$RUN\"*) tmux kill-session -t \"\$s\" 2>/dev/null || true ;; esac
    done
    mkdir -p ~/runs/$RUN_NAME
    chmod +x /tmp/${RUN_NAME}_w${w}.sh
    worker_log=~/runs/$RUN_NAME/worker_w${w}.log
    echo \"[launch] starting worker ${w} attempt ${attempt} at \$(date -Is)\" >> \"\$worker_log\"
    tmux_ok=0
    if command -v tmux >/dev/null 2>&1; then
      # Some TPU images leave a stale tmux socket and the server exits before
      # listing sessions. Clear only this user's socket, then fall back to
      # nohup if tmux is still unavailable.
      rm -rf /tmp/tmux-\$(id -u) 2>/dev/null || true
      if tmux new-session -d -s '${RUN_NAME}_w${w}' \
        \"bash -lc 'bash /tmp/${RUN_NAME}_w${w}.sh; rc=\\\$?; echo [launch] worker ${w} exited rc=\\\$rc at \\\$(date -Is); exit \\\$rc' >> '\$worker_log' 2>&1\" \
        2>~/runs/$RUN_NAME/tmux_start_w${w}.err; then
        if tmux has-session -t '${RUN_NAME}_w${w}' 2>/dev/null; then
          tmux_ok=1
        fi
      fi
    fi
    if [ \${tmux_ok:-0} != 1 ]; then
      echo '[launch] tmux unavailable; falling back to nohup for worker ${w}' \
        | tee -a ~/runs/$RUN_NAME/launcher_w${w}.log
      nohup bash -lc 'bash /tmp/${RUN_NAME}_w${w}.sh; rc=\$?; echo [launch] worker ${w} exited rc=\$rc at \$(date -Is); exit \$rc' >>\"\$worker_log\" 2>&1 < /dev/null &
      echo \$! > ~/runs/$RUN_NAME/pid_w${w}.pid
    fi
    sleep \"${TMX_TRAIN_LAUNCH_STABILITY_SECONDS:-12}\"
    if ! tmux has-session -t '${RUN_NAME}_w${w}' 2>/dev/null \
      && ! pgrep -af 'bash /tmp/${RUN_NAME}_w${w}.sh|[p]ython -m tmx_jax' >/dev/null 2>&1; then
      echo '[launch] ERROR: worker ${w} did not stay alive' >&2
      tail -120 \"\$worker_log\" >&2 || true
      exit 1
    fi
  "; then
      launched=1
      break
    fi
    echo "[launch] WARN: worker ${w} launch attempt ${attempt} failed; retrying" >&2
    sleep "${TMX_TRAIN_LAUNCH_RETRY_SLEEP:-10}"
  done
  if [ "$launched" != "1" ]; then
    echo "[launch] ERROR: worker ${w} failed to launch after retries" >&2
    exit 1
  fi
}

if [ "${TMX_PARALLEL_TRAIN_LAUNCH:-0}" = "1" ]; then
  echo "[launch] parallel train launch enabled workers=${#train_workers[@]} parallelism=${TMX_TRAIN_LAUNCH_PARALLELISM:-16}"
  pids=()
  labels=()
  max_parallel="${TMX_TRAIN_LAUNCH_PARALLELISM:-16}"
  for idx in "${!train_workers[@]}"; do
    w="${train_workers[$idx]}"
    launch_train_worker "$idx" "$w" &
    pids+=("$!")
    labels+=("$w")
    while [ "${#pids[@]}" -ge "$max_parallel" ]; do
      pid="${pids[0]}"
      label="${labels[0]}"
      if ! wait "$pid"; then
        echo "[launch] ERROR: worker ${label} failed during parallel launch" >&2
        exit 1
      fi
      pids=("${pids[@]:1}")
      labels=("${labels[@]:1}")
    done
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[launch] ERROR: worker ${labels[$i]} failed during parallel launch" >&2
      exit 1
    fi
  done
else
  for idx in "${!train_workers[@]}"; do
    w="${train_workers[$idx]}"
    launch_train_worker "$idx" "$w"
  done
fi

echo "RUN_NAME=$RUN_NAME"
echo "COORDINATOR=${coord_ip}:${COORD_PORT}"
echo "VLLM_URLS=$vllm_urls"
