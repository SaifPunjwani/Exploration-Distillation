#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh
resolve_tpu_target_defaults
resolve_tpu_worker_env

TPU_NAME="${TPU_NAME}"
ZONE="${ZONE}"
TPU_TRANSPORT_FLAG="$(tmx_tpu_transport_flag)"
BOOTSTRAP_WORKERS="${BOOTSTRAP_WORKERS:-${WORKERS:-all}}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
VENV_DIR="${VENV_DIR:-.venv_tpu_slice}"
SKIP_SYNC="${SKIP_SYNC:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
RESET_VENV="${RESET_VENV:-0}"
SYNC_WORKERS="${SYNC_WORKERS:-$BOOTSTRAP_WORKERS}"
LOCAL_WHEELHOUSE_TAR="${LOCAL_WHEELHOUSE_TAR:-}"
LOCAL_HF_TAR="${LOCAL_HF_TAR:-}"
GCS_WHEELHOUSE_OBJECT="${GCS_WHEELHOUSE_OBJECT:-}"
GCS_HF_OBJECT="${GCS_HF_OBJECT:-}"
DIRECT_SSH_HOST="${DIRECT_SSH_HOST:-}"
GCS_ASSET_PREFIX="${GCS_ASSET_PREFIX:-}"
TORCH_VERSION="${TORCH_VERSION:-}"
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-}"
LIBTPU_VERSION="${LIBTPU_VERSION:-}"
SKIP_TPU_VALIDATE="${SKIP_TPU_VALIDATE:-0}"

ensure_gcloud_ssh_key() {
  local key_path="${HOME}/.ssh/google_compute_engine"
  if [ ! -f "$key_path" ]; then
    return
  fi
  if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l >/dev/null 2>&1; then
    eval "$(ssh-agent -s)" >/dev/null
  fi
  ssh-add "$key_path" >/dev/null 2>&1 || true
}

ensure_gcloud_ssh_key

if [ "$SKIP_SYNC" != "1" ]; then
  DIRECT_SSH_HOST="$DIRECT_SSH_HOST" SYNC_WORKERS="$SYNC_WORKERS" bash scripts/sync_tpu_slice_repo.sh
fi

use_direct_ssh=0
if [ -n "$DIRECT_SSH_HOST" ] && [ "$BOOTSTRAP_WORKERS" = "0" ]; then
  use_direct_ssh=1
fi

run_remote_command() {
  local command="$1"
  if [ "$use_direct_ssh" = "1" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$DIRECT_SSH_HOST" "$command"
  else
    tmx_gcloud alpha compute tpus tpu-vm ssh \
      "$TPU_NAME" \
      --worker="$BOOTSTRAP_WORKERS" \
      --batch-size=all \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG" \
      --command="$command"
  fi
}

copy_to_remote() {
  local local_path="$1"
  local remote_path="$2"
  if [ "$use_direct_ssh" = "1" ]; then
    scp -o BatchMode=yes -o ConnectTimeout=20 "$local_path" "${DIRECT_SSH_HOST}:${remote_path}"
  else
    tmx_gcloud alpha compute tpus tpu-vm scp \
      "$local_path" \
      "${TPU_NAME}:${remote_path}" \
      --worker="$BOOTSTRAP_WORKERS" \
      --batch-size=all \
      --zone="$ZONE" \
      "$TPU_TRANSPORT_FLAG"
  fi
}

stage_to_gcs_once() {
  local local_path="$1"
  local object_path="$2"
  if gcloud storage ls "$object_path" >/dev/null 2>&1; then
    return
  fi
  gcloud storage cp "$local_path" "$object_path"
}

wheelhouse_object_path=""
if [ -n "$GCS_WHEELHOUSE_OBJECT" ]; then
  wheelhouse_object_path="$GCS_WHEELHOUSE_OBJECT"
elif [ -n "$GCS_ASSET_PREFIX" ]; then
  wheelhouse_name="$(basename "${LOCAL_WHEELHOUSE_TAR:-wheelhouse.tar}")"
  wheelhouse_object_path="${GCS_ASSET_PREFIX%/}/${wheelhouse_name}"
fi

hf_object_path=""
if [ -n "$GCS_HF_OBJECT" ]; then
  hf_object_path="$GCS_HF_OBJECT"
elif [ -n "$GCS_ASSET_PREFIX" ]; then
  hf_name="$(basename "${LOCAL_HF_TAR:-hf_cache.tar}")"
  hf_object_path="${GCS_ASSET_PREFIX%/}/${hf_name}"
fi

if [ -n "$LOCAL_WHEELHOUSE_TAR" ] && [ -f "$LOCAL_WHEELHOUSE_TAR" ]; then
  remote_wheelhouse_tar="/tmp/$(basename "$LOCAL_WHEELHOUSE_TAR")"
  if [ -n "$GCS_ASSET_PREFIX" ]; then
    wheelhouse_gcs_path="$wheelhouse_object_path"
    stage_to_gcs_once "$LOCAL_WHEELHOUSE_TAR" "$wheelhouse_gcs_path"
    run_remote_command "gcloud storage cp '$wheelhouse_gcs_path' '$remote_wheelhouse_tar'"
  else
    copy_to_remote "$LOCAL_WHEELHOUSE_TAR" "$remote_wheelhouse_tar"
  fi
  run_remote_command "mkdir -p '$REMOTE_ROOT/.wheelhouse' && rm -rf '$REMOTE_ROOT/.wheelhouse'/* && tar -xf '$remote_wheelhouse_tar' -C '$REMOTE_ROOT/.wheelhouse'"
elif [ -n "$wheelhouse_object_path" ]; then
  remote_wheelhouse_tar="/tmp/$(basename "$wheelhouse_object_path")"
  run_remote_command "gcloud storage cp '$wheelhouse_object_path' '$remote_wheelhouse_tar' && mkdir -p '$REMOTE_ROOT/.wheelhouse' && rm -rf '$REMOTE_ROOT/.wheelhouse'/* && tar -xf '$remote_wheelhouse_tar' -C '$REMOTE_ROOT/.wheelhouse'"
fi

if [ -n "$LOCAL_HF_TAR" ] && [ -f "$LOCAL_HF_TAR" ]; then
  remote_hf_tar="/tmp/$(basename "$LOCAL_HF_TAR")"
  if [ -n "$GCS_ASSET_PREFIX" ]; then
    hf_gcs_path="$hf_object_path"
    stage_to_gcs_once "$LOCAL_HF_TAR" "$hf_gcs_path"
    run_remote_command "gcloud storage cp '$hf_gcs_path' '$remote_hf_tar'"
  else
    copy_to_remote "$LOCAL_HF_TAR" "$remote_hf_tar"
  fi
  run_remote_command "mkdir -p '$REMOTE_ROOT/.hf' && rm -rf '$REMOTE_ROOT/.hf'/* && tar -xf '$remote_hf_tar' -C '$REMOTE_ROOT/.hf'"
elif [ -n "$hf_object_path" ]; then
  remote_hf_tar="/tmp/$(basename "$hf_object_path")"
  run_remote_command "gcloud storage cp '$hf_object_path' '$remote_hf_tar' && mkdir -p '$REMOTE_ROOT/.hf' && rm -rf '$REMOTE_ROOT/.hf'/* && tar -xf '$remote_hf_tar' -C '$REMOTE_ROOT/.hf'"
fi

run_remote_command "cd '$REMOTE_ROOT' && if [ '$RESET_VENV' = '1' ]; then rm -rf '$VENV_DIR'; fi && VENV_DIR='$VENV_DIR' USE_TPU=1 SKIP_INSTALL='$SKIP_INSTALL' TORCH_VERSION='$TORCH_VERSION' TORCH_XLA_VERSION='$TORCH_XLA_VERSION' LIBTPU_VERSION='$LIBTPU_VERSION' WHEELHOUSE_DIR='$REMOTE_ROOT/.wheelhouse' HF_HOME='$REMOTE_ROOT/.hf' HF_HUB_CACHE='$REMOTE_ROOT/.hf/hub' HF_DATASETS_CACHE='$REMOTE_ROOT/.hf/datasets' TRANSFORMERS_CACHE='$REMOTE_ROOT/.hf/transformers' bash scripts/bootstrap_tpu_env.sh"

if [ "$SKIP_TPU_VALIDATE" != "1" ]; then
  if ! run_remote_command "cd '$REMOTE_ROOT' && . '$VENV_DIR/bin/activate' && export PJRT_DEVICE=TPU && export TMX_TPU_SLICE_LAUNCHED=1 && export TMX_REQUIRE_TPU=1 && export TMX_ALLOW_SINGLE_WORKER_TPU=1 && export TPU_SKIP_MDS_QUERY=1 && export TPU_VISIBLE_CHIPS=0 && export TPU_PROCESS_BOUNDS=1,1,1 && export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 && export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 && export TPU_HOST_BOUNDS=1,1,1 && export TPU_WORKER_HOSTNAMES=localhost && timeout 90s python -c 'from tmx.devices import resolve_device; dev, is_xla, label = resolve_device(\"tpu\"); print(\"resolved\", dev, is_xla, label)'"
  then
    echo "[WARN] Per-worker TPU validation failed on one or more workers. Continuing because per-chip launches may still succeed."
  fi
else
  echo "[INFO] SKIP_TPU_VALIDATE=1; skipping bootstrap TPU validation."
fi

echo "TPU slice bootstrap complete for $TPU_NAME ($ZONE)"
