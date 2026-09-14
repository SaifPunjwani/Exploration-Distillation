#!/usr/bin/env bash
# Create the XPK/GKE cluster for the TMX paper pipeline.
#
# Idempotent — safe to rerun. If the cluster already exists, this is a no-op.
# Network policy is locked to TRC rules: external IPs ON, no NAT, no
# internal-IP-only access.
#
# Required env:
#   PROJECT_ID       — GCP project (e.g. my-gcp-project)
#   ZONE             — zone (e.g. us-central1-a)
# Optional env:
#   CLUSTER_NAME     — default: tmx-${ACCEL_SHORT}
#   TPU_TYPE         — default: v5litepod-256
#   NUM_SLICES       — default: 1
#   USE_PATHWAYS     — default: true (pass --enable-pathways)
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
ZONE="${ZONE:?set ZONE (e.g. us-central1-a or europe-west4-b)}"
TPU_TYPE="${TPU_TYPE:-v5litepod-256}"
NUM_SLICES="${NUM_SLICES:-1}"
USE_PATHWAYS="${USE_PATHWAYS:-true}"

ACCEL_SHORT="$(echo "${TPU_TYPE}" | tr -d '-' | tr '[:upper:]' '[:lower:]')"
CLUSTER_NAME="${CLUSTER_NAME:-tmx-${ACCEL_SHORT}}"
REGION="${ZONE%-*}"

echo "[xpk-cluster] project=${PROJECT_ID}"
echo "[xpk-cluster] zone=${ZONE} region=${REGION}"
echo "[xpk-cluster] cluster=${CLUSTER_NAME} tpu=${TPU_TYPE} slices=${NUM_SLICES}"

if ! command -v xpk >/dev/null 2>&1; then
    echo "[xpk-cluster] ERROR: xpk not on PATH; install from https://github.com/AI-Hypercomputer/xpk" >&2
    exit 2
fi

# Refuse forbidden CLI flags this project bans.
for arg in "$@"; do
    case "${arg}" in
        --internal-ip|--internal-ips|--enable-private-nodes|--enable-private-endpoint)
            echo "[xpk-cluster] ERROR: refusing forbidden flag ${arg} (TRC requires external IPs)" >&2
            exit 2
            ;;
    esac
done

if gcloud --project="${PROJECT_ID}" container clusters describe "${CLUSTER_NAME}" \
        --location="${REGION}" >/dev/null 2>&1; then
    echo "[xpk-cluster] cluster ${CLUSTER_NAME} already exists in ${REGION}; skipping create"
    exit 0
fi

XPK_FLAGS=(
    --project="${PROJECT_ID}"
    --zone="${ZONE}"
    --cluster="${CLUSTER_NAME}"
    --tpu-type="${TPU_TYPE}"
    --num-slices="${NUM_SLICES}"
)
if [ "${USE_PATHWAYS}" = "true" ]; then
    XPK_FLAGS+=(--enable-pathways)
fi

echo "[xpk-cluster] xpk cluster create ${XPK_FLAGS[*]}"
xpk cluster create "${XPK_FLAGS[@]}"

echo "[xpk-cluster] OK ${CLUSTER_NAME}"
