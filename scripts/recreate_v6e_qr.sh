#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <queued-resource-name> <zone> [runtime-version]"
  exit 1
fi

QR_NAME="$1"
ZONE="$2"
RUNTIME_VERSION="${3:-v2-alpha-tpuv6e}"
PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v6e-64}"

if [ "${RUNTIME_VERSION}" = "tpu-vm-tf-2.15.0-pod-pjrt" ]; then
  echo "Refusing broken v6e runtime: ${RUNTIME_VERSION}" >&2
  echo "Use v2-alpha-tpuv6e for v6e queued resources." >&2
  exit 2
fi

node_state="$(
  gcloud alpha compute tpus tpu-vm describe "$QR_NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --format='value(state)' 2>/dev/null || true
)"
qr_state="$(
  gcloud alpha compute tpus queued-resources describe "$QR_NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --format='value(state.state)' 2>/dev/null || true
)"

printf '[%s] node_state=%s qr_state=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${node_state:-NONE}" \
  "${qr_state:-NONE}"

if [ -z "${node_state}" ] && [ -n "${qr_state}" ] && [ "${qr_state}" != "ACTIVE" ]; then
  gcloud alpha compute tpus queued-resources delete "$QR_NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --quiet || true
  sleep 10
  qr_state="$(
    gcloud alpha compute tpus queued-resources describe "$QR_NAME" \
      --project "$PROJECT" \
      --zone "$ZONE" \
      --format='value(state.state)' 2>/dev/null || true
  )"
  printf '[%s] post-delete qr_state=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${qr_state:-NONE}"
fi

if [ -z "${qr_state}" ]; then
  gcloud alpha compute tpus queued-resources create "$QR_NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --runtime-version="$RUNTIME_VERSION" \
    --node-id="$QR_NAME" \
    --provisioning-model=SPOT \
    --internal-ips \
    --network=default \
    --subnetwork=default
fi
