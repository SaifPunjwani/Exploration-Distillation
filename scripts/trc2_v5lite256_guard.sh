#!/usr/bin/env bash
# Safety guard for existing TPU allocations.
#
# TMX_DOCTOR_ALLOW_FORBIDDEN_NETWORK_TOKENS
#
# The trc2 project must use nanny-created v5litepod-256 TPUs with external IPs
# enabled. Non-TRC projects are allowed, but this guard still refuses internal
# SSH flags and verifies that the target TPU exists, is healthy, and matches the
# requested accelerator shape.
set -euo pipefail

TRC2_PROJECT="${TRC2_PROJECT:?set TRC2_PROJECT to the GCP project id of the TRC allocation}"
TRC2_TPU_NAME="${TRC2_TPU_NAME:-trc2-v5lite256-uc1a-r1}"
TRC2_ZONE="${TRC2_ZONE:-us-central1-a}"
TRC2_EXPECTED_ACCELERATOR="${TRC2_EXPECTED_ACCELERATOR:-v5litepod-256}"
# TMX_TRC_PROJECT names the TPU Research Cloud project; the TRC-only checks
# (nanny-created v5litepod-256 names, external IPs) apply when TRC2_PROJECT
# equals it. Other projects get the generic existence/health/shape checks.
TRC_PROJECT_ID="${TMX_TRC_PROJECT:-}"
trc2_guard_is_trc() {
  [ -n "${TRC_PROJECT_ID}" ] && [ "${TRC2_PROJECT}" = "${TRC_PROJECT_ID}" ]
}

trc2_guard_gcloud() {
  gcloud --project="${TRC2_PROJECT}" "$@"
}

trc2_guard_refuse_internal_ip_args() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --internal-ip|--internal-ips|--internal_ip|--internal_ips)
        echo "[trc2-guard] ERROR: refusing internal-IP flag for ${TRC2_PROJECT}; use external IPs only." >&2
        return 2
        ;;
    esac
  done
}

trc2_guard_assert_existing_v5lite256_external() {
  if trc2_guard_is_trc && [[ "${TRC2_TPU_NAME}" != trc2-v5lite256-* ]]; then
    echo "[trc2-guard] ERROR: refusing TPU ${TRC2_TPU_NAME}; expected trc2-v5lite256-*." >&2
    return 2
  fi

  local desc state health accelerator external_ips worker_count
  desc="$(
    trc2_guard_gcloud alpha compute tpus tpu-vm describe "${TRC2_TPU_NAME}" \
      --zone="${TRC2_ZONE}" \
      --format=json 2>/dev/null || true
  )"
  if [ -z "${desc}" ]; then
    echo "[trc2-guard] ERROR: cannot describe ${TRC2_TPU_NAME} in ${TRC2_ZONE}." >&2
    return 2
  fi

  read -r state health accelerator external_ips worker_count < <(
    TRC2_TPU_DESC_JSON="${desc}" python3 - <<'PY'
import json
import os
import sys

raw = json.loads(os.environ["TRC2_TPU_DESC_JSON"])
state = raw.get("state") or "UNKNOWN"
health = raw.get("health") or "UNKNOWN"
accelerator = raw.get("acceleratorType") or "UNKNOWN"
network_config = raw.get("networkConfig") or {}
external = network_config.get("enableExternalIps")
endpoints = raw.get("networkEndpoints") or []
if external is None:
    access_configs = [endpoint.get("accessConfig") for endpoint in endpoints]
    if endpoints and all(value is not None for value in access_configs):
        external = True
    elif endpoints and any(value is not None for value in access_configs):
        external = "UNKNOWN"
    else:
        external = False
print(state, health, accelerator, external, len(endpoints))
PY
  )
  if [ "${state}" != "READY" ] || [ "${health}" != "HEALTHY" ]; then
    echo "[tpu-guard] ERROR: ${TRC2_PROJECT}/${TRC2_ZONE}/${TRC2_TPU_NAME} is state=${state} health=${health}; expected READY/HEALTHY." >&2
    return 2
  fi
  if [ "${accelerator}" != "${TRC2_EXPECTED_ACCELERATOR}" ]; then
    echo "[tpu-guard] ERROR: ${TRC2_TPU_NAME} accelerator=${accelerator}; expected ${TRC2_EXPECTED_ACCELERATOR}." >&2
    return 2
  fi
  if trc2_guard_is_trc && [ "${external_ips}" != "True" ] && [ "${external_ips}" != "true" ]; then
    echo "[trc2-guard] ERROR: ${TRC2_TPU_NAME} external IPs are not enabled." >&2
    return 2
  fi

  echo "[tpu-guard] OK ${TRC2_PROJECT}/${TRC2_ZONE}/${TRC2_TPU_NAME} ${accelerator} workers=${worker_count} external_ips=${external_ips}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  trc2_guard_refuse_internal_ip_args "$@"
  trc2_guard_assert_existing_v5lite256_external
fi
