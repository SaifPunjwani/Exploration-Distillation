#!/usr/bin/env bash
# Guard for the TRC v5litepod-256 allocation.
#
# This script never creates network resources. It only verifies that the
# existing TPU VM is READY/HEALTHY, has the expected accelerator type, and has
# external IPs enabled.
set -euo pipefail

TRC2_PROJECT="${TRC2_PROJECT:?set TRC2_PROJECT to the GCP project of the TRC allocation}"
TRC2_TPU_NAME="${TRC2_TPU_NAME:?set TRC2_TPU_NAME to the TRC TPU VM name}"
TRC2_ZONE="${TRC2_ZONE:?set TRC2_ZONE to the TRC TPU zone}"
TRC2_EXPECTED_ACCELERATOR="${TRC2_EXPECTED_ACCELERATOR:-v5litepod-256}"

refuse_internal_ip_args() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --internal-ip|--internal-ips|--internal_ip|--internal_ips)
        echo "[trc2-guard] ERROR: refusing internal-IP flag; use external IPs only." >&2
        return 2
        ;;
    esac
  done
}

assert_existing_v5lite256_external() {
  if [[ "$TRC2_TPU_NAME" != "${TRC2_EXPECTED_TPU_PREFIX:-trc2-v5lite256-}"* ]]; then
    echo "[trc2-guard] ERROR: refusing TPU $TRC2_TPU_NAME; expected ${TRC2_EXPECTED_TPU_PREFIX:-trc2-v5lite256-}*." >&2
    return 2
  fi

  local desc state health accelerator external_ips worker_count
  desc="$(
    gcloud --project="$TRC2_PROJECT" alpha compute tpus tpu-vm describe "$TRC2_TPU_NAME" \
      --zone="$TRC2_ZONE" \
      --format=json 2>/dev/null || true
  )"
  if [ -z "$desc" ]; then
    echo "[trc2-guard] ERROR: cannot describe $TRC2_PROJECT/$TRC2_ZONE/$TRC2_TPU_NAME." >&2
    return 2
  fi

  read -r state health accelerator external_ips worker_count < <(
    TRC2_TPU_DESC_JSON="$desc" python3 - <<'PY'
import json
import os

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

  if [ "$state" != "READY" ] || [ "$health" != "HEALTHY" ]; then
    echo "[trc2-guard] ERROR: $TRC2_PROJECT/$TRC2_ZONE/$TRC2_TPU_NAME is state=$state health=$health; expected READY/HEALTHY." >&2
    return 2
  fi
  if [ "$accelerator" != "$TRC2_EXPECTED_ACCELERATOR" ]; then
    echo "[trc2-guard] ERROR: $TRC2_TPU_NAME accelerator=$accelerator; expected $TRC2_EXPECTED_ACCELERATOR." >&2
    return 2
  fi
  if [ "$external_ips" != "True" ] && [ "$external_ips" != "true" ]; then
    echo "[trc2-guard] ERROR: $TRC2_TPU_NAME external IPs are not enabled." >&2
    return 2
  fi

  echo "[trc2-guard] OK $TRC2_PROJECT/$TRC2_ZONE/$TRC2_TPU_NAME $accelerator workers=$worker_count external_ips=$external_ips"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  refuse_internal_ip_args "$@"
  assert_existing_v5lite256_external
fi
