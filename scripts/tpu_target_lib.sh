#!/usr/bin/env bash

# TMX_DOCTOR_ALLOW_FORBIDDEN_NETWORK_TOKENS
# This file contains internal-IP flag names only for non-TRC legacy detection
# and explicit TRC refusal. The TRC project (TRC2_PROJECT) always uses external IP/IAP.

set -euo pipefail

ROOT_DIR_TPU_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TPU_PROJECT="${TPU_PROJECT:-}"

tmx_resolve_gcloud_bin() {
  if [ -n "${GCLOUD_BIN:-}" ] && [ -x "${GCLOUD_BIN}" ]; then
    printf '%s\n' "${GCLOUD_BIN}"
    return 0
  fi
  if command -v gcloud >/dev/null 2>&1; then
    command -v gcloud
    return 0
  fi
  if [ -x "${HOME}/Downloads/google-cloud-sdk/bin/gcloud" ]; then
    printf '%s\n' "${HOME}/Downloads/google-cloud-sdk/bin/gcloud"
    return 0
  fi
  return 1
}

tmx_gcloud() {
  local gcloud_bin
  gcloud_bin="$(tmx_resolve_gcloud_bin)"
  if [ -n "${TMX_CLOUDSDK_CONFIG:-}" ]; then
    export CLOUDSDK_CONFIG="${TMX_CLOUDSDK_CONFIG}"
    mkdir -p "${CLOUDSDK_CONFIG}"
    # Fall back to the globally authenticated gcloud profile if the repo-local
    # config is present but its tokens have expired.
    if ! "${gcloud_bin}" auth print-access-token >/dev/null 2>&1; then
      unset CLOUDSDK_CONFIG
    fi
  fi
  if [ -n "${TPU_PROJECT:-}" ]; then
    "${gcloud_bin}" --project="${TPU_PROJECT}" "$@"
  else
    "${gcloud_bin}" "$@"
  fi
}

tmx_candidate_specs() {
  # TPU_CANDIDATES="name:zone;name:zone;..." lists the TPU VMs to probe, in
  # order, when TPU_NAME/ZONE are not set. There is no built-in list.
  if [ -n "${TPU_CANDIDATES:-}" ]; then
    printf '%s\n' "${TPU_CANDIDATES}" | tr ';' '\n'
    return 0
  fi
  return 0
}

tmx_detect_tpu_ssh_mode() {
  if [ -n "${TPU_SSH_MODE:-}" ]; then
    printf '%s\n' "${TPU_SSH_MODE}"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    if curl -fsS \
      -H 'Metadata-Flavor: Google' \
      --connect-timeout 1 \
      --max-time 1 \
      http://metadata.google.internal/computeMetadata/v1/instance/id >/dev/null 2>&1
    then
      printf '%s\n' "internal"
      return 0
    fi
  fi

  printf '%s\n' "iap"
}

tmx_tpu_transport_flag() {
  # TRC2_PROJECT (the TPU Research Cloud project id) enables the TRC refusal.
  if [ -n "${TRC2_PROJECT:-}" ] && [ "${TPU_PROJECT:-}" = "${TRC2_PROJECT}" ]; then
    case "${TPU_SSH_MODE:-iap}" in
      internal)
        echo "[trc2-guard] ERROR: refusing internal TPU SSH mode for the TRC project ${TRC2_PROJECT}; use external IPs only." >&2
        return 2
        ;;
      *)
        # Do not emit --internal-ip/--internal-ips for TRC. IAP tunneling does
        # not create NAT/router resources and keeps existing launch scripts from
        # receiving an empty transport argument.
        printf '%s\n' "--tunnel-through-iap"
        return 0
        ;;
    esac
  fi

  case "$(tmx_detect_tpu_ssh_mode)" in
    internal)
      printf '%s\n' "--internal-ip"
      ;;
    iap)
      printf '%s\n' "--tunnel-through-iap"
      ;;
    *)
      printf '%s\n' "--tunnel-through-iap"
      ;;
  esac
}

tmx_region_from_zone() {
  local zone="$1"
  printf '%s\n' "${zone%-*}"
}

tmx_require_same_region() {
  local left_label="$1"
  local left_zone="$2"
  local right_label="$3"
  local right_zone="$4"
  local allow_cross="${TMX_ALLOW_CROSS_REGION_TPU:-0}"
  local left_region
  local right_region
  left_region="$(tmx_region_from_zone "$left_zone")"
  right_region="$(tmx_region_from_zone "$right_zone")"
  if [ "$left_region" != "$right_region" ] && [ "$allow_cross" != "1" ]; then
    cat >&2 <<EOF
[cost-guard] ERROR: refusing cross-region TPU topology:
[cost-guard]   ${left_label}:  zone=${left_zone} region=${left_region}
[cost-guard]   ${right_label}: zone=${right_zone} region=${right_region}
[cost-guard] Cross-region train↔serve traffic is expensive and slow.
[cost-guard] Use TPUs in the same region, single-slice mode, or explicitly set TMX_ALLOW_CROSS_REGION_TPU=1.
EOF
    return 2
  fi
}

resolve_tpu_target_defaults() {
  if [ -n "${TPU_NAME:-}" ] && [ -n "${ZONE:-}" ]; then
    return 0
  fi

  local spec
  while IFS= read -r spec; do
    [ -n "${spec}" ] || continue
    local candidate_name="${spec%%:*}"
    local candidate_zone="${spec#*:}"
    local state=""
    local health=""
    state="$(tmx_gcloud alpha compute tpus tpu-vm describe "${candidate_name}" --zone "${candidate_zone}" --format='value(state)' 2>/dev/null || true)"
    health="$(tmx_gcloud alpha compute tpus tpu-vm describe "${candidate_name}" --zone "${candidate_zone}" --format='value(health)' 2>/dev/null || true)"
    if [ "${state}" = "READY" ] && { [ -z "${health}" ] || [ "${health}" = "HEALTHY" ]; }; then
      export TPU_NAME="${candidate_name}"
      export ZONE="${candidate_zone}"
      return 0
    fi
  done < <(tmx_candidate_specs)

  export TPU_NAME="${TPU_NAME:?set TPU_NAME (or TPU_CANDIDATES=name:zone;...) to select a TPU VM}"
  export ZONE="${ZONE:?set ZONE to the zone of TPU_NAME}"
}

resolve_tpu_worker_env() {
  if [ -n "${WORKERS:-}" ] && [ -n "${REMOTE_WORKERS:-}" ]; then
    return 0
  fi

  local describe_json=""
  describe_json="$(tmx_gcloud alpha compute tpus tpu-vm describe "${TPU_NAME}" --zone "${ZONE}" --format=json 2>/dev/null || true)"
  if [ -z "${describe_json}" ]; then
    export WORKERS="${WORKERS:-0 1 2 3}"
    export REMOTE_WORKERS="${REMOTE_WORKERS:-1-3}"
    return 0
  fi

  local worker_count=""
  worker_count="$(python3 - <<'PY' "${describe_json}"
import json
import sys

payload = json.loads(sys.argv[1])
for key in ("networkEndpoints", "workerNetworkEndpoints"):
    value = payload.get(key)
    if isinstance(value, list) and value:
        print(len(value))
        break
else:
    print(4)
PY
)"
  if [ -z "${worker_count}" ]; then
    worker_count=4
  fi

  local worker_list=""
  local idx=0
  while [ "${idx}" -lt "${worker_count}" ]; do
    worker_list="${worker_list}${idx} "
    idx=$((idx + 1))
  done
  worker_list="${worker_list% }"
  export WORKERS="${WORKERS:-${worker_list}}"

  if [ "${worker_count}" -le 1 ]; then
    export REMOTE_WORKERS="${REMOTE_WORKERS:-0}"
  else
    export REMOTE_WORKERS="${REMOTE_WORKERS:-1-$((worker_count - 1))}"
  fi
}

tmx_resolve_tpu_accelerator_type() {
  if [ -n "${TPU_ACCELERATOR_TYPE:-}" ]; then
    printf '%s\n' "${TPU_ACCELERATOR_TYPE}"
    return 0
  fi

  if [ -z "${TPU_NAME:-}" ] || [ -z "${ZONE:-}" ]; then
    return 1
  fi

  local accelerator_type=""
  accelerator_type="$(
    tmx_gcloud alpha compute tpus tpu-vm describe "${TPU_NAME}" \
      --zone "${ZONE}" \
      --format='value(acceleratorType)' 2>/dev/null || true
  )"
  accelerator_type="$(printf '%s' "${accelerator_type}" | tr -d '\r')"
  if [ -n "${accelerator_type}" ]; then
    printf '%s\n' "${accelerator_type}"
    return 0
  fi

  return 1
}
