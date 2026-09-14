#!/usr/bin/env bash
# Restart the JAX run with the DAPO algorithmic contract, keeping DR-GRPO as the
# loss normalizer and using separate full TPU slices for training and rollout.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMX_JAX_REPO="${TMX_JAX_REPO:-$(cd "$ROOT_DIR/../tmx-jax" && pwd)}"

if [ -z "${CLOUDSDK_CONFIG:-}" ]; then
  export CLOUDSDK_CONFIG="/tmp/tmx_gcloud_config_${USER:-user}_lite"
  if [ ! -f "$CLOUDSDK_CONFIG/.tmx_copied" ] && [ -d "$HOME/.config/gcloud" ]; then
    mkdir -p "$CLOUDSDK_CONFIG"
    for item in active_config config_sentinel credentials.db access_tokens.db adc.json application_default_credentials.json; do
      [ -e "$HOME/.config/gcloud/$item" ] && cp "$HOME/.config/gcloud/$item" "$CLOUDSDK_CONFIG/$item" 2>/dev/null || true
    done
    for item in configurations legacy_credentials; do
      [ -d "$HOME/.config/gcloud/$item" ] && cp -R "$HOME/.config/gcloud/$item" "$CLOUDSDK_CONFIG/$item" 2>/dev/null || true
    done
    touch "$CLOUDSDK_CONFIG/.tmx_copied" 2>/dev/null || true
  fi
  mkdir -p "$CLOUDSDK_CONFIG"
fi

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "[dapo-restart] ERROR: set $name" >&2
    exit 2
  fi
}

bool_enabled() {
  case "${1:-0}" in
    1|true|True|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

TMX_GCLOUD_TPU_USE_IAP="${TMX_GCLOUD_TPU_USE_IAP:-1}"
GCLOUD_TPU_IAP_FLAGS=""
if bool_enabled "$TMX_GCLOUD_TPU_USE_IAP"; then
  GCLOUD_TPU_IAP_FLAGS="--tunnel-through-iap"
fi

patch_once() {
  local patch_file="$1"
  local label="$2"
  if [ ! -f "$patch_file" ]; then
    echo "[dapo-restart] ERROR: missing patch $patch_file" >&2
    exit 2
  fi
  if git -C "$TMX_JAX_REPO" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "[dapo-restart] $label already applied"
  else
    if git -C "$TMX_JAX_REPO" apply --check "$patch_file" >/dev/null 2>&1; then
      echo "[dapo-restart] applying $label"
      git -C "$TMX_JAX_REPO" apply "$patch_file"
    else
      echo "[dapo-restart] WARN: $label does not apply cleanly; assuming already applied or superseded by later local patches"
    fi
  fi
}

join_by_comma() {
  local first=1
  for item in "$@"; do
    if [ "$first" = "1" ]; then
      printf '%s' "$item"
      first=0
    else
      printf ',%s' "$item"
    fi
  done
}

worker_ips() {
  local tpu_name="$1"
  local zone="$2"
  local project="$3"
  gcloud --project="$project" alpha compute tpus tpu-vm describe "$tpu_name" \
    --zone "$zone" \
    --format='get(networkEndpoints[].ipAddress)'
}

project_for_tpu() {
  # The GCP project is not inferred from the TPU name: set TRAIN_TPU_PROJECT /
  # SERVE_TPU_PROJECT explicitly, or rely on the active gcloud project.
  local tpu_name="$1"
  printf '%s
' "${CLOUDSDK_CORE_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
}

tpu_region_from_zone() {
  local zone="$1"
  printf '%s\n' "${zone%-*}"
}

require_same_region_tpus() {
  local train_region
  local serve_region
  train_region="$(tpu_region_from_zone "$TRAIN_ZONE")"
  serve_region="$(tpu_region_from_zone "$SERVE_ZONE")"
  if [ "$train_region" != "$serve_region" ] && [ "${TMX_ALLOW_CROSS_REGION_TPU:-0}" != "1" ]; then
    cat >&2 <<EOF
[cost-guard] ERROR: refusing cross-region TPU topology:
[cost-guard]   train: ${TRAIN_TPU_NAME} zone=${TRAIN_ZONE} region=${train_region}
[cost-guard]   serve: ${SERVE_TPU_NAME} zone=${SERVE_ZONE} region=${serve_region}
[cost-guard] Cross-region train↔serve traffic is expensive and slow.
[cost-guard] Use same-region TPUs or explicitly set TMX_ALLOW_CROSS_REGION_TPU=1.
EOF
    exit 2
  fi
}

wait_worker_health() {
  local tpu_name="$1"
  local zone="$2"
  local project="$3"
  local workers="$4"
  if bool_enabled "${TMX_PARALLEL_VLLM_HEALTH:-0}"; then
    local deadline tmp_dir pids worker health_cmd fail_count healthy min_healthy healthy_count
    local pending next_pending new_ok
    deadline=$(( $(date +%s) + ${VLLM_HEALTH_TIMEOUT_SECONDS:-1800} ))
    pending="$workers"
    healthy=""
    LAST_HEALTHY_WORKERS=""
    min_healthy="${MIN_HEALTHY_SERVE_WORKERS:-2}"
    while [ -n "$pending" ] && [ "$(date +%s)" -lt "$deadline" ]; do
      tmp_dir="$(mktemp -d /tmp/tmx_vllm_health.XXXXXX)"
      pids=""
      for worker in $pending; do
        if bool_enabled "${TMX_VLLM_MULTI_SERVER:-0}"; then
          health_cmd="set -e; for chip in ${TMX_VLLM_MULTI_SERVER_CHIPS:-0 1 2 3}; do port=\$(( ${TMX_VLLM_MULTI_SERVER_PORT_BASE:-8000} + chip )); curl -fsS --max-time 3 http://127.0.0.1:\${port}/health >/dev/null; done"
        else
          health_cmd="curl -fsS --max-time 3 http://127.0.0.1:${PORT:-8000}/health >/dev/null"
        fi
        (
          if gcloud --project="$project" alpha compute tpus tpu-vm ssh "$tpu_name" \
            --worker="$worker" \
            --zone="$zone" \
            $GCLOUD_TPU_IAP_FLAGS \
            --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
            --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
            --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
            --command="$health_cmd" \
            >/dev/null 2>&1; then
            echo "$worker" >"$tmp_dir/$worker.ok"
          else
            echo "$worker" >"$tmp_dir/$worker.fail"
          fi
        ) &
        pids="$pids $!"
      done
      fail_count=0
      for pid in $pids; do
        wait "$pid" || fail_count=$((fail_count + 1))
      done
      new_ok="$(find "$tmp_dir" -name '*.ok' -exec cat {} \; 2>/dev/null || true)"
      healthy="$(printf '%s\n%s\n' "$healthy" "$new_ok" | tr ' ' '\n' | sed '/^$/d' | sort -n | uniq | xargs || true)"
      next_pending=""
      for worker in $workers; do
        if printf ' %s ' "$healthy" | grep -q " $worker "; then
          :
        else
          next_pending="$next_pending $worker"
        fi
      done
      next_pending="$(printf '%s\n' "$next_pending" | xargs || true)"
      rm -rf "$tmp_dir"
      healthy_count="$(printf '%s\n' "$healthy" | wc -w | tr -d ' ')"
      if [ "$healthy_count" -ge "$min_healthy" ]; then
        LAST_HEALTHY_WORKERS="$healthy"
        echo "[dapo-restart] parallel vLLM health met: workers=[$healthy] count=$healthy_count/$min_healthy"
        return 0
      fi
      pending="$next_pending"
      [ -z "$pending" ] && break
      echo "[dapo-restart] waiting for parallel vLLM health count=$healthy_count/$min_healthy pending=[$pending]"
      sleep "${VLLM_HEALTH_POLL_SECONDS:-10}"
    done
    echo "[dapo-restart] WARN: parallel vLLM health insufficient: workers=[$healthy] count=$healthy_count/$min_healthy" >&2
    return 1
  fi
  local deadline=$(( $(date +%s) + ${VLLM_HEALTH_TIMEOUT_SECONDS:-1800} ))
  local pending="$workers"
  local healthy=""
  LAST_HEALTHY_WORKERS=""
  while [ -n "$pending" ] && [ "$(date +%s)" -lt "$deadline" ]; do
    local next_pending=""
    for worker in $pending; do
      local health_cmd
      if bool_enabled "${TMX_VLLM_MULTI_SERVER:-0}"; then
        health_cmd="set -e; for chip in ${TMX_VLLM_MULTI_SERVER_CHIPS:-0 1 2 3}; do port=\$(( ${TMX_VLLM_MULTI_SERVER_PORT_BASE:-8000} + chip )); curl -fsS --max-time 3 http://127.0.0.1:\${port}/health >/dev/null; done"
      else
        health_cmd="curl -fsS --max-time 3 http://127.0.0.1:${PORT:-8000}/health >/dev/null"
      fi
      if gcloud --project="$project" alpha compute tpus tpu-vm ssh "$tpu_name" \
        --worker="$worker" \
        --zone="$zone" \
        $GCLOUD_TPU_IAP_FLAGS \
        --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
        --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
        --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
        --command="$health_cmd" \
        >/dev/null 2>&1; then
        echo "[dapo-restart] vLLM worker $worker healthy"
        healthy="$healthy $worker"
      else
        next_pending="$next_pending $worker"
      fi
    done
    healthy="$(printf '%s\n' "$healthy" | xargs || true)"
    if bool_enabled "${ALLOW_DEGRADED_SERVE:-0}"; then
      local min_healthy="${MIN_HEALTHY_SERVE_WORKERS:-2}"
      local healthy_count
      healthy_count="$(printf '%s\n' "$healthy" | wc -w | tr -d ' ')"
      if [ "$healthy_count" -ge "$min_healthy" ]; then
        LAST_HEALTHY_WORKERS="$healthy"
        echo "[dapo-restart] minimum vLLM health met: workers=[$healthy] count=$healthy_count/$min_healthy"
        return 0
      fi
    fi
    pending="$(printf '%s\n' "$next_pending" | xargs || true)"
    [ -z "$pending" ] && break
    echo "[dapo-restart] waiting for vLLM workers:$pending"
    sleep "${VLLM_HEALTH_POLL_SECONDS:-10}"
  done
  if [ -n "$pending" ]; then
    echo "[dapo-restart] WARN: vLLM workers did not become healthy:$pending" >&2
    return 1
  fi
  LAST_HEALTHY_WORKERS="$workers"
  return 0
}

healthy_worker_list() {
  local tpu_name="$1"
  local zone="$2"
  local project="$3"
  local workers="$4"
  local healthy=""
  for worker in $workers; do
    local health_cmd
    if bool_enabled "${TMX_VLLM_MULTI_SERVER:-0}"; then
      health_cmd="set -e; for chip in ${TMX_VLLM_MULTI_SERVER_CHIPS:-0 1 2 3}; do port=\$(( ${TMX_VLLM_MULTI_SERVER_PORT_BASE:-8000} + chip )); curl -fsS --max-time 3 http://127.0.0.1:\${port}/health >/dev/null; done"
    else
      health_cmd="curl -fsS --max-time 3 http://127.0.0.1:${PORT:-8000}/health >/dev/null"
    fi
    if gcloud --project="$project" alpha compute tpus tpu-vm ssh "$tpu_name" \
      --worker="$worker" \
      --zone="$zone" \
      $GCLOUD_TPU_IAP_FLAGS \
      --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
      --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
      --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
      --command="$health_cmd" \
      >/dev/null 2>&1; then
      healthy="$healthy $worker"
    fi
  done
  printf '%s\n' "$healthy" | xargs || true
}

verify_train_side_completions() {
  local train_tpu_name="$1"
  local train_zone="$2"
  local train_project="$3"
  local worker="$4"
  local server_urls="$5"
  local model_name="$6"
  if ! bool_enabled "${TMX_VLLM_COMPLETION_PROBE:-1}"; then
    return 0
  fi
  if [ -z "$server_urls" ]; then
    echo "[dapo-restart] ERROR: completion probe has empty server_urls" >&2
    return 1
  fi
  local escaped_urls escaped_model probe_timeout probe_attempts probe_retry_s probe_max_endpoints
  escaped_urls="$(printf '%q' "$server_urls")"
  escaped_model="$(printf '%q' "$model_name")"
  probe_timeout="${TMX_VLLM_COMPLETION_PROBE_TIMEOUT_SECONDS:-20}"
  probe_attempts="${TMX_VLLM_COMPLETION_PROBE_ATTEMPTS:-2}"
  probe_retry_s="${TMX_VLLM_COMPLETION_PROBE_RETRY_SECONDS:-3}"
  probe_max_endpoints="${TMX_VLLM_COMPLETION_PROBE_MAX_ENDPOINTS:-4}"
  echo "[dapo-restart] probing vLLM completions from train worker $worker across $(printf '%s' "$server_urls" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' ') endpoints"
  gcloud --project="$train_project" alpha compute tpus tpu-vm ssh "$train_tpu_name" \
    --worker="$worker" \
    --zone="$train_zone" \
    $GCLOUD_TPU_IAP_FLAGS \
    --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
    --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
    --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
    --command="SERVER_URLS=$escaped_urls SERVED_MODEL_NAME=$escaped_model PROBE_TIMEOUT=$probe_timeout PROBE_ATTEMPTS=$probe_attempts PROBE_RETRY_SECONDS=$probe_retry_s PROBE_MAX_ENDPOINTS=$probe_max_endpoints python3 - <<'PY'
import json
import os
import sys
import time
import urllib.request

urls = [u.strip() for u in os.environ.get('SERVER_URLS', '').split(',') if u.strip()]
model = os.environ.get('SERVED_MODEL_NAME') or 'Qwen/Qwen3-1.7B'
timeout = float(os.environ.get('PROBE_TIMEOUT') or 240)
attempts = max(1, int(os.environ.get('PROBE_ATTEMPTS') or '2'))
sleep_s = float(os.environ.get('PROBE_RETRY_SECONDS') or '3')
max_endpoints = int(os.environ.get('PROBE_MAX_ENDPOINTS') or '0')
if max_endpoints > 0:
    urls = urls[:max_endpoints]
payload = json.dumps({
    'model': model,
    'prompt': '1+1=',
    'max_tokens': 1,
    'temperature': 0.0,
    'stream': False,
}).encode()

def probe(url):
    endpoint = url.rstrip('/') + '/completions'
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    choices = data.get('choices') or []
    if not choices:
        raise RuntimeError('no choices')
    return url

remaining = list(urls)
last_errors = {}
for attempt in range(1, attempts + 1):
    bad = []
    for url in remaining:
        try:
            probe(url)
        except Exception as exc:
            last_errors[url] = f'{type(exc).__name__}: {exc}'
            bad.append(url)
    if not bad:
        print(f'completion_probe_ok endpoints={len(urls)} attempts={attempt}')
        break
    remaining = bad
    print(
        f'completion_probe_retry attempt={attempt}/{attempts} bad={len(bad)}',
        file=sys.stderr,
        flush=True,
    )
    if attempt < attempts:
        time.sleep(sleep_s)
else:
    for url in remaining:
        print(url + ': ' + (last_errors.get(url) or 'unknown error'), file=sys.stderr)
    raise SystemExit(1)
PY"
}

set_vllm_urls_for_workers() {
  local workers="$1"
  local urls=()
  for worker in $workers; do
    if [ -z "${serve_ips[$worker]:-}" ]; then
      echo "[dapo-restart] ERROR: cannot build vLLM URL for serve worker $worker" >&2
      exit 2
    fi
    if bool_enabled "${TMX_VLLM_MULTI_SERVER:-0}"; then
      for chip in ${TMX_VLLM_MULTI_SERVER_CHIPS:-0 1 2 3}; do
        urls+=("http://${serve_ips[$worker]}:$(( ${TMX_VLLM_MULTI_SERVER_PORT_BASE:-8000} + chip ))/v1")
      done
    else
      urls+=("http://${serve_ips[$worker]}:${PORT}/v1")
    fi
  done
  export TMX_VLLM_SERVER_URLS="$(join_by_comma "${urls[@]}")"
  export VLLM_SERVER_URLS="$TMX_VLLM_SERVER_URLS"
  SERVE_WORKERS="$workers"
}

start_train_source_vllm_tunnels() {
  local train_tpu_name="$1"
  local train_zone="$2"
  local train_project="$3"
  local train_worker="$4"
  local serve_tpu_name="$5"
  local serve_zone="$6"
  local serve_project="$7"
  local serve_workers="$8"
  local serve_port="$9"
  if ! bool_enabled "${TMX_VLLM_TUNNEL_FROM_TRAIN_SOURCE:-0}"; then
    return 0
  fi
  local base_port="${TMX_VLLM_TUNNEL_BASE_PORT:-18000}"
  local proxy_base_port="${TMX_VLLM_PROXY_BASE_PORT:-28000}"
  local train_source_ip=""
  if bool_enabled "${TMX_VLLM_RANK_AGNOSTIC_PROXY:-0}"; then
    local train_ips_raw train_ips=()
    train_ips_raw="$(worker_ips "$train_tpu_name" "$train_zone" "$train_project")"
    while IFS= read -r ip; do
      [ -n "$ip" ] && train_ips+=("$ip")
    done < <(printf '%s\n' "$train_ips_raw" | tr ';' '\n' | tr ' ' '\n' | sed '/^$/d')
    train_source_ip="${train_ips[$train_worker]:-}"
    if [ -z "$train_source_ip" ]; then
      echo "[dapo-restart] ERROR: rank-agnostic vLLM proxy requested but cannot resolve train worker $train_worker IP" >&2
      exit 2
    fi
  fi
  local urls=()
  local specs=()
  local idx=0
  for serve_worker in $serve_workers; do
    local local_port=$((base_port + idx))
    local proxy_port=$((proxy_base_port + idx))
    specs+=("${serve_worker}:${local_port}:${proxy_port}")
    if bool_enabled "${TMX_VLLM_RANK_AGNOSTIC_PROXY:-0}"; then
      urls+=("http://${train_source_ip}:${proxy_port}/v1")
    else
      urls+=("http://127.0.0.1:${local_port}/v1")
    fi
    idx=$((idx + 1))
  done
  export TMX_VLLM_SERVER_URLS="$(join_by_comma "${urls[@]}")"
  export VLLM_SERVER_URLS="$TMX_VLLM_SERVER_URLS"
  local specs_csv tunnel_script
  specs_csv="$(join_by_comma "${specs[@]}")"
  tunnel_script="$(mktemp /tmp/tmx_start_vllm_tunnels.XXXXXX.sh)"
  cat > "$tunnel_script" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /tmp/tmx_vllm_tunnels
pkill -f "[g]cloud .*${SERVE_TPU_NAME}.*127.0.0.1:.*:${SERVE_PORT}" 2>/dev/null || true
IFS=',' read -ra pairs <<< "$TUNNEL_SPECS"
for pair in "${pairs[@]}"; do
  IFS=':' read -r serve_worker local_port proxy_port <<< "$pair"
  log="/tmp/tmx_vllm_tunnels/serve_w${serve_worker}_p${local_port}.log"
nohup gcloud --project="$SERVE_PROJECT" alpha compute tpus tpu-vm ssh "$SERVE_TPU_NAME" \
    --zone="$SERVE_ZONE" \
    --worker="$serve_worker" \
    $GCLOUD_TPU_IAP_FLAGS \
    --ssh-flag='-N' \
    --ssh-flag="-L 127.0.0.1:${local_port}:127.0.0.1:${SERVE_PORT}" \
    --ssh-flag='-o ExitOnForwardFailure=yes' \
    --ssh-flag='-o BatchMode=yes' \
    --ssh-flag='-o StrictHostKeyChecking=no' \
    --ssh-flag='-o UserKnownHostsFile=/tmp/tmx_vllm_tunnels/known_hosts' \
    --ssh-flag='-o ServerAliveInterval=15' \
    --ssh-flag='-o ServerAliveCountMax=4' \
    >"$log" 2>&1 &
done
sleep "${TMX_VLLM_TUNNEL_WARMUP_SECONDS:-8}"
for pair in "${pairs[@]}"; do
  IFS=':' read -r serve_worker local_port proxy_port <<< "$pair"
  curl -fsS --max-time 5 "http://127.0.0.1:${local_port}/health" >/dev/null || {
    echo "tunnel health failed serve_worker=${serve_worker} local_port=${local_port}" >&2
    cat "/tmp/tmx_vllm_tunnels/serve_w${serve_worker}_p${local_port}.log" >&2 || true
    exit 1
  }
done
if [ "${TMX_VLLM_RANK_AGNOSTIC_PROXY:-0}" = "1" ] || [ "${TMX_VLLM_RANK_AGNOSTIC_PROXY:-0}" = "true" ]; then
  if ! command -v socat >/dev/null 2>&1; then
    sudo apt-get update -y >/dev/null
    sudo apt-get install -y socat >/dev/null
  fi
  pkill -f '[t]mx_vllm_proxy_' 2>/dev/null || true
  mkdir -p /tmp/tmx_vllm_proxies
  for pair in "${pairs[@]}"; do
    IFS=':' read -r serve_worker local_port proxy_port <<< "$pair"
    log="/tmp/tmx_vllm_proxies/proxy_${proxy_port}.log"
    nohup bash -c "exec -a tmx_vllm_proxy_${proxy_port} socat TCP-LISTEN:${proxy_port},bind=0.0.0.0,fork,reuseaddr TCP:127.0.0.1:${local_port}" >"$log" 2>&1 &
  done
  sleep "${TMX_VLLM_PROXY_WARMUP_SECONDS:-3}"
  for pair in "${pairs[@]}"; do
    IFS=':' read -r serve_worker local_port proxy_port <<< "$pair"
    curl -fsS --max-time 5 "http://127.0.0.1:${proxy_port}/health" >/dev/null || {
      echo "rank-agnostic proxy health failed proxy_port=${proxy_port} local_port=${local_port}" >&2
      cat "/tmp/tmx_vllm_proxies/proxy_${proxy_port}.log" >&2 || true
      exit 1
    }
  done
  echo "rank_agnostic_proxy_health_ok count=${#pairs[@]}"
fi
echo "tunnel_health_ok count=${#pairs[@]}"
SH
  chmod +x "$tunnel_script"
  echo "[dapo-restart] starting train-source IAP tunnels on worker $train_worker for vLLM workers=[$serve_workers]"
  gcloud --project="$train_project" alpha compute tpus tpu-vm scp "$tunnel_script" "$train_tpu_name:/tmp/tmx_start_vllm_tunnels.sh" \
    --worker="$train_worker" \
    --zone="$train_zone" \
    $GCLOUD_TPU_IAP_FLAGS
  rm -f "$tunnel_script"
  local escaped_specs escaped_iap_flags
  escaped_specs="$(printf '%q' "$specs_csv")"
  escaped_iap_flags="$(printf '%q' "$GCLOUD_TPU_IAP_FLAGS")"
  gcloud --project="$train_project" alpha compute tpus tpu-vm ssh "$train_tpu_name" \
    --worker="$train_worker" \
    --zone="$train_zone" \
    $GCLOUD_TPU_IAP_FLAGS \
    --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
    --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
    --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
    --command="SERVE_PROJECT='$serve_project' SERVE_TPU_NAME='$serve_tpu_name' SERVE_ZONE='$serve_zone' SERVE_PORT='$serve_port' TUNNEL_SPECS=$escaped_specs GCLOUD_TPU_IAP_FLAGS=$escaped_iap_flags TMX_VLLM_TUNNEL_WARMUP_SECONDS='${TMX_VLLM_TUNNEL_WARMUP_SECONDS:-8}' TMX_VLLM_RANK_AGNOSTIC_PROXY='${TMX_VLLM_RANK_AGNOSTIC_PROXY:-0}' TMX_VLLM_PROXY_WARMUP_SECONDS='${TMX_VLLM_PROXY_WARMUP_SECONDS:-3}' bash /tmp/tmx_start_vllm_tunnels.sh"
  echo "[dapo-restart] selected_vllm_urls=$TMX_VLLM_SERVER_URLS"
}

kill_stale_slice_jobs() {
  local tpu_name="$1"
  local zone="$2"
  local project="$3"
  local workers="$4"
  local label="$5"
  local worker_count
  worker_count="$(printf '%s\n' "$workers" | wc -w | tr -d ' ')"
  local fast_clean_min_workers="${TMX_FAST_CLEAN_MIN_WORKERS:-16}"
  if [ "${TMX_FAST_CLEAN_SLICE:-1}" = "1" ] && [ "$worker_count" -ge "$fast_clean_min_workers" ]; then
    echo "[dapo-restart] cleaning $label workers in one batched call: [$workers]"
    gcloud --project="$project" alpha compute tpus tpu-vm ssh "$tpu_name" \
      --worker=all \
      --zone="$zone" \
      $GCLOUD_TPU_IAP_FLAGS \
      --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
      --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
      --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
      --command="
        pkill -9 -f '[p]ython -m tmx_jax' 2>/dev/null || true
        pkill -9 -f '[V]LLM::EngineCore|[v]llm.entrypoints.openai.api_server|[v]llm serve' 2>/dev/null || true
        pkill -9 -f '[t]mx_vllm_proxy_' 2>/dev/null || true
        tmux ls 2>/dev/null | while IFS=: read -r s rest; do
          case \"\$s\" in *jax_*|*dapo*|*pipeline*|*vllm*) tmux kill-session -t \"\$s\" 2>/dev/null || true ;; esac
        done
      " >/dev/null || true
    return 0
  fi
  for worker in $workers; do
    echo "[dapo-restart] cleaning $label worker $worker"
    gcloud --project="$project" alpha compute tpus tpu-vm ssh "$tpu_name" \
      --worker="$worker" \
      --zone="$zone" \
      $GCLOUD_TPU_IAP_FLAGS \
      --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
      --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
      --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
      --command="
        pkill -9 -f '[p]ython -m tmx_jax' 2>/dev/null || true
        pkill -9 -f '[V]LLM::EngineCore|[v]llm.entrypoints.openai.api_server|[v]llm serve' 2>/dev/null || true
        pkill -9 -f '[t]mx_vllm_proxy_' 2>/dev/null || true
        tmux ls 2>/dev/null | while IFS=: read -r s rest; do
          case \"\$s\" in *jax_*|*dapo*|*pipeline*|*vllm*) tmux kill-session -t \"\$s\" 2>/dev/null || true ;; esac
        done
      " >/dev/null || true
  done
}

APPLY_TMX_JAX_PATCHES="${APPLY_TMX_JAX_PATCHES:-1}"
APPLY_HF_RELOAD_PATCH="${APPLY_HF_RELOAD_PATCH:-1}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:-jax_dapo_drgrpo_twoslice_$(date +%Y%m%d_%H%M%S)}"
REMOTE_REPO="${REMOTE_REPO:?set REMOTE_REPO to the absolute repo path on every TPU worker}"
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT to the absolute runs directory on the TPU VM}"

TRAIN_TPU_NAME="${TRAIN_TPU_NAME:?set TRAIN_TPU_NAME to the training TPU VM (slice) name}"
TRAIN_ZONE="${TRAIN_ZONE:-us-central1-a}"
SERVE_TPU_NAME="${SERVE_TPU_NAME:?set SERVE_TPU_NAME to the serving (vLLM) TPU VM (slice) name}"
SERVE_ZONE="${SERVE_ZONE:-us-central1-a}"
TRAIN_TPU_PROJECT="${TRAIN_TPU_PROJECT:-$(project_for_tpu "$TRAIN_TPU_NAME")}"
SERVE_TPU_PROJECT="${SERVE_TPU_PROJECT:-$(project_for_tpu "$SERVE_TPU_NAME")}"

require_env TRAIN_TPU_PROJECT
require_env SERVE_TPU_PROJECT

if [ "$TRAIN_TPU_NAME" = "$SERVE_TPU_NAME" ] && [ "$TRAIN_ZONE" = "$SERVE_ZONE" ]; then
  if [ "${TMX_SINGLE_TPU_DISJOINT_TRAIN_SERVE:-0}" != "1" ] && [ "${TMX_ALLOW_SAME_WORKER_CHIP_SPLIT:-0}" != "1" ]; then
    echo "[dapo-restart] ERROR: TRAIN_TPU_NAME and SERVE_TPU_NAME must be different full slices" >&2
    echo "[dapo-restart] Set TMX_SINGLE_TPU_DISJOINT_TRAIN_SERVE=1 only when TRAIN_WORKERS and SERVE_WORKERS are disjoint." >&2
    exit 2
  fi
fi
if [ "${TMX_REQUIRE_SAME_REGION_TPU:-1}" != "0" ]; then
  require_same_region_tpus
fi

if bool_enabled "$APPLY_TMX_JAX_PATCHES" && [ ! -w "$TMX_JAX_REPO/tmx_jax/config.py" ]; then
  patched_repo="/tmp/tmx_jax_dapo_${RUN_NAME}"
  if [ ! -d "$patched_repo" ]; then
    echo "[dapo-restart] local tmx-jax checkout is read-only; creating patched copy at $patched_repo"
    cp -R "$TMX_JAX_REPO" "$patched_repo"
  fi
  TMX_JAX_REPO="$patched_repo"
fi

if bool_enabled "$APPLY_TMX_JAX_PATCHES"; then
  patch_once "$ROOT_DIR/patches/dapo_drgrpo_full_contract.patch" "DAPO+DR-GRPO full-contract patch"
  if bool_enabled "$APPLY_HF_RELOAD_PATCH" && [ -f "$ROOT_DIR/patches/hf_first_reload.patch" ]; then
    patch_once "$ROOT_DIR/patches/hf_first_reload.patch" "HF-first reload patch"
  fi
  if [ -f "$ROOT_DIR/patches/hf_only_artifacts.patch" ]; then
    patch_once "$ROOT_DIR/patches/hf_only_artifacts.patch" "HF-only artifacts patch"
  fi
  if [ -f "$ROOT_DIR/patches/vllm_tp1_topology.patch" ]; then
    patch_once "$ROOT_DIR/patches/vllm_tp1_topology.patch" "vLLM TP=1 topology patch"
  fi
  if [ -f "$ROOT_DIR/patches/vllm_force_v0_engine.patch" ]; then
    patch_once "$ROOT_DIR/patches/vllm_force_v0_engine.patch" "vLLM force V0 engine patch"
  fi
  if [ -f "$ROOT_DIR/patches/vllm_chunked_prefill_switch.patch" ]; then
    patch_once "$ROOT_DIR/patches/vllm_chunked_prefill_switch.patch" "vLLM chunked prefill switch patch"
  fi
  if [ -f "$ROOT_DIR/patches/vllm_bootstrap_timeouts_and_skip.patch" ]; then
    patch_once "$ROOT_DIR/patches/vllm_bootstrap_timeouts_and_skip.patch" "vLLM bootstrap timeouts/skip patch"
  fi
  if [ -f "$ROOT_DIR/patches/train_tmux_cleanup_no_session.patch" ]; then
    patch_once "$ROOT_DIR/patches/train_tmux_cleanup_no_session.patch" "train tmux cleanup no-session patch"
  fi
  if [ -f "$ROOT_DIR/patches/train_tmux_nohup_fallback.patch" ]; then
    patch_once "$ROOT_DIR/patches/train_tmux_nohup_fallback.patch" "train tmux nohup fallback patch"
  fi
  if [ -f "$ROOT_DIR/patches/train_nohup_quote_fix.patch" ]; then
    patch_once "$ROOT_DIR/patches/train_nohup_quote_fix.patch" "train nohup quote fix patch"
  fi
  if [ -f "$ROOT_DIR/patches/train_serial_worker_start.patch" ]; then
    patch_once "$ROOT_DIR/patches/train_serial_worker_start.patch" "train serial worker start patch"
  fi
fi

TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
SERVE_WORKERS="${SERVE_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
if [ "$TRAIN_TPU_NAME" = "$SERVE_TPU_NAME" ] && [ "$TRAIN_ZONE" = "$SERVE_ZONE" ] && [ "${TMX_SINGLE_TPU_DISJOINT_TRAIN_SERVE:-0}" = "1" ] && [ "${TMX_ALLOW_SAME_WORKER_CHIP_SPLIT:-0}" != "1" ]; then
  overlap="$(
    python3 - "$TRAIN_WORKERS" "$SERVE_WORKERS" <<'PY'
import sys
train = set(sys.argv[1].split())
serve = set(sys.argv[2].split())
print(" ".join(sorted(train & serve, key=int)))
PY
  )"
  if [ -n "$overlap" ]; then
    echo "[dapo-restart] ERROR: same-TPU train/serve worker overlap: $overlap" >&2
    exit 2
  fi
fi
TRAIN_WORKER_COUNT="$(printf '%s\n' "$TRAIN_WORKERS" | wc -w | tr -d ' ')"
SERVE_WORKER_COUNT="$(printf '%s\n' "$SERVE_WORKERS" | wc -w | tr -d ' ')"
ALLOW_DEGRADED_SERVE="${ALLOW_DEGRADED_SERVE:-0}"
MIN_HEALTHY_SERVE_WORKERS="${MIN_HEALTHY_SERVE_WORKERS:-$SERVE_WORKER_COUNT}"
export ALLOW_DEGRADED_SERVE MIN_HEALTHY_SERVE_WORKERS

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-20480}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-$((MAX_PROMPT_LEN + MAX_COMPLETION_LEN))}"
SOFT_OVERLONG_CACHE_LEN="${SOFT_OVERLONG_CACHE_LEN:-$((MAX_COMPLETION_LEN / 5))}"
SOFT_OVERLONG_EXPECTED_LEN="${SOFT_OVERLONG_EXPECTED_LEN:-$((MAX_COMPLETION_LEN - SOFT_OVERLONG_CACHE_LEN))}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MAX_TOTAL_LEN}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-$MAX_TOTAL_LEN}"
PORT="${PORT:-8000}"
TMX_VLLM_MULTI_SERVER="${TMX_VLLM_MULTI_SERVER:-0}"
TMX_VLLM_MULTI_SERVER_CHIPS="${TMX_VLLM_MULTI_SERVER_CHIPS:-0 1 2 3}"
TMX_VLLM_MULTI_SERVER_PORT_BASE="${TMX_VLLM_MULTI_SERVER_PORT_BASE:-$PORT}"
if bool_enabled "$TMX_VLLM_MULTI_SERVER"; then
  SERVE_CHIP_COUNT="$(printf '%s\n' "$TMX_VLLM_MULTI_SERVER_CHIPS" | wc -w | tr -d ' ')"
else
  SERVE_CHIP_COUNT=1
fi
SERVE_ENDPOINT_COUNT=$((SERVE_WORKER_COUNT * SERVE_CHIP_COUNT))
export TMX_VLLM_MULTI_SERVER TMX_VLLM_MULTI_SERVER_CHIPS TMX_VLLM_MULTI_SERVER_PORT_BASE SERVE_ENDPOINT_COUNT
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-$SERVE_ENDPOINT_COUNT}"
VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-16}"
export VLLM_CONCURRENCY VLLM_FANOUT_PER_PROMPT

case "$MAX_COMPLETION_LEN" in
  8192|12288|16384|20480) ;;
  *)
    echo "[dapo-restart] ERROR: DAPO contract supports MAX_COMPLETION_LEN=8192/12288/16384/20480" >&2
    exit 2
    ;;
esac
if [ "$MAX_PROMPT_LEN" != "2048" ] || [ "$MAX_TOTAL_LEN" != "$((MAX_PROMPT_LEN + MAX_COMPLETION_LEN))" ]; then
  echo "[dapo-restart] ERROR: DAPO contract requires MAX_PROMPT_LEN=2048 and MAX_TOTAL_LEN=prompt+completion" >&2
  exit 2
fi

serve_ips_raw="$(worker_ips "$SERVE_TPU_NAME" "$SERVE_ZONE" "$SERVE_TPU_PROJECT")"
IFS=';' read -r -a serve_ips <<< "$serve_ips_raw"

if [ -z "${TMX_VLLM_SERVER_URLS:-${VLLM_SERVER_URLS:-}}" ]; then
  set_vllm_urls_for_workers "$SERVE_WORKERS"
else
  export TMX_VLLM_SERVER_URLS="${TMX_VLLM_SERVER_URLS:-$VLLM_SERVER_URLS}"
fi
export VLLM_SERVER_URLS="$TMX_VLLM_SERVER_URLS"

echo "[dapo-restart] run=$RUN_NAME"
echo "[dapo-restart] train=$TRAIN_TPU_NAME project=$TRAIN_TPU_PROJECT zone=$TRAIN_ZONE workers=[$TRAIN_WORKERS]"
echo "[dapo-restart] serve=$SERVE_TPU_NAME project=$SERVE_TPU_PROJECT zone=$SERVE_ZONE workers=[$SERVE_WORKERS]"
echo "[dapo-restart] vllm_urls=$TMX_VLLM_SERVER_URLS"
echo "[dapo-restart] vllm_topology: multi_server=$TMX_VLLM_MULTI_SERVER chips=[$TMX_VLLM_MULTI_SERVER_CHIPS] port_base=$TMX_VLLM_MULTI_SERVER_PORT_BASE endpoints=$SERVE_ENDPOINT_COUNT"
echo "[dapo-restart] serve gate: allow_degraded=$ALLOW_DEGRADED_SERVE min_healthy=$MIN_HEALTHY_SERVE_WORKERS/$SERVE_WORKER_COUNT"
echo "[dapo-restart] contract: dr_grpo, G=${GRPO_NUM_GENERATIONS:-16}, max_completion=$MAX_COMPLETION_LEN, clip_high=0.28, dynamic_sampling, soft_overlong=${SOFT_OVERLONG_EXPECTED_LEN}+${SOFT_OVERLONG_CACHE_LEN}, KL=0"

if bool_enabled "$DRY_RUN"; then
  echo "[dapo-restart] DRY_RUN=1, not launching"
  exit 0
fi

if bool_enabled "${CLEAN_SERVE_SLICE:-1}" && ! bool_enabled "${BOOTSTRAP_SERVE_VLLM:-1}"; then
  kill_stale_slice_jobs "$SERVE_TPU_NAME" "$SERVE_ZONE" "$SERVE_TPU_PROJECT" "$SERVE_WORKERS" "serve"
fi

if ! bool_enabled "${BOOTSTRAP_SERVE_VLLM:-1}" && bool_enabled "${VERIFY_EXISTING_SERVE_VLLM:-0}"; then
  LAST_HEALTHY_WORKERS=""
  if wait_worker_health "$SERVE_TPU_NAME" "$SERVE_ZONE" "$SERVE_TPU_PROJECT" "$SERVE_WORKERS"; then
    selected_workers="${LAST_HEALTHY_WORKERS:-$SERVE_WORKERS}"
    if [ "$selected_workers" != "$SERVE_WORKERS" ]; then
      echo "[dapo-restart] using healthy existing vLLM workers=[$selected_workers]"
      set_vllm_urls_for_workers "$selected_workers"
    fi
    probe_worker="$(printf '%s\n' "$TRAIN_WORKERS" | awk '{print $1}')"
    start_train_source_vllm_tunnels \
      "$TRAIN_TPU_NAME" "$TRAIN_ZONE" "$TRAIN_TPU_PROJECT" "$probe_worker" \
      "$SERVE_TPU_NAME" "$SERVE_ZONE" "$SERVE_TPU_PROJECT" "$selected_workers" "$PORT"
    echo "[dapo-restart] selected_vllm_urls=$TMX_VLLM_SERVER_URLS"
    verify_train_side_completions "$TRAIN_TPU_NAME" "$TRAIN_ZONE" "$TRAIN_TPU_PROJECT" "$probe_worker" "$TMX_VLLM_SERVER_URLS" "$MODEL_NAME"
  else
    echo "[dapo-restart] ERROR: existing vLLM serve did not satisfy health threshold" >&2
    exit 3
  fi
fi

if bool_enabled "${BOOTSTRAP_SERVE_VLLM:-1}"; then
  bootstrap_ok=0
  seq_candidates="${VLLM_MAX_NUM_SEQS_CANDIDATES:-$VLLM_MAX_NUM_SEQS 2 4}"
  # Production path: one TP=4 vLLM server per worker. This gives 16 stable
  # endpoints on a v5litepod-64 without multiplying the RPA warmup failure rate
  # across 64 chip-local processes.
  tp_candidates="${VLLM_TP_SIZE_CANDIDATES:-${VLLM_TP_SIZE:-4}}"
  seen_seq_candidates=""
  seen_tp_candidates=""
  for tp_size in $tp_candidates; do
    case " $seen_tp_candidates " in *" $tp_size "*) continue ;; esac
    seen_tp_candidates="$seen_tp_candidates $tp_size"
    seen_seq_candidates=""
    for seqs in $seq_candidates; do
      case " $seen_seq_candidates " in *" $seqs "*) continue ;; esac
      seen_seq_candidates="$seen_seq_candidates $seqs"
      echo "[dapo-restart] bootstrapping vLLM: max_model_len=$VLLM_MAX_MODEL_LEN tp=$tp_size max_num_seqs=$seqs"
    if bool_enabled "${CLEAN_SERVE_SLICE:-1}"; then
      kill_stale_slice_jobs "$SERVE_TPU_NAME" "$SERVE_ZONE" "$SERVE_TPU_PROJECT" "$SERVE_WORKERS" "serve"
    else
      echo "[dapo-restart] skipping serve cleanup because CLEAN_SERVE_SLICE=0"
    fi
    (
      cd "$TMX_JAX_REPO"
      if bool_enabled "$TMX_VLLM_MULTI_SERVER"; then
        TPU_NAME="$SERVE_TPU_NAME" \
        TPU_PROJECT="$SERVE_TPU_PROJECT" \
        ZONE="$SERVE_ZONE" \
        WORKERS="$SERVE_WORKERS" \
        CHIPS="$TMX_VLLM_MULTI_SERVER_CHIPS" \
        PORT_BASE="$TMX_VLLM_MULTI_SERVER_PORT_BASE" \
        MODEL="$MODEL_NAME" \
        MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
        MAX_NUM_SEQS="$seqs" \
        MAX_NUM_BATCHED_TOKENS="$VLLM_MAX_NUM_BATCHED_TOKENS" \
        VLLM_TPU_BUCKET_PADDING_GAP="${VLLM_TPU_BUCKET_PADDING_GAP:-4096}" \
        VLLM_TPU_MOST_MODEL_LEN="${VLLM_TPU_MOST_MODEL_LEN:-$MAX_TOTAL_LEN}" \
        VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}" \
        VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-1}" \
        VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-tp1_pda1_mlen${VLLM_MAX_MODEL_LEN}_mbt${VLLM_MAX_NUM_BATCHED_TOKENS}}" \
        HEALTH_TIMEOUT_SECONDS="${VLLM_HEALTH_TIMEOUT_SECONDS:-1800}" \
        TMX_VLLM_BOOTSTRAP_RETRIES="${TMX_VLLM_BOOTSTRAP_RETRIES:-4}" \
        TMX_VLLM_BOOTSTRAP_RETRY_SLEEP="${TMX_VLLM_BOOTSTRAP_RETRY_SLEEP:-8}" \
        TMX_VLLM_SSH_CONNECT_TIMEOUT_SECONDS="${TMX_VLLM_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
        bash tmx_jax/scripts/bootstrap_vllm_multiserver_slice.sh
      else
        TPU_NAME="$SERVE_TPU_NAME" \
        TPU_PROJECT="$SERVE_TPU_PROJECT" \
        ZONE="$SERVE_ZONE" \
        WORKERS="$SERVE_WORKERS" \
        MODEL="$MODEL_NAME" \
        PORT="$PORT" \
        TP="$tp_size" \
        MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
        MAX_NUM_SEQS="$seqs" \
        MAX_NUM_BATCHED_TOKENS="$VLLM_MAX_NUM_BATCHED_TOKENS" \
        VLLM_TPU_BUCKET_PADDING_GAP="${VLLM_TPU_BUCKET_PADDING_GAP:-4096}" \
        VLLM_TPU_MOST_MODEL_LEN="${VLLM_TPU_MOST_MODEL_LEN:-$MAX_TOTAL_LEN}" \
        VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}" \
        VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-1}" \
        VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-tp${tp_size}_pda1_mlen${VLLM_MAX_MODEL_LEN}_mbt${VLLM_MAX_NUM_BATCHED_TOKENS}}" \
        MODEL_IMPL_TYPE="${MODEL_IMPL_TYPE:-auto}" \
        TMX_VLLM_SUPERVISE="${TMX_VLLM_SUPERVISE:-1}" \
        TMX_VLLM_RESTART_SLEEP_SECONDS="${TMX_VLLM_RESTART_SLEEP_SECONDS:-5}" \
        TMX_VLLM_BOOTSTRAP_RETRIES="${TMX_VLLM_BOOTSTRAP_RETRIES:-4}" \
        TMX_VLLM_BOOTSTRAP_RETRY_SLEEP="${TMX_VLLM_BOOTSTRAP_RETRY_SLEEP:-8}" \
        TMX_VLLM_SKIP_FAILED_WORKERS="${TMX_VLLM_SKIP_FAILED_WORKERS:-0}" \
        TMX_VLLM_SSH_CONNECT_TIMEOUT_SECONDS="${TMX_VLLM_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
        TMX_VLLM_SSH_SERVER_ALIVE_INTERVAL="${TMX_VLLM_SSH_SERVER_ALIVE_INTERVAL:-10}" \
        TMX_VLLM_SSH_SERVER_ALIVE_COUNT_MAX="${TMX_VLLM_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
        bash tmx_jax/scripts/bootstrap_vllm_slice.sh
      fi
    )
    if [ "${TMX_VLLM_POST_BOOTSTRAP_WARMUP_SECONDS:-0}" != "0" ]; then
      echo "[dapo-restart] waiting ${TMX_VLLM_POST_BOOTSTRAP_WARMUP_SECONDS}s for vLLM post-bootstrap warmup"
      sleep "$TMX_VLLM_POST_BOOTSTRAP_WARMUP_SECONDS"
    fi
    LAST_HEALTHY_WORKERS=""
    if wait_worker_health "$SERVE_TPU_NAME" "$SERVE_ZONE" "$SERVE_TPU_PROJECT" "$SERVE_WORKERS"; then
      selected_workers="${LAST_HEALTHY_WORKERS:-$SERVE_WORKERS}"
      if [ "$selected_workers" != "$SERVE_WORKERS" ]; then
        echo "[dapo-restart] using healthy vLLM workers=[$selected_workers]"
        set_vllm_urls_for_workers "$selected_workers"
      fi
      VLLM_TP_SIZE="$tp_size"
      VLLM_MAX_NUM_SEQS="$seqs"
      bootstrap_ok=1
      break
    fi
    if bool_enabled "${ALLOW_DEGRADED_SERVE:-0}"; then
      healthy_workers="$(healthy_worker_list "$SERVE_TPU_NAME" "$SERVE_ZONE" "$SERVE_TPU_PROJECT" "$SERVE_WORKERS")"
      healthy_count="$(printf '%s\n' "$healthy_workers" | wc -w | tr -d ' ')"
      min_healthy="${MIN_HEALTHY_SERVE_WORKERS:-2}"
      if [ "$healthy_count" -ge "$min_healthy" ]; then
        echo "[dapo-restart] WARN: proceeding with degraded vLLM workers=[$healthy_workers] count=$healthy_count/$min_healthy"
        set_vllm_urls_for_workers "$healthy_workers"
        SERVE_WORKERS="$healthy_workers"
        VLLM_TP_SIZE="$tp_size"
        VLLM_MAX_NUM_SEQS="$seqs"
        bootstrap_ok=1
        break
      fi
      echo "[dapo-restart] degraded serve unavailable: healthy=[$healthy_workers] count=$healthy_count min=$min_healthy"
    fi
      echo "[dapo-restart] vLLM health failed at TP=$tp_size MAX_NUM_SEQS=$seqs; retrying if available"
    done
    [ "$bootstrap_ok" = "1" ] && break
  done
  if [ "$bootstrap_ok" != "1" ]; then
    echo "[dapo-restart] ERROR: vLLM did not become healthy for any MAX_NUM_SEQS candidate" >&2
    exit 3
  fi
  probe_worker="$(printf '%s\n' "$TRAIN_WORKERS" | awk '{print $1}')"
  start_train_source_vllm_tunnels \
    "$TRAIN_TPU_NAME" "$TRAIN_ZONE" "$TRAIN_TPU_PROJECT" "$probe_worker" \
    "$SERVE_TPU_NAME" "$SERVE_ZONE" "$SERVE_TPU_PROJECT" "$SERVE_WORKERS" "$PORT"
  verify_train_side_completions "$TRAIN_TPU_NAME" "$TRAIN_ZONE" "$TRAIN_TPU_PROJECT" "$probe_worker" "$TMX_VLLM_SERVER_URLS" "$MODEL_NAME"
fi

if bool_enabled "${CLEAN_TRAIN_SLICE:-1}"; then
  kill_stale_slice_jobs "$TRAIN_TPU_NAME" "$TRAIN_ZONE" "$TRAIN_TPU_PROJECT" "$TRAIN_WORKERS" "train"
fi

(
  cd "$TMX_JAX_REPO"
  env \
    CLOUDSDK_CORE_PROJECT="$TRAIN_TPU_PROJECT" \
    TPU_NAME="$TRAIN_TPU_NAME" \
    ZONE="$TRAIN_ZONE" \
    TRAIN_WORKERS="$TRAIN_WORKERS" \
    SERVE_WORKERS="" \
    VLLM_RELOAD_TPU_NAME="$SERVE_TPU_NAME" \
    VLLM_RELOAD_TPU_PROJECT="$SERVE_TPU_PROJECT" \
    VLLM_RELOAD_ZONE="$SERVE_ZONE" \
    VLLM_RELOAD_WORKERS="$SERVE_WORKERS" \
    RUN_NAME="$RUN_NAME" \
    RUNS_ROOT="$RUNS_ROOT" \
    REMOTE_REPO="$REMOTE_REPO" \
    MODEL_NAME="$MODEL_NAME" \
    DATASET_NAME="${DATASET_NAME:-dapo_math_17k}" \
    TMX_TRAIN_DATASET_JSONL="${TMX_TRAIN_DATASET_JSONL:-}" \
    TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-64}" \
    TMX_USE_MANUAL_TPU_TOPOLOGY="${TMX_USE_MANUAL_TPU_TOPOLOGY:-1}" \
    TMX_TPU_HOST_BOUNDS="${TMX_TPU_HOST_BOUNDS:-4,4,1}" \
    TMX_JAX_CLUSTER_DETECTION_METHOD="${TMX_JAX_CLUSTER_DETECTION_METHOD-}" \
    JAX_HEARTBEAT_TIMEOUT_SECONDS="${JAX_HEARTBEAT_TIMEOUT_SECONDS:-900}" \
    TMX_TRAIN_LAUNCH_STABILITY_SECONDS="${TMX_TRAIN_LAUNCH_STABILITY_SECONDS:-3}" \
    TMX_PARALLEL_TRAIN_LAUNCH="${TMX_PARALLEL_TRAIN_LAUNCH:-0}" \
    TMX_TRAIN_LAUNCH_PARALLELISM="${TMX_TRAIN_LAUNCH_PARALLELISM:-16}" \
    SYNC_LOCAL_PACKAGE="${SYNC_LOCAL_PACKAGE:-1}" \
    TMX_FAST_PACKAGE_FANOUT="${TMX_FAST_PACKAGE_FANOUT:-1}" \
    TMX_FAST_WORKER_SCRIPT_FANOUT="${TMX_FAST_WORKER_SCRIPT_FANOUT:-1}" \
    TMX_REPAIR_TRAIN_LIBTPU="${TMX_REPAIR_TRAIN_LIBTPU:-0}" \
    TMX_LIBTPU_VERSION="${TMX_LIBTPU_VERSION:-0.0.20}" \
    MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}" \
    MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
    MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
    MAX_TOTAL_LEN="$MAX_TOTAL_LEN" \
    INIT_CHECKPOINT="${INIT_CHECKPOINT:-}" \
    TMX_INIT_CHECKPOINT_LOCAL_PATH="${TMX_INIT_CHECKPOINT_LOCAL_PATH:-}" \
    NOVELTY_FEATURE_MAX_LENGTH="${NOVELTY_FEATURE_MAX_LENGTH:-$MAX_TOTAL_LEN}" \
    TMX_NOVELTY_TRUNCATION_MODE="${TMX_NOVELTY_TRUNCATION_MODE:-tokenizer}" \
    TMX_NOVELTY_TEXT_HASH_FALLBACK="${TMX_NOVELTY_TEXT_HASH_FALLBACK:-0}" \
    TMX_NOVELTY_BROADCAST_DEBUG="${TMX_NOVELTY_BROADCAST_DEBUG:-0}" \
    TMX_NOVELTY_BATCH_SIZE="${TMX_NOVELTY_BATCH_SIZE:-1}" \
    TMX_NOVELTY_PROGRESS_EVERY_ROWS="${TMX_NOVELTY_PROGRESS_EVERY_ROWS:-32}" \
    TMX_ALLOW_LONG_COMPLETION_TRAIN=1 \
    LOGPROB_CHUNK_TOKENS="${LOGPROB_CHUNK_TOKENS:-256}" \
    TMX_GRPO_ROW_VECTOR_CHUNK="${TMX_GRPO_ROW_VECTOR_CHUNK:-1}" \
    TMX_VECTORIZE_GRPO_ROWS="${TMX_VECTORIZE_GRPO_ROWS:-0}" \
    TMX_DISABLE_ROLLOUT_DIVERSITY="${TMX_DISABLE_ROLLOUT_DIVERSITY:-0}" \
    TMX_DAPO_DRGRPO_FULL_CONTRACT=1 \
    TMX_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
    TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN="$SOFT_OVERLONG_EXPECTED_LEN" \
    TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN="$SOFT_OVERLONG_CACHE_LEN" \
    GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-16}" \
    GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}" \
    GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-4}" \
    TMX_EFFECTIVE_PROMPTS_PER_STEP="${TMX_EFFECTIVE_PROMPTS_PER_STEP:-}" \
    TMX_REPLICATE_ROLLOUT_BATCH="${TMX_REPLICATE_ROLLOUT_BATCH:-0}" \
    TMX_BATCH_SHARD_SOURCE_HOST="${TMX_BATCH_SHARD_SOURCE_HOST:-}" \
    TMX_BATCH_SHARD_HTTP_PORT="${TMX_BATCH_SHARD_HTTP_PORT:-18777}" \
    TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS="${TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS:-90}" \
    TMX_BATCH_SHARD_HTTP_ATTEMPTS="${TMX_BATCH_SHARD_HTTP_ATTEMPTS:-240}" \
    TMX_BATCH_SHARD_HTTP_TIMEOUT="${TMX_BATCH_SHARD_HTTP_TIMEOUT:-120}" \
    TMX_EXPECTED_GRPO_BATCH_SIZE="${TMX_EXPECTED_GRPO_BATCH_SIZE:-${GRPO_BATCH_SIZE:-1}}" \
    TMX_EXPECTED_GRPO_GRAD_ACCUM="${TMX_EXPECTED_GRPO_GRAD_ACCUM:-${GRPO_GRAD_ACCUM:-4}}" \
    TRAJECTORY_BATCH_SIZE="${TRAJECTORY_BATCH_SIZE:-4}" \
    GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-250}" \
    ACTUAL_ONLINE_GRPO_MAX_STEPS="${ACTUAL_ONLINE_GRPO_MAX_STEPS:-500}" \
    MIN_ACCEPTED_TRAJECTORIES="${MIN_ACCEPTED_TRAJECTORIES:-500}" \
    MAX_ACCEPTED_TRAJECTORIES="${MAX_ACCEPTED_TRAJECTORIES:-500}" \
    ACCEPTED_SELECTION_POLICY="${ACCEPTED_SELECTION_POLICY:-quality_first}" \
    TMX_PIPELINE_STAGE="${TMX_PIPELINE_STAGE:-explorer_novelty}" \
    TMX_PIPELINE_STOP_AFTER_EXPLORER="${TMX_PIPELINE_STOP_AFTER_EXPLORER:-0}" \
    TMX_PIPELINE_STOP_AFTER_SFT="${TMX_PIPELINE_STOP_AFTER_SFT:-0}" \
    TMX_PIPELINE_SKIP_EXPLORER="${TMX_PIPELINE_SKIP_EXPLORER:-0}" \
    TMX_EXISTING_TRAJECTORIES="${TMX_EXISTING_TRAJECTORIES:-}" \
    GRPO_LR="${GRPO_LR:-1e-6}" \
    ACTUAL_GRPO_LR="${ACTUAL_GRPO_LR:-1e-6}" \
    GRPO_CLIP_EPSILON=0.2 \
    GRPO_CLIP_EPSILON_HIGH=0.28 \
    GRPO_KL_BETA=0.0 \
    ACTUAL_GRPO_KL_COEF=0.0 \
    GRPO_UPDATES_PER_ROLLOUT="${GRPO_UPDATES_PER_ROLLOUT:-16}" \
    ACTUAL_GRPO_UPDATES_PER_ROLLOUT="${ACTUAL_GRPO_UPDATES_PER_ROLLOUT:-16}" \
    TMX_EXPECTED_GRPO_UPDATES_PER_ROLLOUT="${TMX_EXPECTED_GRPO_UPDATES_PER_ROLLOUT:-${GRPO_UPDATES_PER_ROLLOUT:-16}}" \
    GRPO_ADVANTAGE_NORMALIZATION="${GRPO_ADVANTAGE_NORMALIZATION:-none}" \
    DYNAMIC_SAMPLING=true \
    DYNAMIC_SAMPLING_MAX_ATTEMPTS="${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-16}" \
    DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-1}" \
    TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="${TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-1}}" \
    MASK_TRUNCATED_COMPLETIONS="${MASK_TRUNCATED_COMPLETIONS:-false}" \
    REWARD_CORRECT=1.0 \
    REWARD_INCORRECT=-1.0 \
    NONTERMINATION_PENALTY="${NONTERMINATION_PENALTY:-0.0}" \
    INVALID_ANSWER_PENALTY="${INVALID_ANSWER_PENALTY:-0.0}" \
    SOFT_OVERLONG_EXPECTED_LEN="$SOFT_OVERLONG_EXPECTED_LEN" \
    SOFT_OVERLONG_CACHE_LEN="$SOFT_OVERLONG_CACHE_LEN" \
    TEMPERATURE="${TEMPERATURE:-1.0}" \
    TOP_P="${TOP_P:-0.95}" \
    TOP_K="${TOP_K:-20}" \
    VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-64}" \
    VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-16}" \
    TMX_HTTP_POOL_MAXSIZE="${TMX_HTTP_POOL_MAXSIZE:-512}" \
    TMX_VLLM_ENDPOINT_MAX_INFLIGHT="${TMX_VLLM_ENDPOINT_MAX_INFLIGHT:-1}" \
    TMX_VLLM_ALLOW_ERROR_COMPLETIONS="${TMX_VLLM_ALLOW_ERROR_COMPLETIONS:-0}" \
    TMX_VLLM_FAILOVER_ROUNDS="${TMX_VLLM_FAILOVER_ROUNDS:-2}" \
    TMX_VLLM_FAILOVER_ROUND_SLEEP="${TMX_VLLM_FAILOVER_ROUND_SLEEP:-10}" \
    TMX_VLLM_RANDOMIZE_SEEDS="${TMX_VLLM_RANDOMIZE_SEEDS:-0}" \
    TMX_VLLM_SEED_MODE="${TMX_VLLM_SEED_MODE:-api}" \
    TMX_VLLM_SEED_API_FALLBACK="${TMX_VLLM_SEED_API_FALLBACK:-1}" \
    TMX_VLLM_SEED_BASE="${TMX_VLLM_SEED_BASE:-}" \
    TMX_VLLM_FANOUT_DESYNC="${TMX_VLLM_FANOUT_DESYNC:-0}" \
    TMX_VLLM_FANOUT_NONCE="${TMX_VLLM_FANOUT_NONCE:-0}" \
    TMX_VLLM_FANOUT_NONCE_MODE="${TMX_VLLM_FANOUT_NONCE_MODE:-instruction}" \
    TMX_VLLM_FANOUT_NONCE_TEMPLATE="${TMX_VLLM_FANOUT_NONCE_TEMPLATE:-}" \
    TP_SIZE="${VLLM_TP_SIZE:-4}" \
    TMX_VLLM_MULTI_SERVER="$TMX_VLLM_MULTI_SERVER" \
    TMX_VLLM_MULTI_SERVER_CHIPS="$TMX_VLLM_MULTI_SERVER_CHIPS" \
    TMX_VLLM_MULTI_SERVER_PORT_BASE="$TMX_VLLM_MULTI_SERVER_PORT_BASE" \
    TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS="${TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS:-25}" \
    TMX_TRAIN_VLLM_RELOAD_FIRST_STEP="${TMX_TRAIN_VLLM_RELOAD_FIRST_STEP:-1}" \
    TMX_TRAIN_VLLM_RELOAD_ON_RESUME="${TMX_TRAIN_VLLM_RELOAD_ON_RESUME:-1}" \
    TMX_TRAIN_VLLM_RELOAD_MODE="${TMX_TRAIN_VLLM_RELOAD_MODE:-external}" \
    TMX_VLLM_EXPORT_FROM_LIVE_ALLGATHER="${TMX_VLLM_EXPORT_FROM_LIVE_ALLGATHER:-1}" \
    TMX_VLLM_EXPORT_FROM_GCS_CHECKPOINT="${TMX_VLLM_EXPORT_FROM_GCS_CHECKPOINT:-0}" \
    TMX_EXTERNAL_VLLM_RELOAD_REQUIRED="${TMX_EXTERNAL_VLLM_RELOAD_REQUIRED:-1}" \
    TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL="${TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL:-1}" \
    TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT="${TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT:-0}" \
    TMX_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD="${TMX_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD:-0}" \
    TMX_LOCAL_VLLM_RELOAD_KEEP_LAST="${TMX_LOCAL_VLLM_RELOAD_KEEP_LAST:-1}" \
    TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS="${TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS:-0}" \
    TMX_VLLM_RELOAD_ALLOW_PARALLEL="${TMX_VLLM_RELOAD_ALLOW_PARALLEL:-0}" \
    TMX_VLLM_RELOAD_PARALLELISM="${TMX_VLLM_RELOAD_PARALLELISM:-${PARALLELISM:-8}}" \
    TPU_TRANSPORT_FLAG="${TPU_TRANSPORT_FLAG:-$([ "$TMX_GCLOUD_TPU_USE_IAP" = "1" ] && printf '%s' '--tunnel-through-iap' || printf '%s' 'direct')}" \
    TMX_ACTUAL_GRPO_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    TMX_TRAIN_RELOAD_VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
    TMX_ACTUAL_GRPO_VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
    TMX_VLLM_RELOAD_MAX_BATCHED_TOKENS="$VLLM_MAX_NUM_BATCHED_TOKENS" \
    VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-1}" \
    TMX_VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${TMX_VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}}" \
    VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-tp${VLLM_TP_SIZE:-4}_pda1_mlen${VLLM_MAX_MODEL_LEN}_mbt${VLLM_MAX_NUM_BATCHED_TOKENS}}" \
    TMX_VLLM_XLA_CACHE_TAG="${TMX_VLLM_XLA_CACHE_TAG:-${VLLM_XLA_CACHE_TAG:-tp${VLLM_TP_SIZE:-4}_pda1_mlen${VLLM_MAX_MODEL_LEN}_mbt${VLLM_MAX_NUM_BATCHED_TOKENS}}}" \
    TMX_AIME_PROBE_EVERY_STEPS="${TMX_AIME_PROBE_EVERY_STEPS:-25}" \
    TMX_AIME_PROBE_NUM_PROBLEMS="${TMX_AIME_PROBE_NUM_PROBLEMS:-6}" \
    TMX_AIME_PROBE_NUM_ROLLOUTS="${TMX_AIME_PROBE_NUM_ROLLOUTS:-4}" \
    TMX_AIME_PROBE_TEMPERATURE="${TMX_AIME_PROBE_TEMPERATURE:-1.0}" \
    TMX_AIME_PROBE_TOP_P="${TMX_AIME_PROBE_TOP_P:-0.7}" \
    TMX_FINAL_EVAL_TEMPERATURE="${TMX_FINAL_EVAL_TEMPERATURE:-1.0}" \
    TMX_FINAL_EVAL_TOP_P="${TMX_FINAL_EVAL_TOP_P:-0.7}" \
    SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-25}" \
    TMX_LOCAL_CHECKPOINT_EVERY_STEPS="${TMX_LOCAL_CHECKPOINT_EVERY_STEPS:-0}" \
    TMX_INIT_EMPTY_FOR_RESTORE="${TMX_INIT_EMPTY_FOR_RESTORE:-0}" \
    TMX_RESTORE_OPT_STATE="${TMX_RESTORE_OPT_STATE:-1}" \
    TMX_SAVE_OPT_STATE="${TMX_SAVE_OPT_STATE:-1}" \
    TMX_ARTIFACT_BACKEND="${TMX_ARTIFACT_BACKEND:-hf}" \
    TMX_ALLOW_GCS_ARTIFACTS="${TMX_ALLOW_GCS_ARTIFACTS:-0}" \
    TMX_HF_CHECKPOINT_EVERY_STEPS="${TMX_HF_CHECKPOINT_EVERY_STEPS:-${TMX_GCS_CHECKPOINT_EVERY_STEPS:-25}}" \
    TMX_HF_CHECKPOINT_FIRST_STEP="${TMX_HF_CHECKPOINT_FIRST_STEP:-${TMX_GCS_CHECKPOINT_FIRST_STEP:-1}}" \
    TMX_GCS_CHECKPOINT_EVERY_STEPS="${TMX_GCS_CHECKPOINT_EVERY_STEPS:-${TMX_HF_CHECKPOINT_EVERY_STEPS:-25}}" \
    TMX_GCS_CHECKPOINT_FIRST_STEP="${TMX_GCS_CHECKPOINT_FIRST_STEP:-${TMX_HF_CHECKPOINT_FIRST_STEP:-1}}" \
    TMX_ORBAX_DIRECT_GCS_CHECKPOINT="${TMX_ORBAX_DIRECT_GCS_CHECKPOINT:-0}" \
    TMX_SKIP_MULTIHOST_LOCAL_ONLY_CHECKPOINT="${TMX_SKIP_MULTIHOST_LOCAL_ONLY_CHECKPOINT:-1}" \
    TMX_REQUIRE_GCS_CHECKPOINT_UPLOAD="${TMX_REQUIRE_GCS_CHECKPOINT_UPLOAD:-0}" \
    TMX_HF_CHECKPOINT_REPO="${TMX_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}" \
    TMX_HF_ARTIFACT_PREFIX="${TMX_HF_ARTIFACT_PREFIX:-$RUN_NAME}" \
    TMX_HF_MIRROR_CHECKPOINTS=1 \
    TMX_HF_MIRROR_TRAJECTORIES=1 \
    TMX_HF_MIRROR_METRICS=1 \
    TMX_HF_MIRROR_VLLM_EXPORTS="${TMX_HF_MIRROR_VLLM_EXPORTS:-1}" \
    TMX_HF_MIRROR_EVAL_PROBES=1 \
    TMX_HF_MIRROR_RUN_SUMMARY=1 \
    TMX_HF_MIRROR_FINAL_EXPORTS=1 \
    TMX_HF_MIRROR_FINAL_EVAL=1 \
    TMX_HF_MIRROR_ASYNC="${TMX_HF_MIRROR_ASYNC:-1}" \
    TMX_REQUIRE_HF_ARTIFACT_UPLOAD="${TMX_REQUIRE_HF_ARTIFACT_UPLOAD:-0}" \
    TMX_GCLOUD_TPU_USE_IAP="$TMX_GCLOUD_TPU_USE_IAP" \
    bash tmx_jax/scripts/launch_jax_pipeline_multihost_v5lite.sh
)
