#!/usr/bin/env bash
# Local control-plane watcher for JAX TPU training vLLM reload requests.
#
# The training VM writes request JSON files at checkpoint boundaries, then waits
# for a `.done` marker. This watcher runs from the local machine, where gcloud
# TPU SSH works reliably, performs the vLLM slice reload, and marks completion.
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST to the ssh host alias of the training VM}"
REMOTE_RUNS_ROOT="${REMOTE_RUNS_ROOT:?set REMOTE_RUNS_ROOT to the absolute runs directory on the training VM}"
POLL_SECONDS="${POLL_SECONDS:-10}"
LOG="${LOG:-runs_logs/external_vllm_reload_watcher.log}"
RETRY_SECONDS="${TMX_EXTERNAL_VLLM_RELOAD_RETRY_SECONDS:-30}"
MAX_ATTEMPTS="${TMX_EXTERNAL_VLLM_RELOAD_MAX_ATTEMPTS:-0}"
SSH_CONNECT_TIMEOUT="${TMX_EXTERNAL_RELOAD_SSH_CONNECT_TIMEOUT:-10}"
HEARTBEAT_EVERY_POLLS="${TMX_EXTERNAL_RELOAD_HEARTBEAT_EVERY_POLLS:-60}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$(dirname "$LOG")"
log_startup_once="${LOG}.startup"
poll_count=0
SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o LogLevel=ERROR
  -o BatchMode=yes
  -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)

json_get() {
  local json_file="$1"
  local key="$2"
  python3 - "$json_file" "$key" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
value = data.get(sys.argv[2], "")
print(value if value is not None else "")
PY
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

mark_remote() {
  local path="$1"
  local content="$2"
  printf '%s\n' "$content" | ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cat > '$path'"
}

wait_remote_server_health() {
  local server_urls="$1"
  [ -n "$server_urls" ] || return 0
  local deadline timeout_s escaped_urls
  timeout_s="${TMX_EXTERNAL_RELOAD_REMOTE_HEALTH_TIMEOUT:-300}"
  deadline=$(( $(date +%s) + timeout_s ))
  escaped_urls="$(printf '%q' "$server_urls")"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    min_healthy="${TMX_EXTERNAL_RELOAD_MIN_HEALTHY_URLS:-0}"
    if ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "SERVER_URLS=$escaped_urls MIN_HEALTHY_URLS=$min_healthy python3 -c '
import os
import sys
import urllib.request

bad = []
good = 0
urls = [u.strip() for u in os.environ.get(\"SERVER_URLS\", \"\").split(\",\") if u.strip()]
required = int(os.environ.get(\"MIN_HEALTHY_URLS\") or \"0\")
if required <= 0:
    required = len(urls)
for url in urls:
    base = url[:-3] if url.endswith(\"/v1\") else url.rstrip(\"/\")
    try:
        urllib.request.urlopen(base + \"/health\", timeout=3).read()
        good += 1
    except Exception as exc:
        bad.append(f\"{url}: {type(exc).__name__}: {exc}\")
if good < required:
    print(f\"healthy_urls={good}/{len(urls)} required={required}\")
    if bad:
        print(\"\\n\".join(bad))
    sys.exit(1)
print(f\"healthy_urls={good}/{len(urls)} required={required}\")
'"; then
      return 0
    fi
    sleep "${TMX_EXTERNAL_RELOAD_REMOTE_HEALTH_POLL_SECONDS:-5}"
  done
  return 1
}

wait_remote_server_completions() {
  local server_urls="$1"
  local served_model_name="${2:-Qwen/Qwen3-1.7B}"
  [ -n "$server_urls" ] || return 0
  local deadline timeout_s poll_s probe_timeout escaped_urls escaped_model
  timeout_s="${TMX_EXTERNAL_RELOAD_REMOTE_COMPLETION_TIMEOUT:-600}"
  poll_s="${TMX_EXTERNAL_RELOAD_REMOTE_COMPLETION_POLL_SECONDS:-10}"
  probe_timeout="${TMX_EXTERNAL_RELOAD_COMPLETION_PROBE_TIMEOUT:-90}"
  deadline=$(( $(date +%s) + timeout_s ))
  escaped_urls="$(printf '%q' "$server_urls")"
  escaped_model="$(printf '%q' "$served_model_name")"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    min_healthy="${TMX_EXTERNAL_RELOAD_MIN_HEALTHY_URLS:-0}"
    if ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "SERVER_URLS=$escaped_urls SERVED_MODEL_NAME=$escaped_model PROBE_TIMEOUT=$probe_timeout MIN_HEALTHY_URLS=$min_healthy PROBE_CONCURRENCY=${TMX_EXTERNAL_RELOAD_COMPLETION_PROBE_CONCURRENCY:-16} python3 -c '
import concurrent.futures
import json
import os
import sys
import urllib.request

model = os.environ.get(\"SERVED_MODEL_NAME\") or \"Qwen/Qwen3-1.7B\"
timeout = float(os.environ.get(\"PROBE_TIMEOUT\") or 90)
urls = [u.strip() for u in os.environ.get(\"SERVER_URLS\", \"\").split(\",\") if u.strip()]
required = int(os.environ.get(\"MIN_HEALTHY_URLS\") or \"0\")
if required <= 0:
    required = len(urls)
concurrency = max(1, int(os.environ.get(\"PROBE_CONCURRENCY\") or \"16\"))
payload = json.dumps({
    \"model\": model,
    \"prompt\": \"Solve exactly: 2+2. Put the final answer in \\\\boxed{}.\",
    \"max_tokens\": 64,
    \"temperature\": 0.0,
    \"stream\": False,
}).encode()

def check_url(url):
    endpoint = url.rstrip(\"/\") + \"/completions\"
    try:
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={\"Content-Type\": \"application/json\"},
            method=\"POST\",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        choices = data.get(\"choices\") or []
        if not choices:
            return False, f\"{url}: no choices in completion probe\"
        else:
            text = str(choices[0].get(\"text\") or \"\")
            stripped = text.strip()
            bang_fraction = text.count(\"!\") / max(len(text), 1)
            sane_answer = \"4\" in text[:512] or \"boxed\" in text[:512].lower()
            if not stripped:
                return False, f\"{url}: empty completion probe\"
            elif len(text) >= 16 and bang_fraction > 0.80:
                return False, f\"{url}: repeated-bang completion probe: {text[:80]!r}\"
            elif not sane_answer:
                return False, f\"{url}: completion probe missing simple answer: {text[:160]!r}\"
            else:
                return True, f\"{url}: ok\"
    except Exception as exc:
        return False, f\"{url}: {type(exc).__name__}: {exc}\"

bad = []
good = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, max(len(urls), 1))) as pool:
    futures = [pool.submit(check_url, url) for url in urls]
    for fut in concurrent.futures.as_completed(futures):
        ok, message = fut.result()
        if ok:
            good += 1
        else:
            bad.append(message)
        remaining = sum(1 for f in futures if not f.done())
        if good >= required:
            break
        if good + remaining < required:
            break
if good < required:
    print(f\"completion_urls={good}/{len(urls)} required={required}\")
    if bad:
        print(\"\\n\".join(bad))
    sys.exit(1)
print(f\"completion_urls={good}/{len(urls)} required={required}\")
'"; then
      return 0
    fi
    sleep "$poll_s"
  done
  return 1
}

if [ ! -f "$log_startup_once" ]; then
  log "watcher started remote=$REMOTE_HOST max_attempts=$MAX_ATTEMPTS retry_seconds=$RETRY_SECONDS"
  touch "$log_startup_once"
fi

while true; do
  poll_count=$((poll_count + 1))
  active_run="$(ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" 'cat ~/active_jax_run_name 2>/dev/null || true' || true)"
  if [ -z "$active_run" ]; then
    log "no active run yet"
    sleep "$POLL_SECONDS"
    continue
  fi

  requests="$(
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
      "find '$REMOTE_RUNS_ROOT/$active_run' -path '*/vllm_reload_requests/*.request.json' -type f 2>/dev/null | sort | while IFS= read -r f; do b=\${f%.request.json}; if [ -f \"\$b.done\" ]; then continue; fi; if [ -f \"\$b.fail\" ]; then if [ '$MAX_ATTEMPTS' = '0' ]; then rm -f \"\$b.fail\"; echo \"\$f\"; fi; else echo \"\$f\"; fi; done" \
      || true
  )"
  pending_count=0

  while IFS= read -r request_path; do
    [ -n "$request_path" ] || continue
    done_path="${request_path%.request.json}.done"
    fail_path="${request_path%.request.json}.fail"
    attempts_path="${request_path%.request.json}.attempts"
    last_error_path="${request_path%.request.json}.last_error"
    pending_count=$((pending_count + 1))
    log "pending reload request active_run=$active_run request=$(basename "$request_path")"

    tmp_json="$(mktemp /tmp/tmx_reload_request_json.XXXXXX)"
    tmp_log="$(mktemp /tmp/tmx_reload_request_log.XXXXXX)"
    if ! ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cat '$request_path'" > "$tmp_json"; then
      log "failed to fetch request $request_path"
      rm -f "$tmp_json" "$tmp_log"
      continue
    fi

    gcs_model_path="$(json_get "$tmp_json" gcs_model_path)"
    hf_repo_id="$(json_get "$tmp_json" hf_repo_id)"
    hf_repo_type="$(json_get "$tmp_json" hf_repo_type)"
    hf_path_in_repo="$(json_get "$tmp_json" hf_path_in_repo)"
    hf_revision="$(json_get "$tmp_json" hf_revision)"
    tpu_name="$(json_get "$tmp_json" tpu_name)"
    tpu_project="$(json_get "$tmp_json" tpu_project)"
    zone="$(json_get "$tmp_json" zone)"
    workers="$(json_get "$tmp_json" workers)"
    server_urls="$(json_get "$tmp_json" server_urls)"
    multi_server="$(json_get "$tmp_json" multi_server)"
    chips="$(json_get "$tmp_json" chips)"
    port_base="$(json_get "$tmp_json" port_base)"
    model_name="$(json_get "$tmp_json" model_name)"
    max_model_len="$(json_get "$tmp_json" max_model_len)"
    stage="$(json_get "$tmp_json" stage)"
    if [ -z "$hf_repo_type" ]; then
      hf_repo_type="dataset"
    fi
    if [ -z "$gcs_model_path" ] && { [ -z "$hf_repo_id" ] || [ -z "$hf_path_in_repo" ]; }; then
      mark_remote "$last_error_path" "invalid reload request: missing artifact source; need gcs_model_path or hf_repo_id+hf_path_in_repo"
      log "invalid reload request $request_path; missing artifact source"
      rm -f "$tmp_json" "$tmp_log"
      continue
    fi
    if [ -z "$tpu_name" ] || [ -z "$zone" ] || [ -z "$workers" ]; then
      mark_remote "$last_error_path" "invalid reload request: missing tpu_name/zone/workers"
      log "invalid reload request $request_path; missing required fields"
      rm -f "$tmp_json" "$tmp_log"
      continue
    fi
    if [ -z "$tpu_project" ]; then
      # Requests written by train.py carry tpu_project; older ones fall back to the env.
      tpu_project="${TMX_TPU_PROJECT:-${CLOUDSDK_CORE_PROJECT:-}}"
    fi
    attempt="$(
      ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
        "n=0; [ -f '$attempts_path' ] && n=\$(cat '$attempts_path' 2>/dev/null || echo 0); n=\$((n + 1)); printf '%s\n' \"\$n\" > '$attempts_path'; printf '%s\n' \"\$n\"" \
        || echo 1
    )"

    if [ -n "$hf_repo_id" ] && [ -n "$hf_path_in_repo" ]; then
      log "reloading stage=$stage attempt=$attempt hf=${hf_repo_id}/${hf_path_in_repo} workers=[$workers]"
    else
      log "reloading stage=$stage attempt=$attempt gcs=$gcs_model_path workers=[$workers]"
    fi
    if (
      cd "$REPO_ROOT"
      GCS_MODEL_PATH="$gcs_model_path" \
      HF_REPO_ID="$hf_repo_id" \
      HF_REPO_TYPE="$hf_repo_type" \
      HF_PATH_IN_REPO="$hf_path_in_repo" \
      HF_REVISION="$hf_revision" \
      TPU_NAME="$tpu_name" \
      TPU_PROJECT="$tpu_project" \
      ZONE="$zone" \
      WORKERS="$workers" \
      SERVED_MODEL_NAME="$model_name" \
      LOCAL_MODEL_DIR_BASENAME="tmx_external_${stage}" \
      MAX_MODEL_LEN="${max_model_len:-10240}" \
      MAX_NUM_SEQS="${TMX_TRAIN_RELOAD_VLLM_MAX_NUM_SEQS:-32}" \
      MAX_NUM_BATCHED_TOKENS="${TMX_VLLM_RELOAD_MAX_BATCHED_TOKENS:-65536}" \
      TMX_VLLM_MULTI_SERVER="${multi_server:-${TMX_VLLM_MULTI_SERVER:-0}}" \
      CHIPS="${chips:-${TMX_VLLM_MULTI_SERVER_CHIPS:-0 1 2 3}}" \
      PORT_BASE="${port_base:-${TMX_VLLM_MULTI_SERVER_PORT_BASE:-8000}}" \
      VLLM_TPU_BUCKET_PADDING_GAP="${TMX_VLLM_TPU_BUCKET_PADDING_GAP:-${VLLM_TPU_BUCKET_PADDING_GAP:-}}" \
      VLLM_TPU_MOST_MODEL_LEN="${TMX_VLLM_TPU_MOST_MODEL_LEN:-${VLLM_TPU_MOST_MODEL_LEN:-}}" \
      VLLM_XLA_CACHE_PATH="${TMX_VLLM_XLA_CACHE_PATH:-${VLLM_XLA_CACHE_PATH:-}}" \
      VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${TMX_VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-0}}" \
      VLLM_XLA_CACHE_TAG="${TMX_VLLM_XLA_CACHE_TAG:-${VLLM_XLA_CACHE_TAG:-}}" \
      TMX_VLLM_SKIP_FAILED_WORKERS="${TMX_VLLM_SKIP_FAILED_WORKERS:-0}" \
      TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS="${TMX_VLLM_RELOAD_MIN_HEALTHY_WORKERS:-0}" \
      DISABLE_LOG_REQUESTS="${TMX_VLLM_DISABLE_LOG_REQUESTS:-1}" \
      PARALLELISM="${TMX_VLLM_RELOAD_PARALLELISM:-8}" \
      bash tmx_jax/scripts/reload_vllm_direct_slice.sh
    ) >"$tmp_log" 2>&1; then
      if ! wait_remote_server_health "$server_urls" >>"$tmp_log" 2>&1; then
        mark_remote "$last_error_path" "remote server URL health check failed after reload"
        log "reload finished but remote server URL health failed for $stage; keeping request pending"
        sleep "$RETRY_SECONDS"
        rm -f "$tmp_json" "$tmp_log"
        continue
      fi
      if [ "${TMX_EXTERNAL_RELOAD_SKIP_COMPLETION_PROBE:-0}" = "1" ]; then
        printf '%s\n' "completion probe skipped by TMX_EXTERNAL_RELOAD_SKIP_COMPLETION_PROBE=1" >>"$tmp_log"
      elif ! wait_remote_server_completions "$server_urls" "$model_name" >>"$tmp_log" 2>&1; then
        mark_remote "$last_error_path" "remote server URL completion probe failed after reload"
        log "reload finished but completion probe failed for $stage; keeping request pending"
        sleep "$RETRY_SECONDS"
        rm -f "$tmp_json" "$tmp_log"
        continue
      fi
      mark_remote "$done_path" "external reload ok $(date '+%Y-%m-%d %H:%M:%S')"
      ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "rm -f '$last_error_path' '$fail_path'" >/dev/null 2>&1 || true
      log "reload complete for $stage after attempt=$attempt"
    else
      tail -200 "$tmp_log" > "${tmp_log}.tail"
      mark_remote "$last_error_path" "$(cat "${tmp_log}.tail")"
      if [ "$MAX_ATTEMPTS" != "0" ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        mark_remote "$fail_path" "$(cat "${tmp_log}.tail")"
        log "reload FAILED for $stage after attempt=$attempt; wrote $fail_path"
      else
        log "reload failed for $stage attempt=$attempt; keeping request pending and retrying in ${RETRY_SECONDS}s"
        sleep "$RETRY_SECONDS"
      fi
    fi
    cat "$tmp_log" >> "$LOG" || true
    rm -f "$tmp_json" "$tmp_log" "${tmp_log}.tail"
  done <<< "$requests" || true

  if [ "$pending_count" -eq 0 ] && [ "$HEARTBEAT_EVERY_POLLS" != "0" ] && [ $((poll_count % HEARTBEAT_EVERY_POLLS)) -eq 0 ]; then
    log "heartbeat remote=$REMOTE_HOST active_run=$active_run pending=0"
  fi

  sleep "$POLL_SECONDS"
done
