#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"
ZONE="${ZONE:-us-central1-a}"
TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
WORKER="${WORKER:-0}"
EVAL_DIR="${EVAL_DIR:?set EVAL_DIR to the eval output directory on the TPU VM}"
SPEC="${SPEC:-configs/production_cyber_v5lite64_uc1a_explorer_novelty075.json}"
LOG_DIR="${LOG_DIR:-runs_logs}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_DIR"

echo "[wait-lambda075] project=$PROJECT zone=$ZONE tpu=$TPU_NAME worker=$WORKER"
echo "[wait-lambda075] waiting for eval summary under $EVAL_DIR"
echo "[wait-lambda075] then launching spec=$SPEC"

while true; do
  set +e
  status="$(
    gcloud --project="$PROJECT" alpha compute tpus tpu-vm ssh "$TPU_NAME" \
      --zone="$ZONE" --worker="$WORKER" --tunnel-through-iap \
      --command="python3 - <<'PY'
import glob, json, os
eval_dir = os.environ.get('EVAL_DIR', '$EVAL_DIR')
summaries = sorted(glob.glob(os.path.join(eval_dir, '*summary*.json')) + glob.glob(os.path.join(eval_dir, 'summary.json')))
if summaries:
    path = summaries[-1]
    print('SUMMARY=' + path)
    try:
        data = json.load(open(path))
        print('SUMMARY_JSON=' + json.dumps(data, sort_keys=True)[:1000])
    except Exception as exc:
        print('SUMMARY_READ_ERROR=' + repr(exc))
    raise SystemExit(0)
log = os.path.join(os.path.dirname(eval_dir), '.log')
if os.path.exists(log):
    with open(log, 'r', errors='replace') as f:
        lines = f.readlines()[-5:]
    print('LOG_TAIL_START')
    print(''.join(lines), end='')
    print('LOG_TAIL_END')
else:
    print('NO_LOG')
raise SystemExit(1)
PY" 2>&1
  )"
  rc=$?
  set -e
  printf '%s\n' "$status" | tail -40
  if [ "$rc" -eq 0 ] && grep -q '^SUMMARY=' <<<"$status"; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[wait-lambda075] eval complete; validating spec"
python3 scripts/tmx_submit.py validate --spec "$SPEC"

run_name="$(python3 - <<'PY'
import json
with open('configs/production_cyber_v5lite64_uc1a_explorer_novelty075.json') as f:
    print(json.load(f)['name'])
PY
)"
launch_log="$LOG_DIR/${run_name}_launch.log"
echo "[wait-lambda075] launching $run_name; log=$launch_log"
python3 scripts/tmx_submit.py submit \
  --spec "$SPEC" \
  --backend raw-tpu-tmx-jax \
  --execute \
  --wait-ready 2>&1 | tee "$launch_log"
