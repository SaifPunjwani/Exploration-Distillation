#!/usr/bin/env bash
# Terminal monitor for a remote JAX TPU run.
#
# Usage:
#   ./tmx_jax/scripts/monitor_tpu_run.sh HOST RUN_NAME [INTERVAL_SECONDS]
#
# INTERVAL_SECONDS=0 prints once. Default is 60 seconds.
set -euo pipefail

HOST="${1:?usage: monitor_tpu_run.sh HOST RUN_NAME [INTERVAL_SECONDS]}"
RUN="${2:?usage: monitor_tpu_run.sh HOST RUN_NAME [INTERVAL_SECONDS]}"
INTERVAL="${3:-60}"

print_once() {
  ssh "$HOST" "RUN='$RUN' bash -s" <<'REMOTE'
set -u
ROOT="$HOME/two-model-exploration/runs/$RUN"
now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[$now] host=$(hostname) run=$RUN"
if [ ! -d "$ROOT" ]; then
  echo "missing run directory: $ROOT"
  exit 0
fi

proc="$(pgrep -af "python -m tmx_jax\.(pipeline|train).*$RUN" | head -1 || true)"
if [ -n "$proc" ]; then
  echo "process: alive  ${proc%% *}"
else
  echo "process: NOT FOUND"
fi

wandb_id="$(cat "$ROOT/.wandb_run_id" 2>/dev/null || true)"
if [ -n "$wandb_id" ]; then
  echo "wandb: https://wandb.ai/${WANDB_ENTITY:-<entity>}/two-model-explore/runs/$wandb_id"
fi

log="$ROOT/console.log"
latest_train="$(grep -E '^\[train\] step' "$log" 2>/dev/null | tail -1 || true)"
if [ -n "$latest_train" ]; then
  echo "latest: $latest_train"
else
  echo "latest: no train step in $log yet"
fi

metric_path="$(find "$ROOT" -path '*/training_metrics.jsonl' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
if [ -n "$metric_path" ] && [ -f "$metric_path" ]; then
  python - "$metric_path" <<'PY'
import json, math, statistics, sys
path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
print(f"metrics: {path} rows={len(rows)}")
if not rows:
    raise SystemExit

def get(row, *keys, default=0.0):
    for key in keys:
        if key in row:
            try:
                return float(row.get(key) or 0.0)
            except Exception:
                return default
    return default

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

latest = rows[-1]
step = int(latest.get("step") or 0)
max_steps = int(latest.get("max_steps") or 0)
reward = [get(r, "reward", "reward_mean") for r in rows]
correct = [get(r, "correctness", "correct_fraction") for r in rows]
eff = [get(r, "effective_group_fraction") for r in rows]
valid = [get(r, "valid_answer_fraction") for r in rows]
term = [get(r, "terminated_fraction") for r in rows]
clip = [get(r, "clipped_fraction") for r in rows]
total = [get(r, "total_time_s", default=get(r, "rollout_time_s") + get(r, "train_time_s")) for r in rows]
last10_total = mean(total[-10:])
remaining = max(0, max_steps - step)
eta_h = remaining * last10_total / 3600 if last10_total > 0 else 0.0
print(
    "summary: "
    f"step={step}/{max_steps} eta={eta_h:.1f}h "
    f"reward latest={reward[-1]:.3f} first10={mean(reward[:10]):.3f} last10={mean(reward[-10:]):.3f} last25={mean(reward[-25:]):.3f} "
    f"correct latest={correct[-1]:.3f} last10={mean(correct[-10:]):.3f} "
    f"eff_groups last10={mean(eff[-10:]):.2f} "
    f"valid last10={mean(valid[-10:]):.2f} term last10={mean(term[-10:]):.2f} clipped last10={mean(clip[-10:]):.2f}"
)
PY
fi

echo "checkpoint/reload:"
grep -E '\[ckpt\]|vllm-reload|eval-probe' "$log" 2>/dev/null | tail -10 || true

probe_dir="$ROOT/eval_probes"
if [ -d "$probe_dir" ]; then
  python - "$probe_dir" <<'PY'
import glob, json, os, sys
paths = sorted(glob.glob(os.path.join(sys.argv[1], "aime24*.json")))
if not paths:
    print("aime probe: none yet")
    raise SystemExit
print("aime probe:")
for path in paths[-5:]:
    try:
        d = json.load(open(path))
        p = d.get("protocol", {})
        step = p.get("step")
        avg = float(d.get("avg_at_n", 0.0) or 0.0)
        pass4 = float(d.get("pass_at_4", 0.0) or 0.0)
        delta = float(d.get("delta_vs_initial_avg_at_n", 0.0) or 0.0)
        best = float(d.get("best_avg_at_n_so_far", avg) or avg)
        err = float(d.get("error_fraction", 0.0) or 0.0)
        print(f"  {os.path.basename(path)} step={step} avg@{d.get('num_rollouts')}={avg:.3f} pass@4={pass4:.3f} delta={delta:+.3f} best={best:.3f} err={err:.3f}")
    except Exception as exc:
        print(f"  {os.path.basename(path)} parse_error={exc}")
PY
fi
REMOTE
}

if [ "$INTERVAL" = "0" ]; then
  print_once
else
  while true; do
    clear 2>/dev/null || true
    print_once
    sleep "$INTERVAL"
  done
fi
