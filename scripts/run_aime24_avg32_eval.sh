#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# AIME 2024 avg@32 evaluation via vLLM
#
# Launches vLLM with a given model, then runs avg@32 eval (32 samples per
# problem, temperature 1.0, 16k tokens). Results saved to a file.
#
# Usage:
#   MODEL_PATH=/tmp/tmx_model_save_actual \
#   EVAL_LABEL=actual_thinking \
#   bash scripts/run_aime24_avg32_eval.sh
#
# Env vars:
#   MODEL_PATH      Path to model weights (required)
#   EVAL_LABEL      Label for this eval run (default: eval)
#   VLLM_VENV_DIR   vLLM virtualenv (default: ~/vllm_tpu_env)
#   PORT            vLLM port (default: 8000)
#   NUM_ROLLOUTS    Samples per problem (default: 32)
#   TEMPERATURE     Sampling temperature (default: 1.0)
#   MAX_TOKENS      Max completion tokens (default: 16000)
#   MAX_MODEL_LEN   vLLM max context (default: 16384)
#   RESULTS_DIR     Where to save results (default: /tmp)
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ "${TMX_ALLOW_LEGACY_AIME24_AVG32_EVAL:-0}" != "1" ]; then
  cat >&2 <<'EOF'
[avg32-eval] deprecated launcher: this is not the canonical AIME24 avg@32 path.
[avg32-eval] use one of:
[avg32-eval]   bash scripts/run_qwen3_saved_aime24_avg32.sh
[avg32-eval]   bash scripts/run_qwen3_saved_aime24_avg32_sharded.sh
[avg32-eval] set TMX_ALLOW_LEGACY_AIME24_AVG32_EVAL=1 only if you intentionally want the old path.
EOF
  exit 2
fi

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
EVAL_LABEL="${EVAL_LABEL:-eval}"
VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
REPO_ROOT="${REPO_ROOT:-$HOME/two-model-exploration}"
PORT="${PORT:-8000}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-32}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
RESULTS_DIR="${RESULTS_DIR:-/tmp}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-eval_model}"
CONCURRENCY="${CONCURRENCY:-16}"

log() { printf '[avg32-eval] %s\n' "$*"; }

RESULTS_FILE="${RESULTS_DIR}/aime24_avg32_${EVAL_LABEL}.txt"
LOG_FILE="${RESULTS_DIR}/aime24_avg32_${EVAL_LABEL}.log"

# ── Step 1: Kill existing vLLM, launch with target model ──
log "killing existing vLLM..."
pkill -f "vllm serve" 2>/dev/null || true
ray stop -f 2>/dev/null || true
sleep 3

log "launching vLLM with model: $MODEL_PATH"
cd "$REPO_ROOT"
source "$VLLM_VENV_DIR/bin/activate"

MODEL_NAME="$MODEL_PATH" \
SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
PORT="$PORT" \
TP_SIZE=4 \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
MAX_NUM_SEQS=16 \
MAX_NUM_BATCHED_TOKENS=65536 \
nohup bash scripts/run_vllm_server.sh > /tmp/vllm_eval_${EVAL_LABEL}.log 2>&1 &
VLLM_PID=$!

log "waiting for vLLM to become healthy (PID=$VLLM_PID)..."
for i in $(seq 1 90); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    log "vLLM healthy after ${i}0s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    log "ERROR: vLLM process died"
    exit 1
  fi
  if [ "$i" = "90" ]; then
    log "ERROR: vLLM not healthy after 900s"
    exit 1
  fi
  sleep 10
done

# ── Step 2: Run avg@32 eval ──
log "running AIME24 avg@32 eval: ${NUM_ROLLOUTS} samples, temp=${TEMPERATURE}, max_tokens=${MAX_TOKENS}"

python3 -u - "$EVAL_LABEL" <<'EVAL_SCRIPT' 2>&1 | tee "$LOG_FILE"
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

LABEL = sys.argv[1] if len(sys.argv) > 1 else "eval"
PORT = int(os.environ.get("PORT", "8000"))
MODEL = os.environ.get("SERVED_MODEL_NAME", "eval_model")
NUM_ROLLOUTS = int(os.environ.get("NUM_ROLLOUTS", "32"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16000"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/tmp")

# Load AIME 2024
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
problems = list(ds)
print(f"[{LABEL}] Loaded {len(problems)} AIME 2024 problems, generating {NUM_ROLLOUTS} samples each")

def extract_answer(text):
    # Extract from \boxed{...}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        ans = matches[-1].strip()
        try:
            return str(int(float(ans)))
        except (ValueError, OverflowError):
            return ans
    # Fallback: last number
    nums = re.findall(r'-?\d+', text)
    return nums[-1] if nums else ""

def generate_samples(problem_idx, prompt, n):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "n": n,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    resp = requests.post(
        f"http://localhost:{PORT}/v1/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=1800,
    )
    resp.raise_for_status()
    return resp.json()["choices"]

# Build prompts with thinking enabled
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(os.environ.get("MODEL_PATH", MODEL), trust_remote_code=True)
def make_prompt(problem_text):
    messages = [
        {"role": "system", "content": "Solve this math problem. Show your reasoning and put your final answer in \\boxed{}."},
        {"role": "user", "content": problem_text},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

results = []
t0 = time.time()
for i, prob in enumerate(problems):
    gold = str(prob["answer"]).strip()
    prompt = make_prompt(prob["problem"])
    try:
        choices = generate_samples(i, prompt, NUM_ROLLOUTS)
    except Exception as e:
        print(f"  [{i+1}/{len(problems)}] ERROR: {e}")
        results.append({"problem_idx": i, "gold": gold, "correct": 0, "total": NUM_ROLLOUTS, "avg": 0.0})
        continue

    correct = 0
    boxed_count = 0
    finish_counts = {}
    for c in choices:
        text = c["text"]
        fr = c.get("finish_reason", "unknown")
        finish_counts[fr] = finish_counts.get(fr, 0) + 1
        ans = extract_answer(text)
        if ans:
            boxed_count += 1
        if ans == gold:
            correct += 1

    avg = correct / NUM_ROLLOUTS
    elapsed = time.time() - t0
    print(f"  [{i+1}/{len(problems)}] gold={gold} correct={correct}/{NUM_ROLLOUTS} ({avg*100:.1f}%) "
          f"boxed={boxed_count} finish={finish_counts} elapsed={elapsed:.0f}s")
    results.append({"problem_idx": i, "gold": gold, "correct": correct, "total": NUM_ROLLOUTS, "avg": avg})

# Summary
overall_avg = sum(r["avg"] for r in results) / len(results) * 100
print(f"\n{'='*60}")
print(f"[{LABEL}] AIME24 avg@{NUM_ROLLOUTS} = {overall_avg:.1f}%")
print(f"{'='*60}")

# Save results
results_file = f"{RESULTS_DIR}/aime24_avg32_{LABEL}.txt"
with open(results_file, "w") as f:
    f.write(f"AIME24 avg@{NUM_ROLLOUTS} = {overall_avg:.1f}%\n")
    f.write(f"Temperature: {TEMPERATURE}, Max tokens: {MAX_TOKENS}\n")
    for r in results:
        f.write(f"  P{r['problem_idx']+1}: {r['correct']}/{r['total']} ({r['avg']*100:.1f}%) gold={r['gold']}\n")

json_file = f"{RESULTS_DIR}/aime24_avg32_{LABEL}.json"
with open(json_file, "w") as f:
    json.dump({"label": LABEL, "avg_at_32": overall_avg, "results": results}, f, indent=2)
print(f"Results saved to {results_file} and {json_file}")
EVAL_SCRIPT

log "eval complete! Results: $RESULTS_FILE"
