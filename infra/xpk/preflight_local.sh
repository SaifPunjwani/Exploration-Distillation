#!/usr/bin/env bash
# Run all the offline checks before going to the TPU.
# Idempotent. Safe to run anywhere — does NOT touch the TPU.
#
#   1. py_compile (syntax)
#   2. ruff check (style + imports + bugbear)
#   3. vulture (dead code) — best-effort, skips if not installed
#   4. pytest on the orchestration tests
#   5. spec.validate() on every config under configs/tunix/
#   6. dry-run command emission for the resume spec
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[preflight] step 1/6: py_compile"
PY_FILES=(
    tmx/orchestration/optimization.py
    tmx/orchestration/topologies_extra.py
    tmx/orchestration/tunix_rewards.py
    tmx/orchestration/tunix_config.py
    tmx/orchestration/tunix_workload.py
    tmx/orchestration/tunix_main.py
    tmx/orchestration/tunix_learner.py
    tmx/orchestration/novelty_state.py
    tmx/orchestration/novelty_features.py
    tmx/orchestration/novelty_rollout.py
    tests/test_optimization.py
    tests/test_topologies_extra.py
    tests/test_tunix_rewards.py
    tests/test_tunix_config.py
    tests/test_tunix_workload.py
    tests/test_tunix_learner.py
    tests/test_tunix_configs.py
    tests/test_novelty_state.py
    tests/test_novelty_features.py
    tests/test_novelty_reward.py
)
python3 -m py_compile "${PY_FILES[@]}"

echo "[preflight] step 2/6: ruff"
if command -v ruff >/dev/null 2>&1 || python3 -m ruff --version >/dev/null 2>&1; then
    python3 -m ruff check "${PY_FILES[@]}"
else
    echo "[preflight] WARN ruff not installed; skipping"
fi

echo "[preflight] step 3/6: vulture (dead code)"
if python3 -m vulture --version >/dev/null 2>&1; then
    python3 -m vulture "${PY_FILES[@]}" \
        --min-confidence 80 \
        --ignore-names "main,describe,run,cmd_*,test_*,register_extras,emit_command,emit_overrides,correctness,incorrectness_penalty"
else
    echo "[preflight] WARN vulture not installed; skipping (pip install vulture)"
fi

echo "[preflight] step 4/6: pytest (orchestration + novelty suite)"
# On the local macOS/conda env, pytest's capture plugin can segfault before
# collection. Disabling capture keeps the preflight deterministic; TPU-side
# gates still run normally in their own env.
PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}" \
python3 -m pytest -p no:capture \
    tests/test_orchestration.py \
    tests/test_optimization.py \
    tests/test_topologies_extra.py \
    tests/test_tunix_rewards.py \
    tests/test_tunix_config.py \
    tests/test_tunix_workload.py \
    tests/test_tunix_learner.py \
    tests/test_tunix_configs.py \
    tests/test_novelty_state.py \
    tests/test_novelty_features.py \
    tests/test_novelty_reward.py \
    -q

echo "[preflight] step 5/6: spec validation (configs/tunix/*)"
shopt -s nullglob
for spec in configs/tunix/*.json; do
    echo "[preflight] - validating ${spec}"
    python3 scripts/tmx_submit.py validate --spec "${spec}" >/dev/null
done

echo "[preflight] step 6/6: tunix adapter emission/describe"
SMOKE_SPEC="configs/tunix/tmx_smoke_tiny.json"
if [ -f "${SMOKE_SPEC}" ]; then
    python3 -m tmx.orchestration.tunix_config emit \
        --spec "${SMOKE_SPEC}" \
        --format command >/dev/null
    echo "[preflight] - correctness-only Tunix CLI command emits cleanly"
fi
RESUME_SPEC="configs/tunix/tmx_dapo_drgrpo_resume_step100.json"
if [ -f "${RESUME_SPEC}" ]; then
    SPEC_B64="$(python3 -c "
import base64, json, sys
spec = json.load(open(sys.argv[1]))
print(base64.b64encode(json.dumps(spec, separators=(',', ':')).encode()).decode())
" "${RESUME_SPEC}")"
    TMX_EXPERIMENT_SPEC_JSON_B64="${SPEC_B64}" \
        python3 -m tmx.orchestration.tunix_workload describe >/dev/null
    echo "[preflight] - novelty resume describes as Tunix/MaxText programmatic adapter"
fi

echo "[preflight] OK"
