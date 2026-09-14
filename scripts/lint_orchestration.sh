#!/usr/bin/env bash
# Lint + dead-code pass for the production orchestration layer.
#
# Targets the new code (orchestration module, submit/paper scripts, infra),
# not the wider research codebase. Ruff catches style/imports; vulture
# catches dead code; py_compile catches syntax errors. Each tool is
# optional — install pyproject.toml dev tooling first to enable all.
#
# Usage: bash scripts/lint_orchestration.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_TARGETS=(
    "tmx/orchestration"
    "scripts/tmx_submit.py"
    "scripts/tmx_paper.py"
    "tests/test_orchestration.py"
)
SHELL_TARGETS=(
    "infra/xpk"
    "docker/maxtext-tunix"
)

PY_FILES=()
for p in "${PYTHON_TARGETS[@]}"; do
    if [ -d "${p}" ]; then
        while IFS= read -r f; do PY_FILES+=("${f}"); done < <(find "${p}" -name '*.py' -not -path '*/__pycache__/*')
    elif [ -f "${p}" ]; then
        PY_FILES+=("${p}")
    fi
done

SH_FILES=()
for p in "${SHELL_TARGETS[@]}"; do
    if [ -d "${p}" ]; then
        while IFS= read -r f; do SH_FILES+=("${f}"); done < <(find "${p}" -name '*.sh')
    fi
done

echo "[lint] python files: ${#PY_FILES[@]}"
echo "[lint] shell files:  ${#SH_FILES[@]}"

rc=0

if command -v ruff >/dev/null 2>&1; then
    echo "[lint] ruff check"
    ruff check "${PY_FILES[@]}" || rc=$?
else
    echo "[lint] WARN: ruff not installed; skipping (pip install ruff)"
fi

if command -v vulture >/dev/null 2>&1; then
    echo "[lint] vulture (dead code, --min-confidence 80)"
    vulture "${PY_FILES[@]}" --min-confidence 80 \
        --ignore-names "main,describe,run,cmd_*,test_*,register_extras" || rc=$?
else
    echo "[lint] WARN: vulture not installed; skipping (pip install vulture)"
fi

echo "[lint] py_compile"
python3 -m py_compile "${PY_FILES[@]}" || rc=$?

if command -v shellcheck >/dev/null 2>&1; then
    if [ "${#SH_FILES[@]}" -gt 0 ]; then
        echo "[lint] shellcheck"
        shellcheck -x "${SH_FILES[@]}" || rc=$?
    fi
else
    echo "[lint] WARN: shellcheck not installed; skipping shell lint"
fi

if [ "${rc}" -eq 0 ]; then
    echo "[lint] OK"
else
    echo "[lint] FAIL (rc=${rc})"
fi
exit "${rc}"
