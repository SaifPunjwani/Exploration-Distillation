#!/usr/bin/env bash
# Thin wrapper around scripts/tmx_submit.py for the XPK backend.
#
# Required env:
#   SPEC             — path to a config JSON (e.g. configs/production_*.json)
# Optional env:
#   EXECUTE          — default 0 (dry-run); set 1 to actually submit
#   IMAGE            — override backend.container_image; defaults to whatever
#                      docker/maxtext-tunix/.last_image holds
set -euo pipefail

SPEC="${SPEC:?set SPEC=path/to/spec.json}"
EXECUTE="${EXECUTE:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [ ! -f "${SPEC}" ]; then
    echo "[xpk-submit] ERROR: ${SPEC} not found" >&2
    exit 2
fi

# Resolve image. Operator can override with IMAGE=...; otherwise read the
# last-built tag from the build_and_push.sh side-channel.
IMAGE="${IMAGE:-}"
if [ -z "${IMAGE}" ] && [ -f docker/maxtext-tunix/.last_image ]; then
    IMAGE="$(tr -d '\n' < docker/maxtext-tunix/.last_image)"
fi
if [ -z "${IMAGE}" ]; then
    echo "[xpk-submit] ERROR: no IMAGE and docker/maxtext-tunix/.last_image is empty" >&2
    echo "[xpk-submit]        run docker/maxtext-tunix/build_and_push.sh first" >&2
    exit 2
fi

# Patch the spec JSON in-place to point at the resolved image, but write to
# a tmp file rather than mutating the operator's source-of-truth config.
TMP_SPEC="$(mktemp -t tmx-xpk-spec.XXXXXX.json)"
trap 'rm -f "${TMP_SPEC}"' EXIT

python3 - "${SPEC}" "${IMAGE}" "${TMP_SPEC}" <<'PY'
import json
import sys

src, image, dst = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as fh:
    spec = json.load(fh)
spec.setdefault("backend", {})["container_image"] = image
spec["backend"].setdefault("experimental_adapter_ack", True)
with open(dst, "w") as fh:
    json.dump(spec, fh, indent=2, sort_keys=True)
print(f"[xpk-submit] image={image}")
print(f"[xpk-submit] tmp_spec={dst}")
PY

ARGS=(submit --spec "${TMP_SPEC}" --backend xpk-gke-maxtext-tunix)
if [ "${EXECUTE}" = "1" ]; then
    ARGS+=(--execute)
else
    ARGS+=(--dry-run)
fi

python3 scripts/tmx_submit.py "${ARGS[@]}"
