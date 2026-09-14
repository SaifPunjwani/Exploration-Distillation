#!/usr/bin/env bash
# Build and push the MaxText/Tunix TPU image to Artifact Registry.
#
# Required env:
#   PROJECT_ID       — GCP project (e.g. my-gcp-project)
# Optional env:
#   REGION           — Artifact Registry region (default: us-central1)
#   ARTIFACT_REGISTRY_REPO — repo name (default: tmx-images)
#   TAG              — image tag (default: short git sha or timestamp)
#   VLLM_VERSION     — pin vllm at this version (default: latest)
#
# Output: writes the resulting image URI to docker/maxtext-tunix/.last_image
# and stdout.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
REPO="${ARTIFACT_REGISTRY_REPO:-tmx-images}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${TAG:-$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d-%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/tmx-maxtext-tunix:${TAG}"
DOCKERFILE="${ROOT_DIR}/docker/maxtext-tunix/Dockerfile.tpu"

cd "${ROOT_DIR}"

echo "[build] image=${IMAGE}"
echo "[build] dockerfile=${DOCKERFILE}"
echo "[build] context=${ROOT_DIR}"

if [ ! -f "${DOCKERFILE}" ]; then
    echo "[build] ERROR: ${DOCKERFILE} not found" >&2
    exit 2
fi

# Configure docker auth for the region's Artifact Registry. Idempotent.
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Ensure the repo exists. Skipping create when it already does.
if ! gcloud --project="${PROJECT_ID}" artifacts repositories describe "${REPO}" \
        --location="${REGION}" >/dev/null 2>&1; then
    echo "[build] creating Artifact Registry repo ${REPO} in ${REGION}"
    gcloud --project="${PROJECT_ID}" artifacts repositories create "${REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --description="TMX TPU images"
fi

DOCKER_BUILD_ARGS=()
if [ -n "${VLLM_VERSION:-}" ]; then
    DOCKER_BUILD_ARGS+=(--build-arg "VLLM_VERSION=${VLLM_VERSION}")
fi

docker build \
    --platform linux/amd64 \
    -f "${DOCKERFILE}" \
    -t "${IMAGE}" \
    "${DOCKER_BUILD_ARGS[@]}" \
    .

docker push "${IMAGE}"

echo "${IMAGE}" > "${ROOT_DIR}/docker/maxtext-tunix/.last_image"
echo "[build] OK ${IMAGE}"
echo "[build] saved to docker/maxtext-tunix/.last_image"
