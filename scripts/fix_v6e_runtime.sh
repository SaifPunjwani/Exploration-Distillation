#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-gcr.io/cloud-tpu-v2-images/grpc_tpu_worker:tf-2.15.0-pjrt}"

meta() {
  local key="$1"
  curl -sf -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/${key}"
}

worker_id="$(meta agent-worker-number)"
host_ip="$(curl -sf -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip')"
worker_network_endpoints="$(meta worker-network-endpoints)"
mesh_controller_address="$(printf '%s' "$worker_network_endpoints" | cut -d, -f1 | awk -F: '{print $3}')":8476

echo "[v6e-fix] worker_id=${worker_id} host_ip=${host_ip} mesh_controller_address=${mesh_controller_address}"

sudo systemctl stop tpu-runtime.service || true
sudo docker rm -f tpu-runtime >/dev/null 2>&1 || true

sudo docker run -d \
  --net=host \
  --name=tpu-runtime \
  --oom-kill-disable=true \
  --oom-score-adj=-1000 \
  -v /tmp:/tmp \
  --user=2000:2000 \
  --ulimit=memlock=68719476736 \
  --privileged=true \
  -e TPU_STDERR_LOG_LEVEL=0 \
  -e CLOUD_TPU_TASK_ID="${worker_id}" \
  -e TPU_LOCK_DEVICE=true \
  -e TPU_MESH_CONTROLLER_ADDRESS="${mesh_controller_address}" \
  -e TPU_MESH_CONTROLLER_PORT=8476 \
  -e TPU_HOSTNAME_OVERRIDE="${host_ip}" \
  "${IMAGE}" \
  --tpu_hostname_override="${host_ip}" \
  --envelope_enabled=false \
  --specify_env_vars_manually

sleep 5
echo "[v6e-fix] docker ps"
sudo docker ps --filter name=tpu-runtime
echo "[v6e-fix] docker logs"
sudo docker logs --tail 80 tpu-runtime || true
