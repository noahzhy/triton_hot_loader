#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_REF="${IMAGE_REF:-${IMAGE_NAME:-ccr.ccs.tencentyun.com/clobotics/triton-hot-loader:latest}}"
CONTAINER_NAME="${CONTAINER_NAME:-triton-hot-loader}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8090}"
NETWORK_MODE="${NETWORK_MODE:-host}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
RUNTIME_DIR="${RUNTIME_DIR:-${SCRIPT_DIR}/runtime}"
DOCKER_SOCKET="${DOCKER_SOCKET:-/var/run/docker.sock}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

mkdir -p "${RUNTIME_DIR}/model_repository" "${RUNTIME_DIR}/staging"

if [[ ! -S "${DOCKER_SOCKET}" ]]; then
  echo "[hot_triton] Docker socket 不存在: ${DOCKER_SOCKET}" >&2
  echo "[hot_triton] 请确认宿主机 Docker 已启动，并且 socket 已挂载。" >&2
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  echo "[hot_triton] env file: ${ENV_FILE}"
else
  echo "[hot_triton] warning: env file 不存在，将使用镜像内默认配置: ${ENV_FILE}" >&2
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "[hot_triton] removing existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker_args=(
  run -d
  --name "${CONTAINER_NAME}"
  --restart "${RESTART_POLICY}"
  -v "${RUNTIME_DIR}:/app/runtime"
  -v "${DOCKER_SOCKET}:/var/run/docker.sock"
)

if [[ -f "${ENV_FILE}" ]]; then
  docker_args+=(
    --env-file "${ENV_FILE}"
    -v "${ENV_FILE}:/app/.env:ro"
  )
fi

case "${NETWORK_MODE}" in
  host)
    docker_args+=(--network host)
    ACCESS_URL="http://127.0.0.1:${APP_PORT}"
    echo "[hot_triton] using host network，容器内的 127.0.0.1 会直接指向宿主机。"
    ;;
  bridge)
    docker_args+=(
      --add-host host.docker.internal:host-gateway
      -p "${APP_PORT}:${APP_PORT}"
    )
    ACCESS_URL="http://127.0.0.1:${APP_PORT}"
    if [[ -f "${ENV_FILE}" ]] && grep -Eq '^\s*TRT_IP\s*=\s*127\.0\.0\.1\s*$' "${ENV_FILE}"; then
      echo "[hot_triton] warning: bridge 网络下容器内的 127.0.0.1 指向容器自身；如果 Triton 跑在宿主机，请把 TRT_IP 改成 host.docker.internal，或继续使用 NETWORK_MODE=host。" >&2
    fi
    ;;
  *)
    docker_args+=(
      --network "${NETWORK_MODE}"
      -p "${APP_PORT}:${APP_PORT}"
    )
    ACCESS_URL="http://127.0.0.1:${APP_PORT}"
    ;;
esac

if [[ $# -gt 0 ]]; then
  docker_args+=("$@")
fi

docker_args+=(
  "${IMAGE_REF}"
  python cli.py serve
  --host "${APP_HOST}"
  --port "${APP_PORT}"
)

echo "[hot_triton] image: ${IMAGE_REF}"
echo "[hot_triton] container: ${CONTAINER_NAME}"
echo "[hot_triton] runtime dir: ${RUNTIME_DIR}"
echo "[hot_triton] app port: ${APP_PORT}"

CONTAINER_ID="$(docker "${docker_args[@]}")"

echo "[hot_triton] started: ${CONTAINER_ID}"
docker ps --filter "id=${CONTAINER_ID}" --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}'

echo "[hot_triton] ui: ${ACCESS_URL}"
echo "[hot_triton] health: ${ACCESS_URL}/healthz"
echo "[hot_triton] status: ${ACCESS_URL}/api/status"
echo "[hot_triton] logs: docker logs -f ${CONTAINER_NAME}"
