#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_IMAGE_REF="${IMAGE_NAME:-ccr.ccs.tencentyun.com/clobotics/triton-hot-loader:latest}"
IMAGE_REF="${IMAGE_REF:-${DEFAULT_IMAGE_REF}}"
CONTAINER_NAME="${CONTAINER_NAME:-triton-hot-loader}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8090}"
OS_NAME="$(uname -s)"
DEFAULT_NETWORK_MODE="host"
if [[ "${OS_NAME}" == "Darwin" ]]; then
  DEFAULT_NETWORK_MODE="bridge"
fi
NETWORK_MODE="${NETWORK_MODE:-${DEFAULT_NETWORK_MODE}}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
RUNTIME_DIR="${RUNTIME_DIR:-${SCRIPT_DIR}/runtime}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

if [[ $# -gt 0 ]] && [[ "${1}" != -* ]]; then
  IMAGE_REF="${1}"
  shift
fi

mkdir -p "${RUNTIME_DIR}/model_repository" "${RUNTIME_DIR}/staging"

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
    ;;
  bridge)
    docker_args+=(
      --add-host host.docker.internal:host-gateway
      -p "${APP_PORT}:${APP_PORT}"
    )
    ACCESS_URL="http://127.0.0.1:${APP_PORT}"
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
