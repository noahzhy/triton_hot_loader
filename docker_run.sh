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
DOCKER_SOCKET="${DOCKER_SOCKET:-/var/run/docker.sock}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
HOST_DOCKER_CONFIG_DIR="${HOST_DOCKER_CONFIG_DIR:-${DOCKER_CONFIG:-$HOME/.docker}}"
CONTAINER_DOCKER_CONFIG_DIR="/root/.docker"
GENERATED_DOCKER_CONFIG_DIR=""

if [[ $# -gt 0 ]] && [[ "${1}" != -* ]]; then
  IMAGE_REF="${1}"
  shift
fi

mkdir -p "${RUNTIME_DIR}/model_repository" "${RUNTIME_DIR}/staging"

if [[ -f "${HOST_DOCKER_CONFIG_DIR}/config.json" ]]; then
  GENERATED_DOCKER_CONFIG_DIR="${RUNTIME_DIR}/docker-config"
  mkdir -p "${GENERATED_DOCKER_CONFIG_DIR}"
  export HOST_DOCKER_CONFIG_DIR
  export GENERATED_DOCKER_CONFIG_DIR
  python3 <<'PY'
import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

host_config_dir = Path(os.environ["HOST_DOCKER_CONFIG_DIR"])
output_path = Path(os.environ["GENERATED_DOCKER_CONFIG_DIR"]) / "config.json"
config = json.loads((host_config_dir / "config.json").read_text())

result = {"auths": {}}

for registry, auth_cfg in config.get("auths", {}).items():
    auth_value = auth_cfg.get("auth")
    if auth_value:
        result["auths"][registry] = {"auth": auth_value}

def resolve_with_helper(helper_name: str, registry: str):
    helper_bin = shutil.which(f"docker-credential-{helper_name}")
    if not helper_bin:
        return None

    proc = subprocess.run(
        [helper_bin, "get"],
        input=registry.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return None

    payload = json.loads(proc.stdout)
    username = payload.get("Username")
    secret = payload.get("Secret")
    if not username or not secret:
        return None

    token = base64.b64encode(f"{username}:{secret}".encode()).decode()
    return {"auth": token}

cred_helpers = config.get("credHelpers", {})
default_store = config.get("credsStore")

registries = set(config.get("auths", {}).keys()) | set(cred_helpers.keys())
for registry in registries:
    if registry in result["auths"]:
        continue

    helper_name = cred_helpers.get(registry, default_store)
    if not helper_name:
        continue

    resolved = resolve_with_helper(helper_name, registry)
    if resolved:
        result["auths"][registry] = resolved

output_path.write_text(json.dumps(result, indent=2))
PY
fi

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

if [[ -n "${GENERATED_DOCKER_CONFIG_DIR}" && -f "${GENERATED_DOCKER_CONFIG_DIR}/config.json" ]]; then
  docker_args+=(
    -v "${GENERATED_DOCKER_CONFIG_DIR}:${CONTAINER_DOCKER_CONFIG_DIR}:ro"
  )
fi

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
