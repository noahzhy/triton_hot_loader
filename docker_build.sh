#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ccr.ccs.tencentyun.com/clobotics/triton-hot-loader}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
FULL_TAG="${IMAGE_NAME}:${TIMESTAMP}"
LATEST_TAG="${IMAGE_NAME}:latest"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
PYTHON_BASE_IMAGE="${PYTHON_BASE_IMAGE:-docker.m.daocloud.io/library/python:3.11-slim}"
APT_MIRROR="${APT_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/debian}"
APT_SECURITY_MIRROR="${APT_SECURITY_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn/debian-security}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

BUILD_OUTPUT_ARGS=()
HAS_EXPLICIT_OUTPUT=0
IS_MULTI_PLATFORM=0

if [[ "${DOCKER_PLATFORM}" == *,* ]]; then
    IS_MULTI_PLATFORM=1
fi

for arg in "$@"; do
    case "${arg}" in
        --load|--push|--output|-o)
            HAS_EXPLICIT_OUTPUT=1
            break
            ;;
    esac
done

if [[ ${HAS_EXPLICIT_OUTPUT} -eq 0 ]]; then
    if [[ ${IS_MULTI_PLATFORM} -eq 1 ]]; then
        echo "[hot_triton] multi-platform builds cannot use the default docker exporter"
        echo "[hot_triton] rerun with --push or another explicit --output, or set DOCKER_PLATFORM to a single platform"
        exit 1
    fi

    BUILD_OUTPUT_ARGS=(--load)
fi

echo "[hot_triton] building ${FULL_TAG}"
echo "[hot_triton] target platform: ${DOCKER_PLATFORM}"
echo "[hot_triton] python base image: ${PYTHON_BASE_IMAGE}"
echo "[hot_triton] apt mirror: ${APT_MIRROR}"
echo "[hot_triton] pip index: ${PIP_INDEX_URL}"

BUILD_CMD=(
    docker buildx build
    "$@"
    --platform "${DOCKER_PLATFORM}"
    --build-arg "PYTHON_BASE_IMAGE=${PYTHON_BASE_IMAGE}"
    --build-arg "APT_MIRROR=${APT_MIRROR}"
    --build-arg "APT_SECURITY_MIRROR=${APT_SECURITY_MIRROR}"
    --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}"
    --build-arg "PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}"
    -f "${SCRIPT_DIR}/Dockerfile"
    -t "${FULL_TAG}"
    -t "${LATEST_TAG}"
    "${SCRIPT_DIR}"
)

if [[ ${#BUILD_OUTPUT_ARGS[@]} -gt 0 ]]; then
    BUILD_CMD=("${BUILD_CMD[@]:0:3}" "${BUILD_OUTPUT_ARGS[@]}" "${BUILD_CMD[@]:3}")
fi

"${BUILD_CMD[@]}"

echo "[hot_triton] build completed: ${FULL_TAG}"
echo "[hot_triton] latest tag updated: ${LATEST_TAG}"
