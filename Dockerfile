ARG DOCKER_CLI_IMAGE=docker.m.daocloud.io/docker:28-cli
ARG PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim

FROM ${DOCKER_CLI_IMAGE} AS docker_cli

FROM ${PYTHON_BASE_IMAGE}

ARG APT_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/debian
ARG APT_SECURITY_MIRROR=http://mirrors.tuna.tsinghua.edu.cn/debian-security
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

WORKDIR /app

COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i \
            -e "s|http://deb.debian.org/debian|${APT_MIRROR}|g" \
            -e "s|https://deb.debian.org/debian|${APT_MIRROR}|g" \
            -e "s|http://security.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
            -e "s|https://security.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
            /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e "s|http://deb.debian.org/debian|${APT_MIRROR}|g" \
            -e "s|https://deb.debian.org/debian|${APT_MIRROR}|g" \
            -e "s|http://security.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
            -e "s|https://security.debian.org/debian-security|${APT_SECURITY_MIRROR}|g" \
            /etc/apt/sources.list; \
    fi; \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
    -i "${PIP_INDEX_URL}" \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    -r requirements.txt

COPY . .

RUN mkdir -p /app/runtime/model_repository /app/runtime/staging

EXPOSE 8090

CMD ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8090"]
