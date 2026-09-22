#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo on the target host." >&2
  exit 1
fi

if command -v k3s >/dev/null 2>&1 || systemctl is-active --quiet k3s; then
  echo "Existing K3s installation detected; refusing to overwrite it." >&2
  exit 1
fi

if command -v kubelet >/dev/null 2>&1 || systemctl is-active --quiet kubelet; then
  echo "Existing Kubernetes service detected; refusing to install K3s." >&2
  exit 1
fi

for port in 6443 10250; do
  if ss -ltn "sport = :${port}" | grep -q LISTEN; then
    echo "TCP port ${port} is already in use; refusing to install K3s." >&2
    exit 1
  fi
done

available_kb="$(df -Pk /var/lib | awk 'NR == 2 {print $4}')"
if [[ -z "${available_kb}" || "${available_kb}" -lt 10485760 ]]; then
  echo "At least 10 GiB free under /var/lib is required for this test cluster." >&2
  exit 1
fi

K3S_VERSION="${K3S_VERSION:-v1.37.0+k3s1}"
K3S_INSTALL_SCRIPT_URL="${K3S_INSTALL_SCRIPT_URL:-https://rancher-mirror.rancher.cn/k3s/k3s-install.sh}"

# Public image pulls must work before the BusyBox Jobs can be exercised.  Keep
# an operator-supplied registry configuration intact, but provide the mirror
# used by this isolated test environment on hosts without one.
if [[ ! -e /etc/rancher/k3s/registries.yaml ]]; then
  install -d -m 0755 /etc/rancher/k3s
  cat > /etc/rancher/k3s/registries.yaml <<'EOF'
mirrors:
  docker.io:
    endpoint:
      - "https://docker.m.daocloud.io"
EOF
fi

curl -sfL "${K3S_INSTALL_SCRIPT_URL}" | INSTALL_K3S_MIRROR=cn INSTALL_K3S_VERSION="${K3S_VERSION}" INSTALL_K3S_EXEC='server --write-kubeconfig-mode 644' sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
for _ in $(seq 1 36); do
  if kubectl get nodes -o name 2>/dev/null | grep -q .; then
    break
  fi
  sleep 5
done
kubectl wait --for=condition=Ready node --all --timeout=180s
kubectl get storageclass local-path
