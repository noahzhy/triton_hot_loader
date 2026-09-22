#!/usr/bin/env bash
set -euo pipefail

namespace="triton-job-test"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kubectl_cmd=(kubectl --namespace "${namespace}")

kubectl apply -f "${root_dir}/namespace.yaml"
kubectl apply -f "${root_dir}/pvc.yaml"
kubectl apply -f "${root_dir}/write-job.yaml"
"${kubectl_cmd[@]}" wait --for=jsonpath='{.status.phase}'=Bound pvc/job-test-workspace --timeout=120s
"${kubectl_cmd[@]}" wait --for=condition=complete job/job-write-marker --timeout=120s
write_marker="$("${kubectl_cmd[@]}" logs job/job-write-marker | sed -n 's/^MARKER=//p' | tail -n 1)"
test -n "${write_marker}"

kubectl apply -f "${root_dir}/read-job.yaml"
"${kubectl_cmd[@]}" wait --for=condition=complete job/job-read-marker --timeout=120s
read_marker="$("${kubectl_cmd[@]}" logs job/job-read-marker | sed -n 's/^MARKER=//p' | tail -n 1)"
test "${write_marker}" = "${read_marker}"

kubectl apply -f "${root_dir}/failure-jobs.yaml"
"${kubectl_cmd[@]}" wait --for=condition=failed job/job-wrong-mount-path --timeout=120s

image_pull_failed=false
for _ in $(seq 1 24); do
    reason="$("${kubectl_cmd[@]}" get pods -l job-name=job-image-pull-failure \
        -o jsonpath='{.items[0].status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)"
    if [[ "${reason}" == "ImagePullBackOff" || "${reason}" == "ErrImagePull" ]]; then
        image_pull_failed=true
        break
    fi
    sleep 5
done
if [[ "${image_pull_failed}" != true ]]; then
    echo "The unavailable-image Job did not reach an image-pull failure state in time." >&2
    exit 1
fi

"${kubectl_cmd[@]}" get pods,jobs,pvc
"${kubectl_cmd[@]}" get events --sort-by=.lastTimestamp
"${kubectl_cmd[@]}" describe job/job-image-pull-failure
"${kubectl_cmd[@]}" describe job/job-wrong-mount-path

"${kubectl_cmd[@]}" wait --for=delete job/job-write-marker --timeout=120s
"${kubectl_cmd[@]}" wait --for=delete job/job-read-marker --timeout=120s
"${kubectl_cmd[@]}" wait --for=jsonpath='{.status.phase}'=Bound pvc/job-test-workspace --timeout=30s

echo "Write/read marker validated: ${write_marker}"
echo "ImagePullBackOff and nonzero-exit diagnostics validated."
echo "Completed write/read Jobs were deleted by TTL; the PVC remains Bound."
