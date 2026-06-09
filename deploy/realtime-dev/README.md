# realtime-dev 部署说明

先应用 RBAC：

```bash
kubectl apply -f deploy/realtime-dev/triton-controller-rbac.yaml
```

再更新 Deployment：

```bash
kubectl apply -f deploy/realtime-dev/trtis-deployment-realtime-dev.yaml
```

检查 ServiceAccount 是否生效：

```bash
kubectl get deployment trtis-deployment-realtime-dev -n default -o jsonpath='{.spec.template.spec.serviceAccountName}'
```

检查权限：

```bash
kubectl auth can-i list jobs.batch --as=system:serviceaccount:default:triton-controller-sa -n default
```

检查 controller 日志：

```bash
kubectl logs -n default deployment/trtis-deployment-realtime-dev -c triton-controller
```

如果 `GET /models/jobs/{job_name}` 长时间停在 `IMAGE_PULLING` 或 `SCHEDULING`，优先检查 Job Pod 事件：

```bash
kubectl describe pod -n default <job-pod-name>
kubectl get events -n default --field-selector involvedObject.name=<job-pod-name>
```

这份 Deployment 已做的关键变更：

- 增加 `serviceAccountName: triton-controller-sa`
- 删除 `docker.sock` 挂载
- 删除 controller 对 `/var/run/docker.sock` 的依赖
- 增加新的 controller 环境变量：
  - `HOT_TRITON_MODEL_REPOSITORY=/repository/trt_models`
  - `HOT_TRITON_STATE_FILE=/repository/.hot_loader/state.json`
  - `HOT_TRITON_STAGING_ROOT=/repository/.staging`
  - `MODEL_SOURCE_PATH=/trt_models`
  - `MODEL_TARGET_PATH=/repository/trt_models`
  - `TRITON_REPOSITORY_PVC=triton-models-storage`
  - `JOB_TOLERATIONS_JSON=[...]`
  - `JOB_TTL_SECONDS_AFTER_FINISHED=0`
  - `TRITON_URL=http://127.0.0.1:8000`
  - `TRITON_METRICS_URL=http://127.0.0.1:8002/metrics`
