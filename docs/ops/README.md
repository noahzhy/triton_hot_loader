# 运维简要配置与说明

## 1. 用途

Controller 负责:

1. 创建 Kubernetes model-copy Job
2. 把模型从镜像复制到 Triton Repository PVC
3. 调用 Triton API 做 `load` / `reload`

当前实现不依赖 `docker.sock`。

## 2. 部署前提

- Controller 能访问 Kubernetes API
- Controller 能访问 Triton HTTP 地址
- Triton 必须这样启动:

```bash
tritonserver \
  --model-control-mode=EXPLICIT \
  --repository-poll-secs=0
```

- 状态文件和 staging 目录不要放进 Triton model store

## 3. 必配参数

推荐用 `.env` 或环境变量:

```env
TRITON_URL=http://127.0.0.1:8000
HOT_TRITON_MODEL_REPOSITORY=/shared-volume/trt_models
HOT_TRITON_STATE_FILE=/shared-volume/.hot_loader/state.json
HOT_TRITON_STAGING_ROOT=/shared-volume/.staging
MODEL_SOURCE_PATH=/trt_models
MODEL_TARGET_PATH=/repository/trt_models
TRITON_REPOSITORY_PVC=triton-models-storage
K8S_NAMESPACE=default
```

为什么这里需要 `TRITON_REPOSITORY_PVC`，而不是只用临时存储路径:

- model-copy 是单独创建出来的 Kubernetes Job，它和 Triton/Controller 不是同一个 Pod
- `emptyDir` 这类临时目录只在单个 Pod 内可见，Job Pod 不能直接把文件写到 Triton Pod 的临时目录
- PVC 是 Job Pod、Controller Pod、Triton Pod 之间可共享的落盘介质，Job 复制完成后，其它组件才能看到同一份模型文件
- 如果只放临时目录，Pod 重建或漂移后模型文件会丢失，Controller 的状态和实际模型目录会不一致
- 现在的推荐模式是:
  - PVC 负责“跨 Pod 共享”和“稳定落盘”
  - `HOT_TRITON_MODEL_REPOSITORY` 指向临时目录时，负责给 Triton 提供在线读取路径
  - Controller 在 Job 完成后，再把 PVC 中的新模型同步到临时目录

可以把 PVC 理解成 Job 的交付面，把临时目录理解成 Triton 的运行面。两者职责不同，不是重复配置。

常用可选项:

- `JOB_TTL_SECONDS_AFTER_FINISHED=0`
- `JOB_BACKOFF_LIMIT=1`
- `MODEL_COPY_CPU_REQUEST=100m`
- `MODEL_COPY_MEMORY_REQUEST=256Mi`
- `MODEL_COPY_CPU_LIMIT=1`
- `MODEL_COPY_MEMORY_LIMIT=1Gi`
- `MAX_CONCURRENT_JOBS=0`
- `JOB_TOLERATIONS_JSON=[...]`
- `REPOSITORY_MAINTENANCE_IMAGE=...`

完整示例见 [controller.env.example](./controller.env.example)。

## 4. 启动与检查

启动:

```bash
python3 cli.py serve --host 0.0.0.0 --port 8090
```

巡检接口:

- `GET /healthz`
- `GET /runtime/health`
- `GET /api/status`
- `GET /api/jobs/{job_name}`

## 5. 常用命令

加载模型:

```bash
curl -X POST http://127.0.0.1:8090/api/models/load \
  -H 'Content-Type: application/json' \
  -d '{"image":"ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"}'
```

看状态:

```bash
curl http://127.0.0.1:8090/api/status
```

看日志:

```bash
kubectl logs -n default deployment/trtis-deployment-realtime-dev -c triton-controller
```

## 6. 排障入口

- `load/unload` 被 Triton 拒绝:
  检查 `--model-control-mode=EXPLICIT` 和 `--repository-poll-secs=0`
- Job 卡在 `IMAGE_PULLING` 或 `SCHEDULING`:
  运行 `kubectl describe pod` 和 `kubectl get events`
- 卸载时报缺辅助镜像:
  补 `REPOSITORY_MAINTENANCE_IMAGE`
- Triton 仓库里出现 `.hot_loader` 或 `.staging`:
  把状态文件和 staging 目录移到模型目录外层

相关文档:

- [根 README](../../README.md)
- [CLI 使用说明](../cli/README.md)
- [realtime-dev 部署说明](../../deploy/realtime-dev/README.md)
