# CLI 使用说明

## 概览

`cli.py` 现在对应的是 Kubernetes Job 版 Triton 控制器，不再支持旧的 Docker 解包式热加载。

入口：

```bash
python3 cli.py --help
```

## 公共参数

所有子命令共享以下运行时参数：

- `--triton-url`
- `--triton-metrics-url`
- `--model-repository`
- `--state-file`
- `--staging-root`
- `--model-source-path`
- `--model-target-path`
- `--triton-repository-pvc`
- `--k8s-namespace`
- `--model-image-registry-prefix`
- `--job-ttl-seconds-after-finished`
- `--job-backoff-limit`
- `--model-copy-cpu-request`
- `--model-copy-memory-request`
- `--model-copy-cpu-limit`
- `--model-copy-memory-limit`
- `--max-concurrent-jobs`
- `--request-timeout`

补充环境变量：

- `JOB_TOLERATIONS_JSON`
  - 通过 JSON 数组给动态创建的 model-copy Job 注入 Kubernetes tolerations
  - 例如：

```json
[{"key":"gpu","operator":"Exists","effect":"NoSchedule"},{"key":"cpu","operator":"Equal","value":"cveng","effect":"NoSchedule"}]
```

## 子命令

### `serve`

启动 Web UI 和 API：

```bash
python3 cli.py serve --host 0.0.0.0 --port 8090
```

### `load`

创建单个模型的复制 Job：

```bash
python3 cli.py load \
  --image ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430
```

如果不传 `--model-name`，controller 会根据 image tag 自动提取；需要覆盖默认推导时再显式传入。

说明：

- `load` 只负责提交 model-copy Job，本身不会像 HTTP `wait_for_ready=true` 那样阻塞等待终态。
- 模型文件复制完成后，controller 会自动触发 Triton `load/reload`；可以配合 `job-status` 或 `status` 观察后续状态推进。

### `load-batch`

批量创建复制 Job：

```bash
python3 cli.py load-batch --file batch.json
```

`batch.json` 示例：

```json
{
  "models": [
    {
      "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
    },
    {
      "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620"
    }
  ]
}
```

说明：

- `load-batch` 也是异步提交；返回后需要通过 `job-status`、`status` 或 Web UI 继续观察。

### `job-status`

查询单个 Job 的复制与 Triton 重载状态：

```bash
python3 cli.py job-status model-copy-demo-12345678
```

### `status`

查看 Triton、GPU 指标和 controller 状态：

```bash
python3 cli.py status
```

### `list`

查看已管理模型和 Triton repository/index：

```bash
python3 cli.py list
```

### `unload`

支持两种卸载方式：

```bash
python3 cli.py unload --models demo_model
python3 cli.py unload --aliases model_demo_model
```

说明：

- 同名模型现在总是直接替换，不再支持 `model_name@version` 级别的版本卸载。

### `reload`

重载一个或多个模型：

```bash
python3 cli.py reload demo_model another_model
```

### 同名模型替换

当新的加载请求解析出与现有模型相同的 `model_name` 时，controller 会直接替换仓库中的同名模型目录，并把状态里的当前镜像更新为最新值；不会再保留同名模型的历史版本记录。

当前实现不会长时间直接覆盖线上目录，而是先复制到挂载卷内的 `.staging/`，再切换到目标目录，避免 Triton 在替换窗口读到半成品模型目录。

## 运行前提

- Controller 容器或本地环境必须能访问 Kubernetes API。
- `MODEL_TARGET_PATH` 对应的路径应与 Triton Repository PVC 保持一致。
- 如果 Triton 的在线模型仓库必须放在临时目录，请把 `HOT_TRITON_MODEL_REPOSITORY` 显式设为临时目录路径，并确保它与 `MODEL_TARGET_PATH` 不同；这样 controller 会自动进入 repository sync 模式。
- 如果 `MODEL_TARGET_PATH` 指向类似 `/repository/trt_models` 的子目录，建议同时设置：

```bash
export HOT_TRITON_STATE_FILE=/repository/.hot_loader/state.json
export HOT_TRITON_STAGING_ROOT=/repository/.staging
```

- 这样 controller 自己的 `.hot_loader/` 和 `.staging/` 不会落进 Triton model store。
- model-copy Job 会自动把 PVC 挂到 `MODEL_TARGET_PATH` 的父目录，例如目标是 `/repository/trt_models` 时，Job 实际挂载点是 `/repository`。
- Triton 临时目录 + PVC 同步模式下建议同时设置：

```bash
export HOT_TRITON_MODEL_REPOSITORY=/shared-volume/trt_models
export HOT_TRITON_STATE_FILE=/shared-volume/.hot_loader/state.json
export HOT_TRITON_STAGING_ROOT=/shared-volume/.staging
export MODEL_TARGET_PATH=/repository/trt_models
export TRITON_REPOSITORY_PVC=triton-repository-pvc
export REPOSITORY_MAINTENANCE_IMAGE=ccr.ccs.tencentyun.com/clobotics/triton-hot-loader:latest
```

- 这种模式下 controller 需要同时挂载临时目录和 `/repository` 对应的 PVC 路径；Triton 只需要挂载临时目录。
- 其中 `REPOSITORY_MAINTENANCE_IMAGE` 只在 controller 看不到 PVC 挂载点时，才会退回到 cleanup Job 使用。
- `--max-concurrent-jobs` 或 `MAX_CONCURRENT_JOBS` 只有在设置为正整数时才会限流；`0` 表示不限制，这也是当前默认值。
- 如需任务完成后立即自动删除，可设置 `JOB_TTL_SECONDS_AFTER_FINISHED=0`；这也是当前示例部署的默认值。
- 即使 Job 已被 TTL 删除，只要模型目录已经复制完成，controller 仍会继续自动推进 Triton load/reload。
- 调试调度、拉镜像或复制问题时，建议临时把 `JOB_TTL_SECONDS_AFTER_FINISHED` 调大到 `300`，这样 Job / Pod / Event 不会立刻消失。
- Triton 必须启用：

```bash
tritonserver \
  --model-repository=/shared-volume/trt_models \
  --model-control-mode=EXPLICIT \
  --repository-poll-secs=0
```

## 测试

```bash
python3 -m unittest tests.test_hot_loader tests.test_server
```
