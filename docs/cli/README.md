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

支持三种卸载方式：

```bash
python3 cli.py unload --models demo_model
python3 cli.py unload --versions demo_model@3
python3 cli.py unload --aliases model_demo_model_abcdef12
```

### `reload`

重载一个或多个模型：

```bash
python3 cli.py reload demo_model another_model
```

## 运行前提

- Controller 容器或本地环境必须能访问 Kubernetes API。
- `MODEL_TARGET_PATH` 对应的路径应与 Triton Repository PVC 保持一致。
- 如果 `MODEL_TARGET_PATH` 指向类似 `/repository/trt_models` 的子目录，建议同时设置：

```bash
export HOT_TRITON_STATE_FILE=/repository/.hot_loader/state.json
export HOT_TRITON_STAGING_ROOT=/repository/.staging
```

- 这样 controller 自己的 `.hot_loader/` 和 `.staging/` 不会落进 Triton model store。
- model-copy Job 会自动把 PVC 挂到 `MODEL_TARGET_PATH` 的父目录，例如目标是 `/repository/trt_models` 时，Job 实际挂载点是 `/repository`。
- 如需任务完成后立即自动删除，可设置 `JOB_TTL_SECONDS_AFTER_FINISHED=0`；这也是当前示例部署的默认值。
- Triton 必须启用：

```bash
tritonserver \
  --model-repository=/repository/trt_models \
  --model-control-mode=EXPLICIT \
  --repository-poll-secs=0
```

## 测试

```bash
python3 -m unittest tests.test_hot_loader tests.test_server
```
