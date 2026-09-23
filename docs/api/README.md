# Triton Hot Loader HTTP API 使用说明

本文按当前 Controller 实现整理。示例优先使用 `/api/...` 路由；浏览器、脚本和 CLI HTTP 调用都应把 Controller 作为管理 API 地址。

## 基础设置与目标实例

```bash
BASE_URL="http://127.0.0.1:8090"
INSTANCE_ID="default"
```

除实例注册接口外，模型、状态、Job、版本操作和指标请求都可以用下面的 header 选择已登记的 Triton 实例；不传时使用默认实例：

```bash
-H "x-hot-triton-instance-id: ${INSTANCE_ID}"
```

示例中为便于阅读省略了部分重复 header。跨实例查询 Job 或版本操作时，要使用任务所属实例的 ID。

旧 header `x-hot-triton-url` 和 `x-hot-triton-metrics-port` 仅用于匹配已经登记的实例地址；不能与 `x-hot-triton-instance-id` 同时使用。新调用建议统一使用实例 ID。

## Triton 启动前提

Controller 的模型加载、卸载和版本下线依赖 Triton Repository API。所有被 Controller 管理的 Triton 实例都要使用 `EXPLICIT` 控制模式，并关闭 repository polling：

```bash
tritonserver \
  --model-repository=/repository/trt_models \
  --model-control-mode=EXPLICIT \
  --repository-poll-secs=0
```

如果启动命令中已有 `--repository-poll-secs`，必须删除非零设置或改为 `0`。启用 repository polling 时，Triton 会拒绝显式 load/unload API，并返回 `explicit model load / unload is not allowed if polling is enabled`。修改参数后需要重启 Triton。`--load-model=*` 是可选的启动预加载设置；需要启动后由 Controller 控制模型时，可以不设置。

## 接口一览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/healthz` | Controller HTTP 服务存活检查 |
| `GET` | `/runtime/health` | 检查目标 Triton 连通和 Ready 状态 |
| `GET` | `/api/status` | Triton、模型、GPU 指标和 Controller 管理状态汇总 |
| `GET` | `/api/models` | 已管理模型与 Triton Repository 模型概览 |
| `GET` | `/api/state` | 当前实例持久化的管理状态、Job 和版本操作 |
| `GET` | `/api/gpu-status` | 汇总后的 GPU/Triton 状态 |
| `GET` | `/api/gpu-metrics` | Triton 暴露的 GPU metrics 数据 |
| `GET` | `/api/jobs/{job_name}` | 查询 model-copy Job 状态与日志 |
| `POST` | `/api/models/load` | 从镜像加载一个模型 |
| `POST` | `/api/models/load-batch` | 从多个镜像批量加载模型 |
| `POST` | `/api/models/unload` | 卸载一个模型的 Triton 运行态 |
| `POST` | `/api/models/unload-batch` | 按模型、别名批量卸载，或下线指定数字版本 |
| `POST` | `/api/models/reload` | 重新加载一个已有模型 |
| `GET` | `/api/version-operations/{operation_id}` | 查询并推进指定版本下线/回退事务 |
| `GET` | `/api/instances` | 列出已登记 Triton 实例 |
| `POST` | `/api/instances` | 添加 Triton 实例 |
| `PUT` | `/api/instances/{instance_id}` | 更新实例名称和地址 |
| `DELETE` | `/api/instances/{instance_id}` | 删除实例登记 |

## 健康检查与状态查询

```bash
curl -sS "${BASE_URL}/healthz"
curl -sS "${BASE_URL}/runtime/health" \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}"
curl -sS "${BASE_URL}/api/status" \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}"
curl -sS "${BASE_URL}/api/models" \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}"
curl -sS "${BASE_URL}/api/state" \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}"
curl -sS "${BASE_URL}/api/gpu-status" \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}"
curl -sS "${BASE_URL}/api/gpu-metrics" \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}"
```

`/healthz` 只表示 Controller Web 服务存活；操作 Triton 前应同时检查 `/runtime/health` 的 `triton_ready`。

## 加载模型

### 异步加载单个模型

`image` 必填；`model_name` 可省略，Controller 会尝试从镜像 tag 推导。省略 `wait_for_ready` 或设为 `false` 时，接口提交任务后返回，之后用响应里的 `job_name` 查询。

```bash
curl -sS -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}" \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
  }'
```

显式模型名和 callback 的例子：

```bash
curl -sS -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}" \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
    "model_name": "unit_empty_space_uspg_yolov8",
    "callback": {
      "url": "https://your-service.example.com/triton/callback",
      "events": ["terminal"],
      "token": "replace-with-shared-secret"
    }
  }'
```

callback 只支持加载接口，目前只发送终态 `terminal` 事件。终态包括 `MODEL_READY`、`COPY_FAILED` 和 `TRITON_RELOAD_FAILED`。配置 token 后，Controller 会发送 `X-Hot-Loader-Timestamp` 和 `X-Hot-Loader-Signature: sha256=...`；签名为 HMAC-SHA256，内容是 `timestamp + "." + raw_body`。

### 等待单个模型加载到终态

`wait_for_ready: true` 会阻塞 HTTP 请求，直到加载流程进入终态。客户端要为请求设置合适的超时；若调用方不能长时间等待，使用默认异步方式并轮询 Job。

```bash
curl -sS -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}" \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
    "wait_for_ready": true
  }'
```

### 批量加载

每项提供 `image`，`model_name` 可选；整个批次共用 `wait_for_ready` 和 callback 设置。`models` 不能为空。

```bash
curl -sS -X POST "${BASE_URL}/api/models/load-batch" \
  -H 'Content-Type: application/json' \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}" \
  -d '{
    "models": [
      {"image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"},
      {"model_name": "unit_hanging_product_yolov5", "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620"}
    ],
    "wait_for_ready": false
  }'
```

### 查询 Job

从 load 响应复制 `job_name`：

```bash
curl -sS "${BASE_URL}/api/jobs/<job_name>" \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}"
```

响应包含任务状态、模型名、目标版本、重载尝试信息及可用的 Job 日志。重复提交相同模型和镜像的活跃 load 会复用原 Job；同模型仍有其他活跃操作时提交冲突镜像会返回 HTTP `409`。

## 卸载与重载

### 卸载单个模型

只卸载 Triton 运行态，保留 Repository 文件和管理映射，之后可直接 reload。

```bash
curl -sS -X POST "${BASE_URL}/api/models/unload" \
  -H 'Content-Type: application/json' \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}" \
  -d '{"model_name":"unit_empty_space_uspg_yolov8"}'
```

### 批量卸载模型或别名

`models` 和 `aliases` 可以单独使用，也可以在同一个请求中一起提供。此处卸载的是 Triton 运行态，不会删除模型文件。

```bash
curl -sS -X POST "${BASE_URL}/api/models/unload-batch" \
  -H 'Content-Type: application/json' \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}" \
  -d '{
    "models": ["unit_empty_space_uspg_yolov8", "unit_hanging_product_yolov5"],
    "aliases": ["model_unit_empty_space_uspg_yolov8"]
  }'
```

### 下线指定数字版本并回退

使用 `versions` 数组，每项格式为 `model_name@version`。Controller 会至少保留一个数字版本、备份并移出指定版本、更新 `version_policy`，然后**必须重载当前 Triton 实例**，使运行态按新策略切换。只移除磁盘上的版本目录不会让 Triton 自动切换；完成 reload 并确认剩余版本 `READY` 是下线成功的必要条件。`versions` 不能与 `models` 或 `aliases` 混用。

```bash
curl -sS -X POST "${BASE_URL}/api/models/unload-batch" \
  -H 'Content-Type: application/json' \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}" \
  -d '{"versions":["unit_empty_space_uspg_yolov8@2"]}'
```

响应中的 `pending: true` 表示版本已提交下线，但 Triton reload 或 READY 确认仍在运行/等待恢复；此时不能认为下线或回退已完成。使用 `operations[].id` 查询：

```bash
curl -sS "${BASE_URL}/api/version-operations/<operation_id>" \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}"
```

`SUCCEEDED` 表示剩余版本已确认就绪且备份已清理；`FAILED_RESTORED` 表示操作失败后原始目录和配置已恢复；`RECOVERY_REQUIRED` 表示需要运维确认 Triton 状态后恢复处理。查询该接口会尝试推进事务状态。

### 重载模型

```bash
curl -sS -X POST "${BASE_URL}/api/models/reload" \
  -H 'Content-Type: application/json' \
  -H "x-hot-triton-instance-id: ${INSTANCE_ID}" \
  -d '{"model_name":"unit_empty_space_uspg_yolov8"}'
```

reload 直接对 Triton 已有 Repository 模型执行 load，不会创建新的模型复制 Job。

## 管理 Triton 实例

实例清单保存在 Controller 的持久化状态文件。新建/更新实例接口不需要实例选择 header。

### 列出实例

```bash
curl -sS "${BASE_URL}/api/instances"
```

响应包含 `default_instance_id` 和 `instances` 数组，每条记录提供 `id`、`name`、`triton_url`、`metrics_url`。

### 添加实例

```bash
curl -sS -X POST "${BASE_URL}/api/instances" \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"GPU B",
    "triton_url":"http://10.0.0.8:8000",
    "metrics_url":"http://10.0.0.8:8002/metrics"
  }'
```

成功时返回 HTTP `201` 和固定 `id`，之后把该 ID 放到 `x-hot-triton-instance-id` header。只给主机名/IP 时默认 HTTP 端口为 `8000`，Metrics 为同主机 `8002/metrics`。`metrics_url` 可省略。

### 更新实例

```bash
curl -sS -X PUT "${BASE_URL}/api/instances/<instance_id>" \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"GPU B staging",
    "triton_url":"http://10.0.0.9:8000",
    "metrics_url":"http://10.0.0.9:8002/metrics"
  }'
```

地址发生变化时，省略 `metrics_url` 会使用新主机的 `8002/metrics`。若地址不变且省略 `metrics_url`，会保留原 Metrics 地址。实例存在活跃 Job、未投递终态 callback 或未结束版本操作时，不能更改地址。

### 删除实例

```bash
curl -sS -X DELETE "${BASE_URL}/api/instances/<instance_id>"
```

删除只移除实例登记，不删除其历史 Job 或模型文件。默认实例不能删除；有活跃任务、未投递 callback 或未结束版本操作时不能删除。

## Triton 客户端的版本化推理

推理请求直接发送给目标 Triton HTTP 地址，而不是 Controller。版本号放在 Triton 推理 URL 中：

```bash
TRITON_URL="http://127.0.0.1:8000"
curl -sS -X POST "${TRITON_URL}/v2/models/<model_name>/versions/<version>/infer" \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name":"INPUT0", "shape":[1], "datatype":"FP32", "data":[1.0]}
    ]
  }'
```

版本切换过程中，目标版本尚未 `READY` 时，固定版本请求可能收到 Triton HTTP `503`。客户端应根据业务要求重试，或等 Controller 的版本操作进入 `SUCCEEDED` 后切换请求版本。Controller 的 unload/reload 管理 API 本身不代理推理流量。

## 兼容路由与错误

优先使用 `/api/...`。仍提供以下兼容路由：

| 兼容路由 | 推荐路由 |
| --- | --- |
| `POST /models/load` | `POST /api/models/load` |
| `POST /models/load-batch` | `POST /api/models/load-batch` |
| `GET /models/jobs/{job_name}` | `GET /api/jobs/{job_name}` |
| `GET /models` | `GET /api/models` |
| `POST /models/unload` | `POST /api/models/unload` |
| `POST /models/reload` | `POST /api/models/reload` |
| `POST /api/unload` | `POST /api/models/unload-batch` |
| `POST /api/reload` | `POST /api/models/reload`（兼容批量载荷 `{"models":[...]}`） |
| `GET /runtime/gpu-status` | `GET /api/gpu-status` |
| `GET /metrics/gpu` | 历史汇总格式，返回 GPU、Triton 与 manager 信息 |

Controller 业务错误通常返回 `{"success":false,"detail":"..."}`；参数或运行错误返回 HTTP `400`，资源冲突返回 HTTP `409`。JSON 请求统一使用 `Content-Type: application/json`。
