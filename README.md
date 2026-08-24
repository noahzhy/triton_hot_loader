# Triton Hot Loader Controller

一个基于 Kubernetes Job 的 Triton 模型热加载控制器。

它的职责是：

1. 接收 `image` 请求，并在未显式指定时根据镜像 tag 自动提取 `model_name`。
2. 动态创建 model-copy Job。
3. 让 Kubernetes 自动拉取模型镜像。
4. 把镜像内的模型文件复制到 Triton Repository PVC。
5. 调用 Triton Repository API 执行 `load / reload`。
6. 提供 Job、模型、Triton 和 GPU 指标查询接口。

当前实现已经完全移除热加载链路里的 `docker pull / docker create / docker cp / docker rm / docker.sock` 依赖。

同名模型现在采用直接替换语义：无论镜像新旧，只要解析出的 `model_name` 相同，就覆盖当前仓库里的同名模型；controller 不再对同名模型做额外版本管理。

## 文档入口

- 运维简要配置与说明: [docs/ops/README.md](docs/ops/README.md)
- CLI 使用说明: [docs/cli/README.md](docs/cli/README.md)
- realtime-dev 部署说明: [deploy/realtime-dev/README.md](deploy/realtime-dev/README.md)

## 架构

```text
Client / UI
    |
    | POST /models/load
    v
Triton Hot Loader Controller
    |
    | create Job
    v
Kubernetes Job
    |
    | copy ${MODEL_SOURCE_PATH}/${MODEL_NAME} or ${MODEL_SOURCE_PATH} -> ${MODEL_TARGET_PATH}/${MODEL_NAME}
    v
Triton Repository PVC
    |
    | POST /v2/repository/models/{model}/load
    v
Triton Server
    |
    | POST callback (optional, terminal only)
    v
Business System
```

## 环境变量

必填或常用：

```env
HOT_TRITON_MODEL_REPOSITORY=/repository/trt_models
HOT_TRITON_STATE_FILE=/repository/.hot_loader/state.json
HOT_TRITON_STAGING_ROOT=/repository/.staging
MODEL_SOURCE_PATH=/trt_models
MODEL_TARGET_PATH=/repository/trt_models
TRITON_REPOSITORY_PVC=triton-repository-pvc
TRITON_URL=http://triton:8000
TRITON_METRICS_URL=http://triton:8002/metrics
K8S_NAMESPACE=default
```

可选：

```env
MODEL_IMAGE_REGISTRY_PREFIX=ccr.ccs.tencentyun.com/clobotics/
JOB_TTL_SECONDS_AFTER_FINISHED=0
JOB_BACKOFF_LIMIT=1
MODEL_COPY_CPU_REQUEST=100m
MODEL_COPY_MEMORY_REQUEST=256Mi
MODEL_COPY_CPU_LIMIT=1
MODEL_COPY_MEMORY_LIMIT=1Gi
MAX_CONCURRENT_JOBS=0
TRITON_RELOAD_MAX_ATTEMPTS=8
TRITON_RELOAD_RETRY_BASE_SECONDS=2
TRITON_RELOAD_RETRY_MAX_SECONDS=60
TRITON_RELOAD_TIMEOUT_SECONDS=600
JOB_TOLERATIONS_JSON=[{"key":"gpu","operator":"Exists","effect":"NoSchedule"}]
```

说明：

- Controller 需要能访问 Kubernetes API。
- Controller 最好与 Triton 共享同一个 Repository PVC。
- 现有项目里的模型初始化镜像默认把模型放在 `/trt_models/<model_name>/...`，controller 会优先按这个结构复制；如果镜像里直接是单模型内容目录，也会回退兼容。
- 生产环境建议把 `HOT_TRITON_STATE_FILE` 和 `HOT_TRITON_STAGING_ROOT` 放在 `trt_models` 目录外层，避免 `.hot_loader/`、`.staging/` 进入 Triton model store。
- 同名模型替换时，model-copy Job 会先把新模型复制到挂载卷里的 `.staging/`，再切换到目标目录，避免长时间直接覆盖线上模型目录。
- 当 `MODEL_TARGET_PATH` 是 `/repository/trt_models` 这种 PVC 子目录时，model-copy Job 会把 PVC 挂到它的父目录 `/repository`，然后再复制到 `${MODEL_TARGET_PATH}/${MODEL_NAME}`。
- `JOB_TTL_SECONDS_AFTER_FINISHED=0` 表示 Job 一旦进入完成态就立即交给 TTL controller 删除；controller 自己的状态文件仍会保留最近一次结果摘要。
- 即使复制 Job 已被 TTL 清理，只要模型目录已经落盘，controller 仍会继续自动推进后续的 Triton load/reload 状态机。
- reload 只确认本次复制记录的目标版本；同名旧版本的状态不会决定当前 operation。默认最多 8 次、以 2 秒起步并封顶 60 秒的指数退避等待 READY，总时限 600 秒。
- 线上建议保留 `JOB_TTL_SECONDS_AFTER_FINISHED=0`；排查复制或调度问题时，建议临时调大到 `300`，便于直接看 Job / Pod / Event。
- 如果集群节点带 taint，需要通过 `JOB_TOLERATIONS_JSON` 给动态创建的 model-copy Job 补 tolerations。
- Triton 必须使用 `EXPLICIT` 模式，且 `repository_poll_secs=0`。

临时目录 Triton repository + PVC 同步模式：

- 如果 Triton 的在线 `model-store` 必须放在临时目录，可以把 `HOT_TRITON_MODEL_REPOSITORY` 配成共享的 `emptyDir`，例如 `/shared-volume/trt_models`。
- 只要 `HOT_TRITON_MODEL_REPOSITORY` 与 `MODEL_TARGET_PATH` 不同，controller 就会自动切换到 repository sync 模式：`load/load-batch` 先通过 model-copy Job 把模型复制到 `TRITON_REPOSITORY_PVC`，然后 controller 再把 `${MODEL_TARGET_PATH}/${MODEL_NAME}` 同步到本地临时目录，最后调用 Triton load。
- 这种模式下，controller 需要同时挂载：
  - 临时目录，例如 `shared-volume -> /shared-volume`
  - Triton Repository PVC，例如 `triton-repository -> /repository`
- 推荐配置：

```env
HOT_TRITON_MODEL_REPOSITORY=/shared-volume/trt_models
HOT_TRITON_STATE_FILE=/shared-volume/.hot_loader/state.json
HOT_TRITON_STAGING_ROOT=/shared-volume/.staging
MODEL_TARGET_PATH=/repository/trt_models
TRITON_REPOSITORY_PVC=triton-repository-pvc
```

- 这种模式下 Triton 自己只需要挂 `shared-volume`；不需要直接读 PVC。
- `unload` 始终只改变 Triton 运行态，不会删除 PVC 或临时目录中的模型文件，也不会清除镜像映射；可以直接调用 `reload` 恢复。

## HTTP API

### 网页 API

网页前端统一走 `/api/...` 接口。

- `POST /api/models/load`
- `POST /api/models/load-batch`
- `GET /api/jobs/{job_name}`
- `GET /api/status`
- `GET /api/models`
- `GET /api/state`
- `GET /api/gpu-status`
- `GET /api/gpu-metrics`
- `POST /api/models/unload`
- `POST /api/models/unload-batch`
- `POST /api/models/reload`

当前加载接口默认是异步语义：请求在 Job 创建成功后立即返回。
如果调用方确实希望当前 HTTP 请求一直等到 Triton load/reload 进入终态，可以显式传 `wait_for_ready: true`。

如果调用方不想轮询，也可以在加载请求里附带 `callback` 配置；controller 会在 Job 进入终态时主动回调一次业务接口。

### 加载单个模型

```http
POST /api/models/load
```

```json
{
  "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
}
```

### 批量加载

```http
POST /api/models/load-batch
```

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

- `model_name` 现在是可选字段；如果不传，controller 会优先从 image tag 提取。
- `wait_for_ready` 默认是 `false`；设为 `true` 时，接口会阻塞到 Triton 最终进入终态。
- 相同 `model_name + image` 的活跃重复请求会返回原有 Job（`reused: true`），不会创建第二个复制 Job；同一模型的不同镜像在前一 operation 未结束时返回 HTTP 409。
- 提取规则会去掉 tag 末尾常见的日期/时间发布后缀，例如 `unit_empty_space_uspg_yolov8-20260430 -> unit_empty_space_uspg_yolov8`。
- 对以 `-YYYYMMDD` 结尾的镜像 tag，controller 会预先记录该版本为本次 operation 的目标版本；model-copy Job 完成后会以实际复制出的数字版本目录覆盖确认。
- tag 中的 `-`、`.` 会统一规整成 `_`，例如 `model-a -> model_a`。
- 如果新请求解析出的 `model_name` 与当前已加载模型同名，controller 会直接替换旧模型，不保留同名历史版本。
- `callback` 是可选对象；当前只支持 `terminal` 事件，也就是 `MODEL_READY`、`COPY_FAILED`、`TRITON_RELOAD_FAILED` 这三类终态回调。
- `callback.url` 必须是你自己的业务回调接收地址，不应该填写 hot-loader 自己的 `http://10.2.24.10:30890/...`。
- `callback.token` 如果提供，controller 会在回调请求头里附带 `X-Hot-Loader-Signature: sha256=<hmac>`，签名内容是 `timestamp + "." + raw_body`。

### 终态 Callback

```json
{
  "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
  "callback": {
    "url": "https://your-service.example.com/triton/callback",
    "events": ["terminal"],
    "token": "shared-secret"
  }
}
```

回调体示例：

```json
{
  "event_id": "2b96094a-24f1-472d-b0b8-6c52756d7f68",
  "event_type": "job.status.changed",
  "job_name": "model-copy-unit-empty-space-uspg-yolov8-xxxxxx",
  "model_name": "unit_empty_space_uspg_yolov8",
  "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
  "status": "MODEL_READY",
  "detail": "Triton 已确认本次交付的目标版本 READY",
  "terminal": true,
  "triton_ready": true,
  "target_versions": ["20260430"],
  "triton_reload_attempts": 1,
  "updated_at": "2026-06-09T09:58:10+00:00",
  "callback_attempt": 1
}
```

回调请求头：

- `Content-Type: application/json`
- `X-Hot-Loader-Event: job.status.changed`
- `X-Hot-Loader-Job-Name: <job_name>`
- `X-Hot-Loader-Timestamp: <unix-ts>`
- `X-Hot-Loader-Signature: sha256=<hmac>`（仅当提供 `callback.token` 时）

### curl POST 示例

假设服务运行在 `http://127.0.0.1:8090`：

```bash
BASE_URL="http://127.0.0.1:8090"
```

加载单个模型：

```bash
curl -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
  }'
```

批量加载：

```bash
curl -X POST "${BASE_URL}/api/models/load-batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "models": [
      {
        "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
      },
      {
        "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620"
      }
    ]
  }'
```

加载并注册 callback：

```bash
curl -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
    "callback": {
      "url": "https://your-service.example.com/triton/callback",
      "events": ["terminal"],
      "token": "shared-secret"
    }
  }'
```

卸载单个模型：

```bash
curl -X POST "${BASE_URL}/api/models/unload" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "unit_empty_space_uspg_yolov8"
  }'
```

`unload` 只卸载 Triton 运行态：PVC 中的模型文件和 controller 的镜像映射都会保留。随后调用 `reload` 即可直接恢复，无需再次创建 model-copy Job。

批量卸载模型：

```bash
curl -X POST "${BASE_URL}/api/models/unload-batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "models": [
      "unit_empty_space_uspg_yolov8",
      "unit_hanging_product_yolov5"
    ]
  }'
```

重载单个模型：

```bash
curl -X POST "${BASE_URL}/api/models/reload" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "unit_empty_space_uspg_yolov8"
  }'
```

### 查询 Job

```http
GET /api/jobs/{job_name}
```

### 查询模型概览

```http
GET /api/models
```

### 卸载模型

```http
POST /api/models/unload
```

```json
{
  "model_name": "unit_empty_space_uspg_yolov8"
}
```

### 重载模型

```http
POST /api/models/reload
```

```json
{
  "model_name": "unit_empty_space_uspg_yolov8"
}
```

### reload 重试与状态

模型文件复制完成后，controller 只会确认本次复制交付的 `target_versions` 是否 READY；不会因同名旧版本状态而提前返回。未就绪时使用指数退避重试，默认最多 8 次、起始间隔 2 秒、单次间隔上限 60 秒、总时限 10 分钟。状态响应会返回 `triton_reload_attempts`、`triton_reload_last_attempt_at`、`triton_reload_next_attempt_at` 和最后一次 `error`；达到次数或时限后状态固定为 `TRITON_RELOAD_FAILED`。

### GPU / Triton 指标

```http
GET /api/gpu-status
GET /api/gpu-metrics
```

兼容状态接口仍然保留：

- `POST /models/load`
- `POST /models/load-batch`
- `GET /models/jobs/{job_name}`
- `GET /models`
- `POST /models/unload`
- `POST /models/reload`
- `GET /runtime/health`
- `GET /runtime/gpu-status`
- `POST /api/unload`
- `POST /api/reload`

## CLI

启动服务：

```bash
python3 cli.py serve --host 0.0.0.0 --port 8090
```

加载单模型：

```bash
python3 cli.py load \
  --image ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430
```

批量加载：

```bash
python3 cli.py load-batch --file batch.json
```

查询 Job：

```bash
python3 cli.py job-status model-copy-unit-empty-space-uspg-yolov8-xxxxxxx
```

查看状态：

```bash
python3 cli.py status
python3 cli.py list
```

## 本地开发

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

运行测试：

```bash
python3 -m unittest tests.test_hot_loader tests.test_server
```

## 构建镜像

项目保留了镜像构建脚本：

```bash
./docker_build.sh
```

直接推送：

```bash
./docker_build.sh --push
```

默认镜像名：

```text
ccr.ccs.tencentyun.com/clobotics/triton-hot-loader
```

## Kubernetes 权限

Controller 需要 namespace 级别权限：

- `jobs`: `create`, `get`, `list`, `watch`, `delete`
- `pods`: `get`, `list`, `watch`
- `pods/log`: `get`
- `events`: `get`, `list`, `watch`
- `persistentvolumeclaims`: `get`, `list`

## 约束

- `image` 必须命中 `MODEL_IMAGE_REGISTRY_PREFIX`
- `model_name` 只允许小写字母、数字、`-`、`_`
- Job 不申请 GPU
- Job 会挂载 `TRITON_REPOSITORY_PVC`
- `MAX_CONCURRENT_JOBS` 仅在设置为正整数时才会限制同时运行的 model-copy Job 数量；`0` 表示不限制
