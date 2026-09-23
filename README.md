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

同名模型的不同数字版本目录可共存，并使用共享 `version_policy` 加载。支持登记多个 Triton 实例，共用模型仓库，分别控制各实例的加载、卸载和重载。

### Triton 启动参数

Controller 通过 Triton Repository API 显式执行模型 `load/unload`。每个受管理的 Triton 实例都必须使用 `EXPLICIT` 模型控制模式，并关闭 repository polling：

```bash
tritonserver \
  --model-repository=/repository/trt_models \
  --model-control-mode=EXPLICIT \
  --repository-poll-secs=0
```

若使用 `--repository-poll-secs`，请将其设为 `0`；不要启用非零轮询。Polling 与显式 load/unload 冲突时，Triton 会拒绝 API 请求并返回 `explicit model load / unload is not allowed if polling is enabled`。Kubernetes Deployment 中应检查 Triton 容器的 `command/args`，修改启动参数后重启 Triton Pod。`--load-model=*` 可按需用于 Triton 启动时预加载仓库中的模型，不替代以上控制模式和 polling 设置。

## 指定版本下线与回退

页面每个版本行的“下线此版本并重载”会移除该版本目录、更新共享 `specific` 策略，并**必须重载当前 Triton 实例**，使运行态按新版本策略重新收敛。只移除磁盘目录不会切换 Triton 当前加载的版本；重载并确认剩余版本 `READY` 是下线完成的必要步骤。此操作不会先卸载整个模型，且至少保留一个数字版本。

```bash
python3 cli.py unload --instance-id default --versions demo@2
curl -X POST http://127.0.0.1:8090/api/models/unload-batch \
  -H 'Content-Type: application/json' \
  -H 'x-hot-triton-instance-id: default' \
  -d '{"versions":["demo@2"]}'
```

`/api/unload` 同样支持 `versions`。不能与 `models` 或 `aliases` 混用。重复版本会去重，同模型的多个版本合并成一次重载，不同模型分别返回处理结果。

响应包含 `success`、`pending` 和 `operations`。下线版本后必须完成 Triton reload，并确认剩余版本 `READY`；`pending: true` 表示重载/确认仍在处理或等待恢复确认，不能当作切换成功。使用 `GET /api/version-operations/{id}`（同一实例 header）或 `GET /api/status` 查看进度。页面任务列表也会展示版本操作。纯 CLI 使用者需运行 `serve` 提供后台续处理，或调用 `status` 推进操作。

| 状态 | 含义 |
| --- | --- |
| `VALIDATING` / `PREPARING` | 验证目录与策略，备份并移出版本 |
| `REQUESTING` / `VERIFYING` | 已准备重载请求或等待剩余版本全部 READY、移除版本不再 READY |
| `COMMITTING` / `SUCCEEDED` | 已确认切换，更新元数据及清理备份；SUCCEEDED 才表示完成 |
| `RESTORING` / `FAILED_RESTORED` | 恢复中或失败后已恢复原始配置和目录（验证失败时文件未变） |
| `RECOVERY_REQUIRED` | 超过重载总时限或恢复/清理异常，保留记录、备份及模型互斥保护 |

备份放在模型仓库的同卷兄弟目录 `.hot-loader-version-backups/<operation_id>/<model>/`，状态文件持久化原始配置与事务阶段。Triton 明确拒绝重载时恢复原文件；网络错误、代理超时或版本尚未收敛时继续查询，不自动恢复可能仍被加载的仓库，也不重复发出结果未知的 Load 请求。重启后后台继续确认；若最终确认切换完成，会清理备份并解除保护。

同名模型存在加载或版本操作时，新的版本下线被拒绝；版本操作未结束时，同名加载、卸载、重载均被阻止，目标实例不能修改地址或删除。`RECOVERY_REQUIRED` 持续保留此保护。排障先核对目标 Triton 是否仍在加载及具体版本状态，不要直接删状态记录解锁；若请求是否执行始终无法确定，需要运维确认 Triton 已停止模型变更后恢复备份和配置，再核对实际运行态。

共享仓库中的移除影响所有实例下次加载，但本操作只重载当前实例。固定请求已删除版本的客户端不会自动改用老版本；序列模型的会话连续性及不中断推理需在实际 Triton/backend 环境验证。Job-only 模式必须使用包含 `/app/version_transaction.py`（协议 1）的本项目维护镜像；缺少脚本时维护 Job 在修改文件前失败。镜像构建后再配置 `REPOSITORY_MAINTENANCE_IMAGE`，单元测试不等于 Kubernetes/GPU 上线验证。

## 多 Triton 实例

页面顶部选择当前实例，在“管理 Triton 实例”中添加、编辑或删除登记。添加时输入名称、IP 或 HTTP 地址，以及可选 Metrics 地址；纯 IP 默认使用 HTTP 8000、Metrics 8002。自定义 Metrics 端口请填写完整地址。所有模型操作、Job 和 GPU 查询均针对当前实例；共享仓库中的文件和镜像记录由所有实例共用。

实例列表保存在 `HOT_TRITON_STATE_FILE` 的 `instances` 字段，浏览器仅保存选中的 ID。首次启动将环境配置登记为 `default`，旧 Job 自动归属该实例；以后地址修改通过页面或 API 完成，重启或修改环境变量不会覆盖已保存的登记。状态文件必须位于持久化卷。

| 接口 | 用途 |
| --- | --- |
| `GET /api/instances` | 返回 `default_instance_id` 和 `instances` 列表 |
| `POST /api/instances` | 添加实例，返回含固定 `id` 的记录，HTTP 201 |
| `PUT /api/instances/{id}` | 更新名称和地址，ID 不变 |
| `DELETE /api/instances/{id}` | 删除登记，保留历史任务及模型文件 |

添加或更新的请求体：

更新时若 HTTP 地址不变且未填写 `metrics_url`，保留已有 Metrics 配置；新实例或更换 HTTP 地址时，省略该字段使用同主机的 8002 端口。

```json
{"name":"GPU B","triton_url":"10.0.0.8","metrics_url":"http://10.0.0.8:8002/metrics"}
```

取得返回的 `id` 后，在现有 API 请求中添加 `x-hot-triton-instance-id`：

```bash
curl -X POST http://127.0.0.1:8090/api/models/load \
  -H 'Content-Type: application/json' \
  -H 'x-hot-triton-instance-id: <返回的实例 ID>' \
  -d '{"model_name":"demo","image":"ccr.ccs.tencentyun.com/clobotics/demo:1"}'
```

不传实例 ID 时使用 `default`。兼容 header `x-hot-triton-url` / `x-hot-triton-metrics-port` 只能选择已登记的地址，不能与实例 ID 同时使用，未知地址返回 HTTP 400。任务结果和终态回调包含 `instance_id`、提交时的 `triton_url`；后台重试绑定原实例，不受页面切换影响。

同一共享模型有活跃加载任务时，其他实例提交该模型返回 HTTP 409；只有同实例、同镜像的重复提交复用原 Job。不同模型可以并行加载。实例有活跃任务或待投递回调时，修改地址和删除返回 HTTP 409；仍允许只修改名称。默认实例不能删除。

**部署要求：**所有 Triton 必须读取同一共享 PVC 中的模型仓库，不能各自使用独立 `emptyDir`。目前支持单 controller 进程（单副本、单 worker）；详细挂载要求见 [运维说明](docs/ops/README.md#多-triton-共享仓库部署)。共享文件的更新会在其他实例下一次加载/重载时生效，不会自动向所有实例广播重载。部署前应在真实双 Triton/PVC 环境验证。

## 文档入口

- 运维简要配置与说明: [docs/ops/README.md](docs/ops/README.md)
- HTTP API 完整使用说明: [docs/api/README.md](docs/api/README.md)
- CLI 使用说明: [docs/cli/README.md](docs/cli/README.md)
- realtime-dev 部署说明: [deploy/realtime-dev/README.md](deploy/realtime-dev/README.md)
- 单节点 K3s Job 测试环境: [deploy/k3s-job-test/README.md](deploy/k3s-job-test/README.md)

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

### PVC 与模型仓库路径

| 环境变量 | 用途 | 示例 |
| --- | --- | --- |
| `TRITON_REPOSITORY_PVC` | PVC 的 `metadata.name`，供复制 Job 挂载 | `triton-repository-pvc` |
| `MODEL_SOURCE_PATH` | 镜像内模型目录 | `/trt_models` |
| `MODEL_TARGET_PATH` | PVC 内模型仓库目录 | `/repository/trt_models` |
| `HOT_TRITON_MODEL_REPOSITORY` | Triton 在线仓库路径 | `/repository/trt_models` |
| `HOT_TRITON_STATE_FILE` | 持久化 Controller 状态文件 | `/repository/.hot_loader/state.json` |
| `HOT_TRITON_STAGING_ROOT` | 仓库外的暂存目录 | `/repository/.staging` |

PVC 名称不是容器路径。目标路径为 `/repository/trt_models` 时，复制 Job 将 PVC 挂到 `/repository`。直读模式下，Controller 和 Triton 都挂载该 PVC，且在线仓库路径与目标路径一致；临时仓库模式下，Controller 同时挂载 PVC 和临时卷并负责同步，Triton 只挂临时卷。

PVC volume 示例：

```yaml
volumes:
  - name: model-repository
    persistentVolumeClaim:
      claimName: triton-repository-pvc
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
- `TRITON_REPOSITORY_PVC` 是 Kubernetes PVC 的 `metadata.name`；controller 创建的 model-copy Job 会将其填入 `persistentVolumeClaim.claimName`，不是容器内路径。
- `MODEL_TARGET_PATH` 是 model-copy Job 挂载该 PVC 后的写入目录。例如设为 `/repository/trt_models` 时，Job 将 PVC 挂载到 `/repository`，并把模型写入 `${MODEL_TARGET_PATH}/${MODEL_NAME}`。
- `HOT_TRITON_MODEL_REPOSITORY` 是 Controller 用于检查、同步并调用 Triton 加载的在线模型仓库路径。若 Triton 与 Controller 直接读取同一 PVC，应与 `MODEL_TARGET_PATH` 一致；若 Triton 使用 `emptyDir` 或共享临时卷，则配置为该在线目录，Controller 会在 Job 写入 PVC 后同步模型过去。
- 现有项目里的模型初始化镜像默认把模型放在 `/trt_models/<model_name>/...`，controller 会优先按这个结构复制；如果镜像里直接是单模型内容目录，也会回退兼容。
- 生产环境建议把 `HOT_TRITON_STATE_FILE` 和 `HOT_TRITON_STAGING_ROOT` 放在 `trt_models` 目录外层，避免 `.hot_loader/`、`.staging/` 进入 Triton model store。
- 同名模型的新版本会先与仓库中已有数字版本目录合并到挂载卷里的 `.staging/`，再原子切换到目标目录；controller 会把 `config.pbtxt` 更新为包含所有发现版本的 `specific` 策略，使 Triton 同时加载它们。
- 当 `MODEL_TARGET_PATH` 是 `/repository/trt_models` 这种 PVC 子目录时，model-copy Job 会把 PVC 挂到它的父目录 `/repository`，然后再复制到 `${MODEL_TARGET_PATH}/${MODEL_NAME}`。
- `JOB_TTL_SECONDS_AFTER_FINISHED=0` 表示 Job 一旦进入完成态就立即交给 TTL controller 删除；controller 自己的状态文件仍会保留最近一次结果摘要。
- 即使复制 Job 已被 TTL 清理，只要模型目录已经落盘，controller 仍会继续自动推进后续的 Triton load/reload 状态机。
- reload 只确认本次复制记录的目标版本；同名旧版本的状态不会决定当前 operation。默认最多 8 次、以 2 秒起步并封顶 60 秒的指数退避等待 READY，总时限 600 秒。
- 线上建议保留 `JOB_TTL_SECONDS_AFTER_FINISHED=0`；排查复制或调度问题时，建议临时调大到 `300`，便于直接看 Job / Pod / Event。
- 如果集群节点带 taint，需要通过 `JOB_TOLERATIONS_JSON` 给动态创建的 model-copy Job 补 tolerations。
- Triton 必须使用 `EXPLICIT` 模式，且 `repository_poll_secs=0`。

同一 PVC 直读模式的推荐配置：

```env
TRITON_REPOSITORY_PVC=triton-models-storage
MODEL_TARGET_PATH=/repository/trt_models
HOT_TRITON_MODEL_REPOSITORY=/repository/trt_models
```

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
- 按 `models/aliases` 卸载只改变 Triton 运行态，保留文件和镜像映射；`versions` 下线会移除指定版本并自动重载，见上文。

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
- 如果新请求解析出的 `model_name` 与当前已加载模型同名，controller 会保留旧数字版本目录并加载新版本；状态中的 `managed_model_versions` 可查看当前版本集合。
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
