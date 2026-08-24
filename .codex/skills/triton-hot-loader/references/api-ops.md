# Triton Hot Loader API And Ops Reference

## 目录

- 接口使用规则
- 请求与响应约定
- 主要 API
- curl 示例
- 常用命令速查
- 运维说明

## 接口使用规则

- 编写文档和示例时，优先使用 `/api/...` 路由
- `/models/...` 以及 `/api/unload`、`/api/reload` 属于兼容路由
- `GET /healthz` 只检查 Web 服务是否存活
- `GET /runtime/health` 用于检查 controller 到 Triton 的连通性和 ready 状态

## 请求与响应约定

- 加载接口默认是异步语义
  - `wait_for_ready: false` 或省略时，只要 Job 创建成功就返回
  - `wait_for_ready: true` 时，会阻塞到 Triton load/reload 进入终态
- `load` 时可以不传 `model_name`
  - controller 会从镜像 tag 自动提取
  - 常见发布后缀如 `-20260430` 会被去掉
  - `-` 和 `.` 会被归一化成 `_`
- 如果新请求解析出的模型名与现有模型同名，会直接替换，不保留同名历史版本
- 应用级错误返回 HTTP 400，格式通常为：

```json
{
  "success": false,
  "detail": "..."
}
```

- 可以通过请求头按请求覆盖运行时目标：
  - `x-hot-triton-url`
  - `x-hot-triton-metrics-port`

## 主要 API

### 健康检查与状态

- `GET /healthz`
- `GET /runtime/health`
- `GET /api/status`
- `GET /api/models`
- `GET /api/state`
- `GET /api/gpu-status`
- `GET /api/gpu-metrics`
- `GET /api/jobs/{job_name}`

### 加载单个模型

`POST /api/models/load`

```json
{
  "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
  "model_name": "unit_empty_space_uspg_yolov8",
  "wait_for_ready": false
}
```

字段说明：

- `image`：必填
- `model_name`：可选
- `wait_for_ready`：可选，默认 `false`
- `callback`：可选

### 批量加载

`POST /api/models/load-batch`

```json
{
  "models": [
    {
      "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
    },
    {
      "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620"
    }
  ],
  "wait_for_ready": false
}
```

### Callback

callback 只支持加载接口，并且当前只支持终态事件。

```json
{
  "callback": {
    "url": "https://your-service.example.com/triton/callback",
    "events": ["terminal"],
    "token": "shared-secret"
  }
}
```

当前终态包括：

- `MODEL_READY`
- `COPY_FAILED`
- `TRITON_RELOAD_FAILED`

如果提供了 `callback.token`，controller 会附带：

- `X-Hot-Loader-Timestamp`
- `X-Hot-Loader-Signature: sha256=<hmac>`

签名内容是 `timestamp + "." + raw_body`。

### 卸载单个模型

`POST /api/models/unload`

```json
{
  "model_name": "unit_empty_space_uspg_yolov8"
}
```

### 批量卸载

`POST /api/models/unload-batch`

这里要以 `server.py` 的真实请求模型为准，不要沿用旧示例。

```json
{
  "models": [
    "unit_empty_space_uspg_yolov8",
    "unit_hanging_product_yolov5"
  ],
  "aliases": []
}
```

说明：

- `models` 和 `aliases` 至少要有一个
- `versions` 已被拒绝，同名版本级卸载不再支持

### 重载

`POST /api/models/reload`

```json
{
  "model_name": "unit_empty_space_uspg_yolov8"
}
```

## curl 示例

先设置基础地址：

```bash
BASE_URL="http://127.0.0.1:8090"
```

健康检查：

```bash
curl "${BASE_URL}/healthz"
curl "${BASE_URL}/runtime/health"
```

异步加载：

```bash
curl -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
  }'
```

同步加载：

```bash
curl -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
    "wait_for_ready": true
  }'
```

带 callback 的加载：

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

批量卸载：

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

按请求覆盖 Triton 目标：

```bash
curl -X GET "${BASE_URL}/api/status" \
  -H 'x-hot-triton-url: http://127.0.0.1:18000' \
  -H 'x-hot-triton-metrics-port: 18002'
```

## 常用命令速查

### CLI

查看帮助：

```bash
python3 cli.py --help
python3 cli.py serve --help
python3 cli.py load --help
python3 cli.py load-batch --help
python3 cli.py job-status --help
python3 cli.py status --help
python3 cli.py unload --help
python3 cli.py reload --help
```

启动服务：

```bash
python3 cli.py serve --host 0.0.0.0 --port 8090
```

查看整体状态：

```bash
python3 cli.py status
python3 cli.py list
```

加载单个模型：

```bash
python3 cli.py load \
  --image ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430
```

加载单个模型并显式指定模型名：

```bash
python3 cli.py load \
  --model-name unit_empty_space_uspg_yolov8 \
  --image ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430
```

批量加载，使用文件：

```bash
python3 cli.py load-batch --file batch.json
```

批量加载，直接传 JSON：

```bash
python3 cli.py load-batch \
  --json '[
    {"image":"ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"},
    {"image":"ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620"}
  ]'
```

查询单个 Job：

```bash
python3 cli.py job-status model-copy-demo-12345678
```

按模型名卸载：

```bash
python3 cli.py unload --models unit_empty_space_uspg_yolov8
```

按 alias 卸载：

```bash
python3 cli.py unload --aliases model_unit_empty_space_uspg_yolov8
```

重载模型：

```bash
python3 cli.py reload unit_empty_space_uspg_yolov8
python3 cli.py reload unit_empty_space_uspg_yolov8 unit_hanging_product_yolov5
```

### HTTP API

基础地址：

```bash
BASE_URL="http://127.0.0.1:8090"
```

健康检查：

```bash
curl "${BASE_URL}/healthz"
curl "${BASE_URL}/runtime/health"
curl "${BASE_URL}/api/status"
curl "${BASE_URL}/api/models"
curl "${BASE_URL}/api/state"
curl "${BASE_URL}/api/gpu-status"
curl "${BASE_URL}/api/gpu-metrics"
```

加载单个模型：

```bash
curl -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
  }'
```

同步等待加载完成：

```bash
curl -X POST "${BASE_URL}/api/models/load" \
  -H 'Content-Type: application/json' \
  -d '{
    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
    "wait_for_ready": true
  }'
```

批量加载：

```bash
curl -X POST "${BASE_URL}/api/models/load-batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "models": [
      {"image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"},
      {"image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620"}
    ]
  }'
```

查询 Job：

```bash
curl "${BASE_URL}/api/jobs/model-copy-demo-12345678"
```

卸载单个模型：

```bash
curl -X POST "${BASE_URL}/api/models/unload" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "unit_empty_space_uspg_yolov8"
  }'
```

批量卸载：

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

重载模型：

```bash
curl -X POST "${BASE_URL}/api/models/reload" \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "unit_empty_space_uspg_yolov8"
  }'
```

注册终态 callback：

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

按请求覆盖 Triton 地址：

```bash
curl -X GET "${BASE_URL}/api/status" \
  -H 'x-hot-triton-url: http://127.0.0.1:18000' \
  -H 'x-hot-triton-metrics-port: 18002'
```

### Docker

构建镜像：

```bash
bash docker_build.sh
```

构建并推送：

```bash
bash docker_build.sh --push
```

本地起容器：

```bash
bash docker_run.sh
```

指定镜像启动：

```bash
bash docker_run.sh ccr.ccs.tencentyun.com/clobotics/triton-hot-loader:latest
```

看容器日志：

```bash
docker logs -f triton-hot-loader
```

### Kubernetes / 运维

应用 RBAC：

```bash
kubectl apply -f deploy/realtime-dev/triton-controller-rbac.yaml
```

更新 Deployment：

```bash
kubectl apply -f deploy/realtime-dev/trtis-deployment-realtime-dev.yaml
```

检查 ServiceAccount：

```bash
kubectl get deployment trtis-deployment-realtime-dev -n default -o jsonpath='{.spec.template.spec.serviceAccountName}'
```

检查权限：

```bash
kubectl auth can-i list jobs.batch --as=system:serviceaccount:default:triton-controller-sa -n default
```

查看 controller 日志：

```bash
kubectl logs -n default deployment/trtis-deployment-realtime-dev -c triton-controller
```

排查 Job Pod：

```bash
kubectl describe pod -n default <job-pod-name>
kubectl get events -n default --field-selector involvedObject.name=<job-pod-name>
```

## 运维说明

### 为什么除了临时运行目录，还需要 PVC

- `model-copy` 运行在单独的 Kubernetes Job Pod 中
- `emptyDir` 这类 Pod 本地临时目录不能跨 Pod 共享
- Job Pod 不能直接把模型写进 Triton 所在 Pod 的临时运行目录
- PVC 是 Job、Controller 和 Triton 相关路径之间的稳定共享交付面
- 如果只使用临时存储，Pod 重建后模型文件可能丢失，controller 的状态也可能和真实模型仓库不一致

可以这样理解：

- PVC = 交付面和稳定共享仓库
- `HOT_TRITON_MODEL_REPOSITORY` = Triton 在线读取的运行面

当 `HOT_TRITON_MODEL_REPOSITORY != MODEL_TARGET_PATH` 时，controller 会进入 repository sync 模式：

1. Job 先把模型复制到 PVC 路径
2. Controller 等待复制出的模型目录完整可见
3. Controller 再把模型从 PVC 同步到运行时目录
4. 然后调用 Triton `load`

### 运行规则

- Triton 必须使用 `--model-control-mode=EXPLICIT`
- Triton 必须使用 `--repository-poll-secs=0`
- `JOB_TTL_SECONDS_AFTER_FINISHED=0` 会较快清理完成态 Job；调试时应临时调大
- 只有在 controller 无法直接访问 repository PVC 路径时，卸载流程才需要 `REPOSITORY_MAINTENANCE_IMAGE`
