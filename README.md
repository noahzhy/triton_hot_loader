# Triton Hot Loader

这个目录提供了一个**可直接运行的 Triton 模型热加载器示例**，包含：

- 一个 Web UI：可以直接粘贴一段 JSON 配置，执行热加载
- 一个 HTTP API：方便其他服务或脚本调用
- 一个 CLI：支持 `apply / status / list / unload / reload`，其中 `unload` 支持按 alias、模型名或 `model@version` 下线
- 一个核心热加载器：负责拉取镜像、提取 `/trt_models`、同步 Triton model repository，并通过 Triton API 执行 `load / unload`

目标是：**在不重启 Triton 服务的前提下，按镜像配置对模型进行热加载、热切换和热卸载。**

当前示例基于以下 Triton 镜像：

- `ccr.ccs.tencentyun.com/clobotics/tritonserver:24.12-py3`

## 补充文档

- 独立 CLI 命令手册：[`docs/cli/README.md`](docs/cli/README.md)
- 项目 Skill：[`.codex/skills/triton-hot-loader/SKILL.md`](.codex/skills/triton-hot-loader/SKILL.md)

## 适用输入

输入是一组 `占位键 -> image` 的 JSON，**系统只读取 value，不再把 key 当作 alias**：

```json
{
  "unit_model": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hpc_us_yolov8-20260107",
  "sku_model": "ccr.ccs.tencentyun.com/clobotics/sku-model-init:sku_hpc_us_resnet-20260527",
  "secondary_unit_model": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hanging_product_yolov5-20230620",
  "hanging_box_model": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430"
}
```

这里的键（例如 `unit_model`）只是为了让 JSON 对象合法，热加载器不会再把它们当成长期 alias。真正参与管理的是**镜像地址本身**以及镜像里发现的 Triton 模型：

- 当前生效的镜像地址
- 该镜像中实际包含的 Triton 模型名列表
- 每个模型可用的版本目录，以及当前激活版本
- 最近更新时间

另外，所有 `mlman_config / mlmanconfig` 相关条目都会在解析配置时被自动忽略。

如果你要按“一个请求里显式指定多个模型镜像”的 controller 风格调用，服务端现在也兼容以下载荷：

```json
{
  "triton_url": "http://triton-server:8000",
  "triton_metrics_url": "http://triton-server:8002/metrics",
  "models": [
    {
      "model_name": "sku_classifier",
      "image": "registry.xxx.com/models/sku-classifier:20260605"
    },
    {
      "model_name": "volume_detector",
      "image": "registry.xxx.com/models/volume-detector:20260605"
    }
  ],
  "options": {
    "load_after_copy": true,
    "overwrite": true
  }
}
```

这一路径下，当前项目的实际镜像内源路径是：

- `/trt_models/{model_name}`

不是 `/model-repository/{model_name}`。这个约定来自仓库现有实现 `HotLoaderConfig.image_model_root="/trt_models"` 和实际的 `docker cp` 逻辑。

## 目录说明

当前目录下的关键文件：

- `hot_loader.py`：核心热加载逻辑
- `server.py`：FastAPI 服务与 Web UI 入口
- `cli.py`：命令行入口
- `sample_config.json`：示例配置
- `static/index.html`：浏览器管理页
- `static/app.js`：前端交互逻辑
- `static/style.css`：页面样式
- `requirements.txt`：Python 依赖

运行时会默认使用以下路径：

- model repository：`runtime/model_repository`
- state file：`runtime/state.json`
- staging dir：`runtime/staging`

如果你希望通过环境变量把 PVC 挂载点作为模型目录使用，可以显式设置：

```bash
HOT_TRITON_MODEL_REPOSITORY=/repository
```

此时默认路径会变成：

- model repository：`/repository`
- state file：`/repository/.hot_loader/state.json`
- staging dir：`/repository/.staging`

## 前置条件

### 1. 宿主机依赖

- Python 3.10+
- `docker` CLI 可用
- 当前用户对 Docker daemon 有访问权限
- 已完成镜像仓库登录，例如 `docker login ccr.ccs.tencentyun.com`

### 2. 镜像内容约定

待热加载的镜像中必须包含标准 Triton model repository 目录结构，例如：

```text
/trt_models/
  unit_detector/
    config.pbtxt
    1/
      model.onnx
  sku_classifier/
    config.pbtxt
    1/
      model.plan
```

也就是说，热加载器假设镜像内部存在：

- `/trt_models/<model_name>/<version>/...`
- 可选的 `config.pbtxt`

如果走新的 `POST /models/load-from-image` 接口，服务端会按单模型复制：

- source path：`/trt_models/{model_name}`
- target path：`runtime/model_repository/{model_name}`（Triton 容器内通常挂载为 `/models/{model_name}`）

如果同时设置了 `HOT_TRITON_MODEL_REPOSITORY=/repository`，这里的 target path 会对应变成：

- `/repository/{model_name}`

> 第一版不会自动生成 `config.pbtxt`，请确保模型镜像里已经准备好 Triton 可识别的目录结构。

## Triton 启动方式

### 关键要求：必须使用 EXPLICIT 模式，且禁用 polling

如果希望通过 Triton API 手动加载或卸载模型，**必须**启用 `model-control-mode=EXPLICIT`。

同时要注意：**不能让 repository polling 处于开启状态**。如果你设置了 `--repository-poll-secs` 且值大于 `0`，或者服务实际处于 polling 模式，那么 Triton 会拒绝 `/load` 和 `/unload` API。

请在文档或命令里明确保留这句：

```bash
tritonserver --model-control-mode=EXPLICIT --repository-poll-secs=0 # 通过 API 加载
```

在这个项目里，建议完整启动方式如下：

```bash
docker run -d \
  --name hot_triton_server \
  --gpus=all \
  --ipc=host \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /home/haoyu/projects/hot_triton/runtime/model_repository:/models \
  ccr.ccs.tencentyun.com/clobotics/tritonserver:24.12-py3 \
  tritonserver \
    --model-repository=/models \
    --model-control-mode=EXPLICIT \
    --repository-poll-secs=0 \
    --strict-model-config=false
```

> 上面的宿主机路径 `/home/haoyu/projects/hot_triton/runtime/model_repository` 必须和热加载器使用的是同一个目录，这样热加载器写入文件后，Triton 容器才能实时看到。
>
> 另外，`--repository-poll-secs=0` 很重要：这表示关闭 polling，由 hot loader 通过 API 显式管理模型生命周期。

### 启动后先做健康检查

```bash
curl http://127.0.0.1:8000/v2/health/ready
curl -X POST http://127.0.0.1:8000/v2/repository/index
```

## 安装

在你当前的 Python 环境中安装依赖：

```bash
pip install -r /home/haoyu/projects/hot_triton/requirements.txt
```

默认的 Triton URL 现在支持从环境变量或项目根目录的 `.env` 读取，推荐直接使用：

```bash
TRT_IP=127.0.0.1
HTTP_PORT=8000
METRICS_PORT=8002
```

这会自动派生出：

- Triton HTTP endpoint：`http://127.0.0.1:8000`
- Metrics endpoint：`http://127.0.0.1:8002/metrics`

如果你已经有完整 URL，也仍然兼容：

```bash
HOT_TRITON_TRITON_URL=http://127.0.0.1:8000
# 可选：HOT_TRITON_TRITON_METRICS_URL=http://127.0.0.1:8002/metrics
```

如果浏览器页面里手动保存过新的 Triton URL 或 Metrics 端口，则当前浏览器会优先使用你保存的值；未保存时才回落到环境变量 / `.env` / 启动参数。

## 启动 Web UI

```bash
python /home/haoyu/projects/hot_triton/cli.py serve \
  --host 0.0.0.0 \
  --port 8090 \
  --triton-url http://127.0.0.1:8000
```

启动后访问：

- `http://127.0.0.1:8090`

UI 支持：

- 直接粘贴 JSON 配置并执行热加载
- Triton URL 可在页面中直接编辑并保存，刷新页面后仍会记住
- Metrics 端口可单独配置；留空时会自动按 Triton URL 与 `+2` 端口探测
- 显示当前操作进度条、分阶段提示和最近进展
- 查看 Triton Ready 状态
- 查看已管理镜像列表
- 按模型分组查看所有版本、当前激活版本与来源镜像
- 查看 Triton `repository/index`
- 勾选具体模型名做热卸载或重载
- 勾选具体 `model@version` 做单版本卸载

## UI 使用流程

1. 先确保 Triton 已按 `EXPLICIT` 模式启动。
2. 启动 Web UI。
3. 打开页面后，点击“填充示例 JSON”或手动粘贴配置。
4. 如需让“本次 JSON 里已删除的镜像”自动下线，手动勾选“自动卸载”。
5. 点击“执行热加载”。
6. 观察页面中的：
   - Triton Ready 状态
  - 已管理镜像列表
  - 模型版本管理视图
   - Triton 当前模型列表
   - 最近一次操作结果
7. 如需热卸载：
  - 勾选模型标题后点击“卸载选中模型”
  - 或勾选具体版本后点击“卸载选中模型/版本”

## CLI 命令

### 1. 启动 UI/API 服务

```bash
python /home/haoyu/projects/hot_triton/cli.py serve \
  --host 0.0.0.0 \
  --port 8090
```

### 2. 通过配置文件执行热加载

```bash
python /home/haoyu/projects/hot_triton/cli.py apply \
  --config-file /home/haoyu/projects/hot_triton/sample_config.json
```

### 3. 直接传 JSON 字符串执行热加载

```bash
python /home/haoyu/projects/hot_triton/cli.py apply \
  --json '{"unit_model":"ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hpc_us_yolov8-20260107"}'
```

### 4. 查询状态

```bash
python /home/haoyu/projects/hot_triton/cli.py status
```

### 5. 查看已管理镜像与 Triton 当前模型

```bash
python /home/haoyu/projects/hot_triton/cli.py list
```

### 6. 按 alias 热卸载

```bash
python /home/haoyu/projects/hot_triton/cli.py unload \
  --aliases unit_model sku_model
```

### 7. 按模型名热卸载

```bash
python /home/haoyu/projects/hot_triton/cli.py unload \
  --models unit_detector sku_classifier
```

### 8. 按指定版本热卸载

```bash
python /home/haoyu/projects/hot_triton/cli.py unload \
  --versions unit_detector@20260530 sku_classifier@20260527
```

> 说明：Triton 原生 `/unload` 只能按模型名卸载，不能直接按版本卸载；这里的 `model@version` 能力由 `hot_triton` 自己完成：删除对应版本目录、更新 `version_policy`，再触发一次 `load` 做 reload。

### 9. 按模型名重载

```bash
python /home/haoyu/projects/hot_triton/cli.py reload \
  unit_detector sku_classifier
```

### 10. 若不想自动卸载本次配置中缺失的 alias

```bash
python /home/haoyu/projects/hot_triton/cli.py apply \
  --config-file /home/haoyu/projects/hot_triton/sample_config.json \
  --no-prune-missing
```

### 11. 若镜像地址相同也要强制重载

```bash
python /home/haoyu/projects/hot_triton/cli.py apply \
  --config-file /home/haoyu/projects/hot_triton/sample_config.json \
  --force
```

## HTTP API

除了现有 `/api/*` 路径外，服务端也提供更贴近 controller 方案的兼容接口。

### `GET /runtime/health`

查看当前 Triton Ready 状态与实际连接的 Triton URL。

```bash
curl http://127.0.0.1:8090/runtime/health
```

### `GET /runtime/gpu-status`

返回按 GPU 聚合后的显存、利用率和功耗摘要。

```bash
curl http://127.0.0.1:8090/runtime/gpu-status
```

### `GET /api/status`

查看 Triton 连通性、已管理镜像数量、model repository 路径等。

```bash
curl http://127.0.0.1:8090/api/status
```

### `GET /api/state`

查看热加载器自己的状态文件内容摘要。

```bash
curl http://127.0.0.1:8090/api/state
```

### `GET /api/models`

查看：

- 当前已管理镜像
- Triton `repository/index` 返回的模型列表

```bash
curl http://127.0.0.1:8090/api/models
```

### `POST /api/apply-config`

```bash
curl -X POST http://127.0.0.1:8090/api/apply-config \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "unit_model": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_hpc_us_yolov8-20260107",
      "sku_model": "ccr.ccs.tencentyun.com/clobotics/sku-model-init:sku_hpc_us_resnet-20260527"
    },
    "prune_missing": true,
    "force": false
  }'
```

### `POST /models/load-from-image`

按新方案里的 `models[]` 结构，串行执行“拉镜像 -> 复制单模型 -> 可选 load”：

```bash
curl -X POST http://127.0.0.1:8090/models/load-from-image \
  -H 'Content-Type: application/json' \
  -d '{
    "triton_url": "http://127.0.0.1:8000",
    "triton_metrics_url": "http://127.0.0.1:8002/metrics",
    "models": [
      {
        "model_name": "unit_detector",
        "image": "registry.example.com/team/unit-detector:20260605"
      },
      {
        "model_name": "sku_classifier",
        "image": "registry.example.com/team/sku-classifier:20260605"
      }
    ],
    "options": {
      "load_after_copy": true,
      "overwrite": true
    }
  }'
```

说明：

- 这里按 `model_name` 从镜像内复制 `/trt_models/{model_name}`
- `overwrite=false` 时，如果目标模型目录已存在会直接报错
- `load_after_copy=false` 时只复制共享目录，不触发 Triton `load`

### `POST /api/unload`

按 alias 卸载：

```bash
curl -X POST http://127.0.0.1:8090/api/unload \
  -H 'Content-Type: application/json' \
  -d '{"aliases": ["unit_model"], "models": []}'
```

按模型名卸载：

```bash
curl -X POST http://127.0.0.1:8090/api/unload \
  -H 'Content-Type: application/json' \
  -d '{"aliases": [], "models": ["unit_detector"]}'
```

按指定版本卸载：

```bash
curl -X POST http://127.0.0.1:8090/api/unload \
  -H 'Content-Type: application/json' \
  -d '{"aliases": [], "models": [], "versions": ["unit_detector@20260530"]}'
```

### `POST /api/reload`

```bash
curl -X POST http://127.0.0.1:8090/api/reload \
  -H 'Content-Type: application/json' \
  -d '{"models": ["unit_detector"]}'
```

## 直接调用 Triton API 的常用命令

有时候你也会想跳过 UI，直接看看 Triton 本身的状态，这几个命令很有用：

```bash
curl http://127.0.0.1:8000/v2/health/ready
curl -X POST http://127.0.0.1:8000/v2/repository/index
curl -X POST http://127.0.0.1:8000/v2/repository/models/unit_detector/load
curl -X POST http://127.0.0.1:8000/v2/repository/models/unit_detector/unload
```

## 实际热加载流程

热加载器执行 `apply` 时的大致过程如下：

1. 接收一份 `占位键 -> image` 的 JSON 配置，并忽略 key。
2. 对每个需要更新的镜像执行 `docker pull`。
3. 使用 `docker create` + `docker cp` 从镜像中提取模型目录。
4. 扫描 bundle 中包含的 Triton 模型目录。
5. 将模型目录写入共享的 model repository。
6. 对应模型调用 Triton API：
  - 替换共享 repository 中的模型目录
  - 若存在 `config.pbtxt`，写入/更新 `version_policy: specific`
  - 直接调用 `load` 触发 Triton reload，并激活目标版本
7. 将镜像与模型名、版本信息的映射关系写入 `runtime/state.json`。

如果走 `POST /models/load-from-image`，第 3 步会进一步收敛为按单模型复制：

- source path：`/trt_models/{model_name}`
- target path：`runtime/model_repository/{model_name}`

## 验证与测试建议

建议至少做以下验证：

1. **Triton 启动时间**
   - 记录容器启动到 `/v2/health/ready` 返回 200 所需时间

2. **模型切换时间**
   - 记录一次 `apply` 请求从提交到返回完成所需时间

3. **切换前后正确性**
   - 切换前后分别调用业务推理接口，确认返回符合预期

4. **业务影响窗口**
   - 关注模型 `unload -> load` 间的短暂窗口，观察是否会影响请求成功率

5. **切换过程中的服务可用性**
   - 在切换时并发压测，关注延迟和错误码

## 注意事项

- 确保新的模型镜像已经正确构建并推送到镜像仓库。
- 热加载过程中可能存在短暂抖动，建议在低流量时段切换。
- 当前实现假设宿主机已经完成 `docker login`，UI 不负责输入仓库用户名密码。
- 当前实现会阻止两个不同镜像同时接管同一个 Triton 模型名，避免互相覆盖。
- 当模型目录内存在多个数字版本子目录时，当前实现会默认激活最大的版本号，并在 `state.json` 中记录 `active_versions`。
- 如果模型目录包含 `config.pbtxt`，热加载器会自动改写其中的 `version_policy`，把激活版本固定到本次选中的版本。
- 通过 `model@version` 卸载时，如果同模型还有其它版本残留，热加载器会保留目录、重算激活版本并触发 Triton reload；如果这是最后一个版本，则会下线整个模型。
- 通过“按模型名卸载”时，如果对应模型目录位于当前共享 repository 中，也会一并删除该目录。
- 如果你手工往 repository 里塞了非本工具管理的模型，依然可以在 Triton 里看到；但是否应该用本工具卸载它们，请先确认再按按钮，别让模型无辜“下班”。

## 常见报错与排障

### 1. `explicit model load / unload is not allowed if polling is enabled`

如果你看到类似报错：

```json
{
  "error": "Triton API 调用失败: POST /v2/repository/models/unit_hpc_us_yolov8/unload -> 503\n{\"error\":\"explicit model load / unload is not allowed if polling is enabled\"}"
}
```

说明当前 Triton **没有处于“可显式管理模型”的状态**。常见原因有两个：

1. `--model-control-mode` 不是 `EXPLICIT`
2. 开启了 `--repository-poll-secs > 0`

解决办法：

1. 停掉当前 Triton 容器或进程
2. 用下面这种方式重启：

```bash
docker run -d \
  --name hot_triton_server \
  --gpus=all \
  --shm-size=1g \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /home/haoyu/projects/hot_triton/runtime/model_repository:/models \
  ccr.ccs.tencentyun.com/clobotics/tritonserver:24.12-py3 \
  tritonserver \
    --model-repository=/models \
    --model-control-mode=EXPLICIT \
    --repository-poll-secs=0 \
    --strict-model-config=false
```

3. 再次确认：

```bash
curl http://127.0.0.1:8000/v2/health/ready
curl -X POST http://127.0.0.1:8000/v2/repository/index
```

4. 然后重新在 UI 或 CLI 中执行热加载 / 热卸载

如果你之前是用其他脚本启动 Triton，请重点检查其中是否偷偷带了：

```bash
--repository-poll-secs=15
```

或者：

```bash
--model-control-mode=none
--model-control-mode=poll
```

## 快速开始

如果你只想先跑起来，按下面四步即可：

1. 启动 Triton，并确保：

   ```bash
  tritonserver --model-control-mode=EXPLICIT --repository-poll-secs=0 # 通过 API 加载
   ```

2. 安装依赖：

   ```bash
   pip install -r /home/haoyu/projects/hot_triton/requirements.txt
   ```

3. 启动 UI：

   ```bash
   python /home/haoyu/projects/hot_triton/cli.py serve --host 0.0.0.0 --port 8090
   ```

4. 打开浏览器访问 `http://127.0.0.1:8090`，粘贴 JSON 后点击“执行热加载”。
