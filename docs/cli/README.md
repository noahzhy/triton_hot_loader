# Triton Hot Loader CLI

这份文档只说明这个项目的命令行入口 `cli.py`，用于补充主 [README.md](../../README.md) 里偏整体性的介绍。

默认假设你已经在项目根目录执行命令：

```bash
cd /Users/haoyu/Documents/Projects/triton_hot_loader
```

## 命令入口

查看总帮助：

```bash
python3 cli.py --help
```

当前支持的子命令：

- `serve`：启动 Web UI 和 HTTP API
- `apply`：提交配置并执行热加载
- `status`：查看 Triton 与 manager 当前状态
- `list`：查看受管状态与 Triton repository/index
- `unload`：按 bundle、模型名或指定版本卸载
- `reload`：对指定模型触发一次 Triton `load`

所有子命令都会把结果打印成 JSON；业务错误会输出到 stderr，前缀是 `[hot_triton]`。

## 运行前提

至少确认这几件事：

- Python 3.10+
- `docker` CLI 可用，并且当前用户可以访问 Docker daemon
- 已完成镜像仓库登录，例如 `docker login ...`
- Triton 已启动，并且使用 `EXPLICIT` 模式

推荐的 Triton 关键参数：

```bash
tritonserver --model-control-mode=EXPLICIT --repository-poll-secs=0
```

如果没有这两个条件，`load / unload` 相关命令会失败。

## 全局参数

除 `reload` 的模型名位置参数外，所有子命令都支持同一组运行时参数：

- `--triton-url`：Triton HTTP 地址，默认 `http://127.0.0.1:8000`
- `--triton-metrics-url`：Triton Prometheus metrics 地址
- `--model-repository`：共享模型目录，默认 `runtime/model_repository`
- `--state-file`：状态文件，默认 `runtime/state.json`
- `--staging-root`：解包临时目录，默认 `runtime/staging`
- `--image-model-root`：镜像里的模型根目录，默认 `/trt_models`
- `--request-timeout`：Triton API 超时秒数，默认 `60`
- `--docker-binary`：Docker 可执行文件名，默认 `docker`

这些参数的默认值来自 `HotLoaderConfig.default()`，也会读取环境变量或项目根目录 `.env`：

- `TRT_IP` + `HTTP_PORT`：派生 `triton-url`
- `METRICS_PORT`：派生 metrics 地址
- `HOT_TRITON_TRITON_URL` 或 `TRITON_URL`：直接指定 Triton URL
- `HOT_TRITON_TRITON_METRICS_URL`：直接指定 metrics URL

命令行参数优先级最高。

## 配置输入规则

`apply` 接收的是 `JSON object`，格式通常是：

```json
{
  "unit_model": "registry.example.com/team/unit-model:v1",
  "sku_model": "registry.example.com/team/sku-model:v2"
}
```

这里有三个容易误解的点：

1. JSON 的 key 只是占位符，真正参与管理的是 value 对应的镜像地址。
2. 同一个镜像地址会被去重，不会重复应用。
3. `mlman_config`、`mlmanconfig` 相关条目会自动跳过。

内部状态里仍然会生成一个稳定的 bundle id，格式类似 `bundle_<sha1前12位>`。

## 子命令详解

### `serve`

启动 Web UI 与 HTTP API 服务。

```bash
python3 cli.py serve --host 0.0.0.0 --port 8090
```

常见用法：

```bash
python3 cli.py serve \
  --host 0.0.0.0 \
  --port 8090 \
  --triton-url http://127.0.0.1:8000
```

说明：

- 这个命令会前台阻塞运行
- UI 首页默认是 `http://127.0.0.1:8090/`
- 同一个进程里也会提供 `/api/status`、`/api/models`、`/api/apply-config` 等接口

### `apply`

提交一份配置并执行热加载。必须二选一提供：

- `--config-file`
- `--json`

通过文件加载：

```bash
python3 cli.py apply --config-file sample_config.json
```

直接传 JSON：

```bash
python3 cli.py apply \
  --json '{"unit_model":"registry.example.com/team/unit-model:v1"}'
```

控制行为的两个关键参数：

- `--prune-missing` / `--no-prune-missing`
- `--force`

示例：保留本次配置之外的已加载镜像

```bash
python3 cli.py apply \
  --config-file sample_config.json \
  --no-prune-missing
```

示例：镜像地址没变也强制重新加载

```bash
python3 cli.py apply \
  --config-file sample_config.json \
  --force
```

返回结果里最常见的字段：

- `applied`：本次实际执行过的镜像
- `skipped`：镜像没变化而跳过的项
- `removed`：因 `prune_missing=true` 被自动卸载的 bundle
- `errors`：执行失败项
- `state`：最新受管状态快照

### `status`

查看 Triton 与 manager 当前状态：

```bash
python3 cli.py status
```

返回内容包含：

- `triton.url`
- `triton.ready`
- `triton.metrics`
- `triton.repository_models`
- `manager.config`
- `manager.managed_images`
- `manager.managed_models`

适合排查当前连到哪个 Triton、Triton 是否 ready、以及本地状态文件记录了什么。

### `list`

列出受管状态和 Triton 当前 repository/index：

```bash
python3 cli.py list
```

输出结构比 `status` 更轻，主要有：

- `managed_state`
- `triton_models`

如果你已经确认 Triton 可达，只想对照“本工具管理了什么”和“Triton 现在看到了什么”，优先用它。

### `unload`

支持三种目标，并且可以组合：

- `--aliases`
- `--models`
- `--versions`

按 bundle id 卸载：

```bash
python3 cli.py unload --aliases bundle_123456789abc
```

按模型名卸载：

```bash
python3 cli.py unload --models unit_detector sku_classifier
```

按指定版本卸载：

```bash
python3 cli.py unload --versions unit_detector@20260530 sku_classifier@20260527
```

组合卸载：

```bash
python3 cli.py unload \
  --models unit_detector \
  --versions sku_classifier@20260527
```

关于 `--versions` 需要注意：

- 格式必须是 `model_name@数字版本`
- 如果删除后该模型还有其他版本，工具会重写 `version_policy` 并触发一次 reload
- 如果这是最后一个版本，工具会直接卸载整个模型并清理对应状态

### `reload`

对一个或多个 Triton 模型名执行一次 `load`：

```bash
python3 cli.py reload unit_detector sku_classifier
```

特点：

- 会自动去重
- 不会先执行 `unload`
- 适合目录已准备好，只想通知 Triton 重新加载的场景

## 典型操作流程

### 1. 启动 UI/API

```bash
python3 cli.py serve --host 0.0.0.0 --port 8090
```

### 2. 首次热加载

```bash
python3 cli.py apply --config-file sample_config.json
```

### 3. 查看当前状态

```bash
python3 cli.py status
```

### 4. 只下线一个版本

```bash
python3 cli.py unload --versions unit_detector@20260530
```

### 5. 强制重新加载现有镜像

```bash
python3 cli.py apply --config-file sample_config.json --force
```

## 常见问题

### 报错：`explicit model load / unload is not allowed if polling is enabled`

说明 Triton 不是显式控制模式，或者开启了 polling。检查：

- `--model-control-mode=EXPLICIT`
- `--repository-poll-secs=0`

### 报错：`--json 不是合法 JSON`

说明 `apply --json` 传入的字符串不是有效 JSON。优先改用 `--config-file`，更不容易被 shell 转义坑到。

### 报错：`请至少提供一个 model name`

说明 `reload` 没传位置参数，或者 `unload --models` 最终没有实际模型名。

### 报错：`未找到命令: 'docker'`

说明当前环境里找不到 Docker CLI，可以显式传：

```bash
python3 cli.py status --docker-binary /usr/bin/docker
```

## 相关源码

- [cli.py](../../cli.py)
- [hot_loader.py](../../hot_loader.py)
- [server.py](../../server.py)
- [sample_config.json](../../sample_config.json)
