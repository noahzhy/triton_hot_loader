---
name: "triton-hot-loader"
description: "Use when working in this repository on Triton hot-loading workflows, especially to explain or call the HTTP API, document or debug `cli.py`, update ops docs, debug `load/load-batch/job-status/unload/reload`, or reason about runtime state, PVC sync, and Triton repository behavior."
---

# Triton Hot Loader

这个 skill 用于当前仓库。

## 何时使用

- 用户询问这个仓库的 CLI 如何使用
- 用户询问 HTTP API 如何调用或如何编写 API 文档
- 用户询问 callback 载荷、轮询方式、健康检查或状态接口
- 用户询问为什么设计上同时使用 PVC 和临时运行目录
- 用户需要命令示例、文档更新或 README 调整
- 用户需要排查热加载、卸载、重载或状态文件相关问题
- 用户询问这个运行时对应的 Web UI 或 HTTP API

## 事实来源

在相信文档说明之前，先读代码：

- `cli.py`：命令入口与参数定义
- `hot_loader.py`：`load/unload/reload/state` 和 repository sync 的真实行为
- `server.py`：HTTP API 路由、请求模型、callback 流程与 header override
- `references/api-ops.md`：API 调用方式、请求体、callback 语义和 PVC/runtime 说明
- `docs/ops/README.md`：面向运维的简要部署说明
- `docs/cli/README.md`：面向使用者的 CLI 说明
- `tests/test_hot_loader.py` 与 `tests/test_server.py`：用于校验文档没有和边界行为冲突

其中：

- `cli.py` 是命令入口和参数定义
- `hot_loader.py` 是 `load/unload/reload/state` 和 repository sync 的真实行为
- `server.py` 是 HTTP API、请求模型、callback 流程和 header override 的真实来源
- `references/api-ops.md` 用来沉淀 API 调用方式、请求体、callback 语义和 PVC/runtime 说明
- `docs/ops/README.md` 是面向运维的简要部署说明
- `docs/cli/README.md` 是面向使用者的 CLI 说明
- `tests/test_hot_loader.py` 和 `tests/test_server.py` 用于确认文档没有和边界行为冲突

## 仓库事实

- CLI 入口是 `python3 cli.py`
- 当前子命令是 `serve`、`load`、`load-batch`、`job-status`、`status`、`list`、`unload`、`reload`
- 所有子命令共享 `add_common_runtime_args(...)` 里定义的运行时参数
- 默认运行时路径在 `runtime/` 下
- 编写文档或给 curl 示例时，优先使用 `/api/...`，`/models/...` 仅作为兼容别名
- `load` 使用 `--image`，`--model-name` 可选
- `load-batch` 使用 `--file` 或 `--json`
- `reload` 只触发 Triton `load`，不会先显式 `unload`
- `unload --versions` 已废弃且会被拒绝，应该改用 `--models` 或 `--aliases`
- HTTP API 上 `wait_for_ready` 默认是 `false`
- callback 当前只支持终态事件
- 如果 `HOT_TRITON_MODEL_REPOSITORY` 与 `MODEL_TARGET_PATH` 不同，controller 会进入 repository sync 模式

## Triton 运行规则

在判断 load 或 unload 是否应该生效之前，先确认 Triton 启动参数满足：

- `--model-control-mode=EXPLICIT`
- `--repository-poll-secs=0`

如果用户看到 `explicit model load / unload is not allowed if polling is enabled`，优先从这两个参数排查。

## 工作流

1. 在写 CLI 说明前，先看 `cli.py`，或者先运行 `python3 cli.py --help`。
2. 遇到 API 问题时，先看 `references/api-ops.md`，再到 `server.py` 核对有歧义的请求体或响应细节。
3. 如果文档和代码不一致，以 `server.py` 的请求模型和路由、以及 `hot_loader.py` 的运行时行为为准。
4. 命令示例优先写成 `python3 cli.py load --image ...` 这种相对路径形式。
5. 更新文档时，命令和路径要和当前仓库名 `triton_hot_loader` 保持一致，不要写旧的 `hot_triton`。
6. 如果代码行为发生变化，要同步更新文档和测试。

## 校验

- `python3 cli.py --help`
- `python3 cli.py <subcommand> --help`
- `python3 cli.py load --help`
- `python3 cli.py load-batch --help`
- 当行为或 API 契约变更时，运行 `python3 -m unittest tests.test_hot_loader tests.test_server`
