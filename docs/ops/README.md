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

### 版本下线事务

指定版本下线会将目录移动到仓库外、同文件系统的 `.hot-loader-version-backups/<operation_id>/<model>/`。模型仓库必须位于挂载卷的子目录，例如挂载 `/repository`，仓库为 `/repository/trt_models`；直接把卷挂到仓库目录导致兄弟备份跨文件系统时会拒绝操作。

状态文件必须持久化。恢复时使用原始仓库路径、PVC 和维护镜像配置，不能把待确认事务迁到另一份仓库。共享目录的变更通过单 controller 协调；操作期间不要用其他工具直接修改目录或向同模型发送加载/卸载请求。

Job-only 模式的 `REPOSITORY_MAINTENANCE_IMAGE` 必须使用包含 `/app/version_transaction.py` 的新构建 controller 镜像（协议 1，Python 标准库即可运行）。维护 Job 顺序执行 inspect、prepare、commit 或 restore，`backoffLimit=0`，保留终态 Job 24 小时；controller 通过协议日志确认完成，不会把“Job 已提交”当作文件已改好。

网络超时不会触发立即恢复。后台确认全部剩余版本 READY 且移除版本已退出 READY；超过 `TRITON_RELOAD_TIMEOUT_SECONDS` 标为 `RECOVERY_REQUIRED`，仍保留备份和全局同名模型保护。明确失败恢复后为 `FAILED_RESTORED`，成功清理完成为 `SUCCEEDED`。具体记录可用 `/api/version-operations/{id}` 查看。

真实验收需对当前 Triton 持续发送未指定版本的推理请求，下线新版本后检查剩余版本与请求错误率；固定版本请求、序列模型会话和其他仍使用共享文件的实例应分别验证，不宣称所有场景都可无缝切换。

### 多 Triton 共享仓库部署

多实例模式要求 model-copy Job、controller 和每个 Triton 都能访问同一份模型文件。所有 Triton 的 `--model-repository` 必须指向同一 PVC 的同一模型子目录；容器内路径可以不同，底层文件必须相同。跨节点部署需存储提供方支持共享访问，通常使用 RWX 卷；不要把单节点 RWO 卷当成跨节点共享文件系统。

现有 `deploy/realtime-dev` 示例使用 Pod 内 `emptyDir` 作为在线仓库，适合单实例，不可直接扩成多个独立 Pod 的共享仓库。多实例推荐把 PVC 挂到 `/repository`，将所有 Triton 的仓库设为 `/repository/trt_models`，controller 使用：

```env
HOT_TRITON_MODEL_REPOSITORY=/repository/trt_models
MODEL_TARGET_PATH=/repository/trt_models
HOT_TRITON_STATE_FILE=/repository/.hot_loader/state.json
HOT_TRITON_STAGING_ROOT=/repository/.staging
TRITON_REPOSITORY_PVC=triton-models-storage
```

仅运行一个 controller 副本、一个 Uvicorn worker；当前共享文件状态与模型互斥锁不支持多 controller 进程。实例列表与 Job 共用原子写入的状态文件，迁移前备份此文件。首次启动登记默认实例并将旧 Job 归属默认实例；后续通过 `/api/instances` 编辑地址。

每个 Triton 使用 EXPLICIT 模式、关闭 polling，并确保 controller 能访问登记的 HTTP 与 Metrics 端口。实例 IP 必须指向对应服务器；不要为需要独立控制的实例填写会随机分流到多副本的 Service 地址。

验收时添加 A/B 两个实例，分别加载、卸载、重载同一模型，确认只有目标实例的运行态变化；检查 B 的任务在页面切回 A、controller 重启后仍在 B 完成，回调带 B 的实例 ID。暂时中断 A 后确认 B 的后台任务继续推进。共享配置和版本集合会影响后续 reload，不提供每实例独立版本策略。

### 单实例配置示例

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
- `TRITON_RELOAD_MAX_ATTEMPTS=8`
- `TRITON_RELOAD_RETRY_BASE_SECONDS=2`
- `TRITON_RELOAD_RETRY_MAX_SECONDS=60`
- `TRITON_RELOAD_TIMEOUT_SECONDS=600`
- `JOB_TOLERATIONS_JSON=[...]`

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
- Job 长时间处于 `TRITON_RELOAD_RUNNING`:
  查看 `triton_reload_attempts`、`triton_reload_next_attempt_at` 和 `error`；达到配置上限后会固定为 `TRITON_RELOAD_FAILED`
- Triton 仓库里出现 `.hot_loader` 或 `.staging`:
  把状态文件和 staging 目录移到模型目录外层

相关文档:

- [根 README](../../README.md)
- [CLI 使用说明](../cli/README.md)
- [realtime-dev 部署说明](../../deploy/realtime-dev/README.md)
