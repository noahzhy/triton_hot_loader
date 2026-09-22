# 单节点 K3s Job 测试环境

此目录在单机 K3s 中建立 `triton-job-test` 命名空间，用公开 `busybox:1.36` 镜像验证 Job 调度、PVC 挂载、日志、完成态、失败诊断和 TTL 清理。它不部署 Triton、Controller 或 GPU 工作负载，也不改动 `deploy/realtime-dev`。

## 1. 主机预检与安装

将本仓库同步到目标主机后，以 root 执行：

```bash
sudo ./deploy/k3s-job-test/bootstrap-k3s.sh
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl get nodes
kubectl get storageclass local-path
```

脚本会在发现已有 K3s、kubelet、端口 `6443` / `10250` 被占用，或 `/var/lib` 少于 10 GiB 可用空间时退出，不会覆盖现有集群。
默认经 Rancher 中国镜像安装 `v1.37.0+k3s1`，避免依赖 stable-channel 自动发现；如需指定其他 K3s 版本，可设置 `K3S_VERSION`。若目标主机没有 `/etc/rancher/k3s/registries.yaml`，脚本会创建 Docker Hub 的 Daocloud 镜像配置，确保公开 BusyBox 测试镜像可以拉取；已有的 registry 配置不会被改写。

## 2. 运行验证

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
./deploy/k3s-job-test/verify-jobs.sh
```

脚本依次验证：

1. `triton-job-test` 命名空间、资源限制和 128 MiB 动态 PVC。
2. 写入 Job 在 PVC 创建唯一 marker，读取 Job 从同一 PVC 读取完全相同的 marker。
3. 不存在的镜像保留为 `ImagePullBackOff` 诊断样例。
4. 错误挂载路径的 Job 以非零状态退出，便于检查 Job/Pod 事件。
5. 成功 Job 使用 `ttlSecondsAfterFinished: 30`，脚本确认它们已删除且 PVC 仍是 `Bound`。

```bash
kubectl -n triton-job-test get jobs,pods
```

## 3. 排障与清理

```bash
kubectl -n triton-job-test get events --sort-by=.lastTimestamp
kubectl -n triton-job-test describe job job-image-pull-failure
kubectl -n triton-job-test describe job job-wrong-mount-path
kubectl delete namespace triton-job-test
```

命名空间删除会清理测试 Job、Pod、PVC 和资源限制。它不会卸载 K3s；若不再需要集群，请在确认没有其他工作负载后执行 K3s 官方卸载脚本。
