from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hot_loader import HotLoaderConfig, HotLoaderError, TritonHotLoader
from server import start_server


def default_config() -> HotLoaderConfig:
    return HotLoaderConfig.default()


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    defaults = default_config()
    parser.add_argument(
        "--triton-url",
        default=defaults.triton_url,
        help=f"Triton HTTP endpoint，默认: {defaults.triton_url}",
    )
    parser.add_argument(
        "--triton-metrics-url",
        default=defaults.triton_metrics_url,
        help="Triton Prometheus metrics 地址（默认自动尝试 triton-url/metrics 与端口+2 的 /metrics）",
    )
    parser.add_argument(
        "--model-repository",
        default=str(defaults.model_repository),
        help=f"Controller 可见的模型仓库路径，默认: {defaults.model_repository}",
    )
    parser.add_argument(
        "--state-file",
        default=str(defaults.state_file),
        help=f"状态文件路径，默认: {defaults.state_file}",
    )
    parser.add_argument(
        "--staging-root",
        default=str(defaults.staging_root),
        help=f"临时操作目录，默认: {defaults.staging_root}",
    )
    parser.add_argument(
        "--model-source-path",
        default=defaults.model_source_path,
        help=f"Job 容器内模型源目录，默认: {defaults.model_source_path}",
    )
    parser.add_argument(
        "--model-target-path",
        default=defaults.model_target_path,
        help=f"Job 容器内模型目标目录，默认: {defaults.model_target_path}",
    )
    parser.add_argument(
        "--triton-repository-pvc",
        default=defaults.triton_repository_pvc,
        help=f"模型仓库 PVC 名称，默认: {defaults.triton_repository_pvc}",
    )
    parser.add_argument(
        "--k8s-namespace",
        default=defaults.k8s_namespace,
        help=f"Controller 运行的 Kubernetes namespace，默认: {defaults.k8s_namespace}",
    )
    parser.add_argument(
        "--model-image-registry-prefix",
        default=defaults.model_image_registry_prefix,
        help=f"允许的镜像前缀，默认: {defaults.model_image_registry_prefix}",
    )
    parser.add_argument(
        "--job-ttl-seconds-after-finished",
        type=int,
        default=defaults.job_ttl_seconds_after_finished,
        help=f"Job 完成后的保留秒数，默认: {defaults.job_ttl_seconds_after_finished}",
    )
    parser.add_argument(
        "--job-backoff-limit",
        type=int,
        default=defaults.job_backoff_limit,
        help=f"Job backoffLimit，默认: {defaults.job_backoff_limit}",
    )
    parser.add_argument(
        "--model-copy-cpu-request",
        default=defaults.model_copy_cpu_request,
        help=f"Job CPU request，默认: {defaults.model_copy_cpu_request}",
    )
    parser.add_argument(
        "--model-copy-memory-request",
        default=defaults.model_copy_memory_request,
        help=f"Job Memory request，默认: {defaults.model_copy_memory_request}",
    )
    parser.add_argument(
        "--model-copy-cpu-limit",
        default=defaults.model_copy_cpu_limit,
        help=f"Job CPU limit，默认: {defaults.model_copy_cpu_limit}",
    )
    parser.add_argument(
        "--model-copy-memory-limit",
        default=defaults.model_copy_memory_limit,
        help=f"Job Memory limit，默认: {defaults.model_copy_memory_limit}",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=defaults.max_concurrent_jobs,
        help=f"允许同时运行的 Job 上限，默认: {defaults.max_concurrent_jobs}（0 表示不限制）",
    )
    parser.add_argument(
        "--repository-maintenance-image",
        default=defaults.repository_maintenance_image,
        help="Job-only repository 模式下用于清理 PVC 模型目录的辅助镜像",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=defaults.request_timeout,
        help=f"Triton API 超时秒数，默认: {defaults.request_timeout}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Triton Hot Loader Controller：通过 Kubernetes Job 将模型镜像复制到 PVC 并触发 Triton 热加载。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="启动 Web UI + API 服务")
    add_common_runtime_args(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1", help="Web UI 监听地址")
    serve_parser.add_argument("--port", type=int, default=8090, help="Web UI 监听端口")

    load_parser = subparsers.add_parser("load", help="为单个模型创建 model-copy Job")
    add_common_runtime_args(load_parser)
    load_parser.add_argument("--model-name", help="Triton 模型名；不传则根据 image tag 自动提取")
    load_parser.add_argument("--image", required=True, help="模型初始化镜像")

    batch_parser = subparsers.add_parser("load-batch", help="批量创建 model-copy Job")
    add_common_runtime_args(batch_parser)
    batch_source = batch_parser.add_mutually_exclusive_group(required=True)
    batch_source.add_argument("--json", help='直接传入 JSON，格式: {"models":[...]} 或 [...]')
    batch_source.add_argument("--file", help="批量任务 JSON 文件路径")

    job_status_parser = subparsers.add_parser("job-status", help="查询单个 Job 状态")
    add_common_runtime_args(job_status_parser)
    job_status_parser.add_argument("job_name", help="Kubernetes Job 名称")

    status_parser = subparsers.add_parser("status", help="查看 Triton 与 controller 状态")
    add_common_runtime_args(status_parser)

    list_parser = subparsers.add_parser("list", help="列出已管理模型与 Triton repository/index")
    add_common_runtime_args(list_parser)

    unload_parser = subparsers.add_parser("unload", help="按 alias 或模型名卸载")
    add_common_runtime_args(unload_parser)
    unload_parser.add_argument("--aliases", nargs="*", default=[], help="要卸载的 alias 列表")
    unload_parser.add_argument("--models", nargs="*", default=[], help="要卸载的模型名列表")
    unload_parser.add_argument(
        "--versions",
        nargs="*",
        default=[],
        help="已废弃；同名模型不再支持按版本管理和卸载",
    )

    reload_parser = subparsers.add_parser("reload", help="重载指定模型")
    add_common_runtime_args(reload_parser)
    reload_parser.add_argument("models", nargs="+", help="要重载的 Triton model name")

    return parser


def build_config_from_args(args: argparse.Namespace) -> HotLoaderConfig:
    defaults = default_config()
    return defaults.with_updates(
        triton_url=args.triton_url,
        triton_metrics_url=args.triton_metrics_url,
        model_repository=Path(args.model_repository),
        state_file=Path(args.state_file),
        staging_root=Path(args.staging_root),
        model_source_path=args.model_source_path,
        model_target_path=args.model_target_path,
        triton_repository_pvc=args.triton_repository_pvc,
        k8s_namespace=args.k8s_namespace,
        model_image_registry_prefix=args.model_image_registry_prefix,
        job_ttl_seconds_after_finished=args.job_ttl_seconds_after_finished,
        job_backoff_limit=args.job_backoff_limit,
        model_copy_cpu_request=args.model_copy_cpu_request,
        model_copy_memory_request=args.model_copy_memory_request,
        model_copy_cpu_limit=args.model_copy_cpu_limit,
        model_copy_memory_limit=args.model_copy_memory_limit,
        max_concurrent_jobs=args.max_concurrent_jobs,
        repository_maintenance_image=args.repository_maintenance_image,
        request_timeout=args.request_timeout,
    )


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _load_batch_payload(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.file:
        raw_text = Path(args.file).expanduser().read_text(encoding="utf-8")
    else:
        raw_text = args.json

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HotLoaderError("批量加载 JSON 不是合法 JSON") from exc

    if isinstance(payload, dict):
        models = payload.get("models")
    else:
        models = payload

    if not isinstance(models, list) or not models:
        raise HotLoaderError("批量加载 JSON 必须包含非空 models 列表")

    normalized = []
    for item in models:
        if not isinstance(item, dict):
            raise HotLoaderError("models 列表中的每一项都必须是对象")
        normalized.append(
            {
                "model_name": str(item.get("model_name", "")),
                "image": str(item.get("image", "")),
            }
        )
    return normalized


def execute(args: argparse.Namespace) -> int:
    config = build_config_from_args(args)

    if args.command == "serve":
        start_server(config, host=args.host, port=args.port)
        return 0

    loader = TritonHotLoader(config)

    if args.command == "load":
        print_json(loader.create_model_copy_job(args.model_name or "", args.image))
        return 0

    if args.command == "load-batch":
        print_json(loader.load_models_from_images(_load_batch_payload(args)))
        return 0

    if args.command == "job-status":
        print_json(loader.get_job_status(args.job_name))
        return 0

    if args.command == "status":
        print_json(loader.get_status())
        return 0

    if args.command == "list":
        print_json(loader.get_models_overview())
        return 0

    if args.command == "unload":
        if not args.aliases and not args.models and not args.versions:
            raise HotLoaderError("unload 至少要提供 --aliases 或 --models 之一")
        if args.versions:
            raise HotLoaderError("同名模型已取消版本管理，请改用 --models 或 --aliases")

        payload = {}
        if args.aliases:
            payload["alias_result"] = loader.unload_aliases(args.aliases)
        if args.models:
            payload["model_result"] = loader.unload_models(args.models)

        if len(payload) == 1:
            print_json(next(iter(payload.values())))
            return 0

        print_json(payload)
        return 0

    if args.command == "reload":
        print_json(loader.reload_models(args.models))
        return 0

    raise HotLoaderError(f"未知命令: {args.command}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return execute(args)
    except HotLoaderError as exc:
        print(f"[hot_triton] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
