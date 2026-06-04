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
        help=f"共享 model repository 路径，默认: {defaults.model_repository}",
    )
    parser.add_argument(
        "--state-file",
        default=str(defaults.state_file),
        help=f"状态文件路径，默认: {defaults.state_file}",
    )
    parser.add_argument(
        "--staging-root",
        default=str(defaults.staging_root),
        help=f"staging 目录路径，默认: {defaults.staging_root}",
    )
    parser.add_argument(
        "--image-model-root",
        default=defaults.image_model_root,
        help=f"镜像中的模型根目录，默认: {defaults.image_model_root}",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=defaults.request_timeout,
        help=f"Triton API 超时秒数，默认: {defaults.request_timeout}",
    )
    parser.add_argument(
        "--docker-binary",
        default=defaults.docker_binary,
        help=f"Docker 可执行文件名，默认: {defaults.docker_binary}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Triton hot loader：支持 UI、CLI 与 Triton API 的模型热加载器。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="启动 Web UI + API 服务")
    add_common_runtime_args(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1", help="Web UI 监听地址")
    serve_parser.add_argument("--port", type=int, default=8090, help="Web UI 监听端口")

    apply_parser = subparsers.add_parser("apply", help="提交一份 JSON 配置并执行热加载")
    add_common_runtime_args(apply_parser)
    apply_source = apply_parser.add_mutually_exclusive_group(required=True)
    apply_source.add_argument("--config-file", help="配置文件路径")
    apply_source.add_argument("--json", help="直接传入 JSON 字符串")
    apply_parser.add_argument(
        "--prune-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="未出现在本次 JSON 中的 alias 是否自动卸载（默认开启）",
    )
    apply_parser.add_argument(
        "--force",
        action="store_true",
        help="镜像地址未变化时也强制重载",
    )

    status_parser = subparsers.add_parser("status", help="查看 Triton 与 manager 状态")
    add_common_runtime_args(status_parser)

    list_parser = subparsers.add_parser("list", help="列出受管 alias 与 Triton repository/index")
    add_common_runtime_args(list_parser)

    unload_parser = subparsers.add_parser("unload", help="按 alias、模型名或指定版本卸载")
    add_common_runtime_args(unload_parser)
    unload_parser.add_argument("--aliases", nargs="*", default=[], help="要卸载的 alias 列表")
    unload_parser.add_argument("--models", nargs="*", default=[], help="要卸载的模型名列表")
    unload_parser.add_argument(
        "--versions",
        nargs="*",
        default=[],
        help="要卸载的版本列表，格式: model_name@123",
    )

    reload_parser = subparsers.add_parser("reload", help="重载指定模型")
    add_common_runtime_args(reload_parser)
    reload_parser.add_argument("models", nargs="+", help="要重载的 Triton model name")

    return parser


def build_config_from_args(args: argparse.Namespace) -> HotLoaderConfig:
    return HotLoaderConfig(
        triton_url=args.triton_url,
        triton_metrics_url=args.triton_metrics_url,
        model_repository=Path(args.model_repository),
        state_file=Path(args.state_file),
        staging_root=Path(args.staging_root),
        image_model_root=args.image_model_root,
        request_timeout=args.request_timeout,
        docker_binary=args.docker_binary,
    )


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def load_apply_payload(loader: TritonHotLoader, args: argparse.Namespace) -> dict[str, str]:
    if args.config_file:
        return loader.load_config_file(args.config_file)

    try:
        payload = json.loads(args.json)
    except json.JSONDecodeError as exc:
        raise HotLoaderError("--json 不是合法 JSON") from exc
    return loader._validate_config_map(payload)


def execute(args: argparse.Namespace) -> int:
    config = build_config_from_args(args)

    if args.command == "serve":
        start_server(config, host=args.host, port=args.port)
        return 0

    loader = TritonHotLoader(config)

    if args.command == "apply":
        payload = load_apply_payload(loader, args)
        print_json(
            loader.apply_config(
                payload,
                prune_missing=args.prune_missing,
                force=args.force,
            )
        )
        return 0

    if args.command == "status":
        print_json(loader.get_status())
        return 0

    if args.command == "list":
        print_json(
            {
                "managed_state": loader.get_managed_state(),
                "triton_models": loader.list_repository_models(safe=True),
            }
        )
        return 0

    if args.command == "unload":
        if not args.aliases and not args.models and not args.versions:
            raise HotLoaderError("unload 至少要提供 --aliases、--models 或 --versions 之一")

        payload = {}
        if args.aliases:
            payload["alias_result"] = loader.unload_aliases(args.aliases)
        if args.models:
            payload["model_result"] = loader.unload_models(args.models)
        if args.versions:
            payload["version_result"] = loader.unload_model_versions(args.versions)

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
