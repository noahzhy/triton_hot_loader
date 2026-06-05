from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import quote, urlsplit, urlunsplit

import httpx


class HotLoaderError(RuntimeError):
    """Raised when a hot-loading operation fails."""


_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_TRITON_VERSION_DIR_PATTERN = re.compile(r"^\d+$")
_MODEL_VERSION_REF_PATTERN = re.compile(r"^(?P<model>[^@\s][^@]*)@(?P<version>\d+)$")
_LOAD_UNLOAD_PATH_PATTERN = re.compile(r"^/v2/repository/models/.+/(load|unload)$")
_MLMAN_CONFIG_PATTERN = re.compile(r"mlman(?:_|-)?config|mlmanconfig", re.IGNORECASE)
_PROMETHEUS_SAMPLE_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_PROMETHEUS_LABEL_PATTERN = re.compile(
    r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:\\.|[^"])*)"'
)
_DEFAULT_TRITON_URL = "http://127.0.0.1:8000"
_TRITON_URL_ENV_NAMES = ("HOT_TRITON_TRITON_URL", "TRITON_URL")
_TRITON_METRICS_URL_ENV_NAMES = ("HOT_TRITON_TRITON_METRICS_URL",)
_TRITON_HOST_ENV_NAMES = ("TRT_IP",)
_TRITON_HTTP_PORT_ENV_NAMES = ("HTTP_PORT",)
_TRITON_METRICS_PORT_ENV_NAMES = ("METRICS_PORT",)
_RUNTIME_ROOT_ENV_NAMES = ("HOT_TRITON_RUNTIME_ROOT",)
_MODEL_REPOSITORY_ENV_NAMES = ("HOT_TRITON_MODEL_REPOSITORY",)
_STATE_FILE_ENV_NAMES = ("HOT_TRITON_STATE_FILE",)
_STAGING_ROOT_ENV_NAMES = ("HOT_TRITON_STAGING_ROOT",)

_EXPLICIT_CONTROL_HINT = (
    "当前 Triton 不允许通过 API 显式执行 load/unload。\n"
    "请确认以下三点：\n"
    "1. Triton 使用 --model-control-mode=EXPLICIT 启动；\n"
    "2. 不要开启 repository polling；如果设置了 --repository-poll-secs，请删除该参数或显式设为 0；\n"
    "3. 修改启动参数后需要重启 Triton。\n"
    "推荐启动方式：\n"
    "tritonserver --model-repository=/models --model-control-mode=EXPLICIT --repository-poll-secs=0"
)


def _load_dotenv_values(env_file: Path) -> Dict[str, str]:
    if not env_file.exists():
        return {}

    try:
        raw_lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    values: Dict[str, str] = {}
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value

    return values


def _env_default(*names: str) -> str | None:
    dotenv_values = _load_dotenv_values(Path(__file__).resolve().parent / ".env")

    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()

    for name in names:
        value = dotenv_values.get(name)
        if value and value.strip():
            return value.strip()

    return None


def _normalize_host_for_url(host: str) -> str:
    value = host.strip()
    if not value:
        raise HotLoaderError("TRT_IP 不能为空")

    if "://" in value:
        parsed = urlsplit(value)
        if not parsed.hostname:
            raise HotLoaderError(f"TRT_IP 格式不正确: {host!r}")
        return parsed.hostname

    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]

    if value.count(":") == 1 and not value.startswith("["):
        maybe_host, maybe_port = value.rsplit(":", 1)
        if maybe_port.isdigit() and maybe_host:
            return maybe_host

    return value


def _format_host_for_url(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _normalize_port_for_url(port_text: str, *, env_name: str) -> int:
    candidate = port_text.strip()
    if not candidate.isdigit():
        raise HotLoaderError(f"{env_name} 必须是 1-65535 的整数")

    port = int(candidate)
    if port < 1 or port > 65535:
        raise HotLoaderError(f"{env_name} 必须在 1-65535 之间")
    return port


def _build_url_from_host_and_port(host: str, port: int, *, path: str = "") -> str:
    normalized_host = _format_host_for_url(_normalize_host_for_url(host))
    normalized_path = path or ""
    return f"http://{normalized_host}:{port}{normalized_path}"


def _derived_triton_urls_from_env() -> tuple[str | None, str | None]:
    host = _env_default(*_TRITON_HOST_ENV_NAMES)
    http_port = _env_default(*_TRITON_HTTP_PORT_ENV_NAMES)
    metrics_port = _env_default(*_TRITON_METRICS_PORT_ENV_NAMES)

    if not host and not http_port and not metrics_port:
        return None, None
    if not host or not http_port:
        raise HotLoaderError("使用 TRT_IP 派生 Triton URL 时，必须同时提供 HTTP_PORT")

    triton_url = _build_url_from_host_and_port(
        host,
        _normalize_port_for_url(http_port, env_name="HTTP_PORT"),
    )

    metrics_url = None
    if metrics_port:
        metrics_url = _build_url_from_host_and_port(
            host,
            _normalize_port_for_url(metrics_port, env_name="METRICS_PORT"),
            path="/metrics",
        )

    return triton_url, metrics_url


def _default_runtime_paths(
    base_dir: Path,
) -> tuple[Path, Path, Path]:
    explicit_runtime_root = _env_default(*_RUNTIME_ROOT_ENV_NAMES)
    explicit_model_repository = _env_default(*_MODEL_REPOSITORY_ENV_NAMES)
    explicit_state_file = _env_default(*_STATE_FILE_ENV_NAMES)
    explicit_staging_root = _env_default(*_STAGING_ROOT_ENV_NAMES)

    if explicit_runtime_root:
        runtime_root = Path(explicit_runtime_root).expanduser()
        model_repository = (
            Path(explicit_model_repository).expanduser()
            if explicit_model_repository
            else runtime_root / "model_repository"
        )
        state_file = (
            Path(explicit_state_file).expanduser()
            if explicit_state_file
            else runtime_root / "state.json"
        )
        staging_root = (
            Path(explicit_staging_root).expanduser()
            if explicit_staging_root
            else runtime_root / "staging"
        )
        return model_repository, state_file, staging_root

    if explicit_model_repository:
        model_repository = Path(explicit_model_repository).expanduser()
        state_file = (
            Path(explicit_state_file).expanduser()
            if explicit_state_file
            else model_repository / ".hot_loader" / "state.json"
        )
        staging_root = (
            Path(explicit_staging_root).expanduser()
            if explicit_staging_root
            else model_repository / ".staging"
        )
        return model_repository, state_file, staging_root

    model_repository = base_dir / "model_repository"
    state_file = (
        Path(explicit_state_file).expanduser()
        if explicit_state_file
        else base_dir / "state.json"
    )
    staging_root = (
        Path(explicit_staging_root).expanduser()
        if explicit_staging_root
        else base_dir / "staging"
    )
    return model_repository, state_file, staging_root


@dataclass
class HotLoaderConfig:
    """Configuration for the Triton hot loader."""

    triton_url: str
    model_repository: Path
    state_file: Path
    staging_root: Path
    triton_metrics_url: str | None = None
    image_model_root: str = "/trt_models"
    request_timeout: float = 60.0
    docker_binary: str = "docker"

    def __post_init__(self) -> None:
        self.triton_url = self.triton_url.rstrip("/")
        if self.triton_metrics_url:
            self.triton_metrics_url = self.triton_metrics_url.rstrip("/")
        self.model_repository = Path(self.model_repository).expanduser().resolve()
        self.state_file = Path(self.state_file).expanduser().resolve()
        self.staging_root = Path(self.staging_root).expanduser().resolve()

    @classmethod
    def default(cls) -> "HotLoaderConfig":
        base_dir = Path(__file__).resolve().parent / "runtime"
        derived_triton_url, derived_metrics_url = _derived_triton_urls_from_env()
        model_repository, state_file, staging_root = _default_runtime_paths(base_dir)
        return cls(
            triton_url=derived_triton_url or _env_default(*_TRITON_URL_ENV_NAMES) or _DEFAULT_TRITON_URL,
            triton_metrics_url=derived_metrics_url or _env_default(*_TRITON_METRICS_URL_ENV_NAMES),
            model_repository=model_repository,
            state_file=state_file,
            staging_root=staging_root,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triton_url": self.triton_url,
            "triton_metrics_url": self.triton_metrics_url,
            "model_repository": str(self.model_repository),
            "state_file": str(self.state_file),
            "staging_root": str(self.staging_root),
            "image_model_root": self.image_model_root,
            "request_timeout": self.request_timeout,
            "docker_binary": self.docker_binary,
        }

    def with_updates(self, **updates: Any) -> "HotLoaderConfig":
        payload = self.to_dict()
        payload.update(updates)
        return HotLoaderConfig(**payload)


class TritonHotLoader:
    """Manage model bundles and load or unload them through Triton APIs."""

    def __init__(self, config: HotLoaderConfig | None = None) -> None:
        self.config = config or HotLoaderConfig.default()
        self._ensure_runtime_dirs()

    def _ensure_runtime_dirs(self) -> None:
        self.config.model_repository.mkdir(parents=True, exist_ok=True)
        self.config.staging_root.mkdir(parents=True, exist_ok=True)
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.config.state_file.exists():
            self._save_state(self._empty_state())

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {"aliases": {}, "updated_at": None}

    def _load_state(self) -> Dict[str, Any]:
        if not self.config.state_file.exists():
            return self._empty_state()
        try:
            with self.config.state_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise HotLoaderError(f"状态文件不是合法 JSON: {self.config.state_file}") from exc

        if not isinstance(data, dict):
            raise HotLoaderError(f"状态文件格式错误: {self.config.state_file}")

        aliases = data.get("aliases", {})
        if not isinstance(aliases, dict):
            raise HotLoaderError("状态文件 aliases 字段格式错误")

        return {
            "aliases": aliases,
            "updated_at": data.get("updated_at"),
        }

    def _save_state(self, state: Dict[str, Any]) -> None:
        temp_file = self.config.state_file.with_suffix(".tmp")
        with temp_file.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temp_file.replace(self.config.state_file)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_alias(alias: str) -> None:
        if not alias or not _ALIAS_PATTERN.match(alias):
            raise HotLoaderError(
                f"非法 alias: {alias!r}。只允许字母、数字、下划线、点和短横线。"
            )

    def _validate_config_map(self, config_map: Mapping[str, str]) -> Dict[str, str]:
        if not isinstance(config_map, Mapping) or not config_map:
            raise HotLoaderError("配置不能为空，且必须是 JSON 对象，value 为镜像地址")

        normalized: Dict[str, str] = {}
        for raw_key, image in config_map.items():
            if not isinstance(image, str) or not image.strip():
                raise HotLoaderError(f"配置项 {raw_key!r} 对应的镜像地址不能为空")
            image_ref = image.strip()
            if self._should_skip_config_entry(raw_key, image_ref):
                continue
            normalized[image_ref] = image_ref

        if not normalized:
            raise HotLoaderError("过滤 mlman_config 后没有可用模型镜像")
        return normalized

    @staticmethod
    def _should_skip_config_entry(raw_key: Any, image_ref: str) -> bool:
        key_text = raw_key if isinstance(raw_key, str) else ""
        combined = f"{key_text} {image_ref}"
        return bool(_MLMAN_CONFIG_PATTERN.search(combined))

    @staticmethod
    def _bundle_key_for_image(image_ref: str) -> str:
        digest = hashlib.sha1(image_ref.encode("utf-8")).hexdigest()[:12]
        return f"bundle_{digest}"

    @staticmethod
    def _bundle_key_for_model_image(model_name: str, image_ref: str) -> str:
        digest = hashlib.sha1(f"{model_name}\0{image_ref}".encode("utf-8")).hexdigest()[:12]
        return f"model_{model_name}_{digest}"

    def _find_state_entry_by_image(
        self,
        aliases: Mapping[str, Any],
        image_ref: str,
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        for alias, meta in aliases.items():
            if isinstance(meta, Mapping) and meta.get("image") == image_ref:
                return alias, meta
        return None, None

    def _find_state_entry_by_model(
        self,
        aliases: Mapping[str, Any],
        model_name: str,
    ) -> tuple[str | None, Mapping[str, Any] | None]:
        for alias, meta in aliases.items():
            if not isinstance(meta, Mapping):
                continue
            models = meta.get("models", [])
            if isinstance(models, list) and model_name in models:
                return alias, meta
        return None, None

    def _run_command(self, args: Sequence[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                list(args),
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise HotLoaderError(f"未找到命令: {args[0]!r}，请确认已安装并在 PATH 中") from exc

        if result.returncode != 0 and not allow_failure:
            stderr = result.stderr.strip() or result.stdout.strip() or "(无输出)"
            raise HotLoaderError(f"命令执行失败: {' '.join(args)}\n{stderr}")
        return result

    def _triton_request(
        self,
        method: str,
        path: str,
        *,
        payload: Any | None = None,
        tolerate_404: bool = False,
    ) -> httpx.Response:
        url = f"{self.config.triton_url}{path}"
        try:
            response = httpx.request(
                method,
                url,
                json=payload,
                timeout=self.config.request_timeout,
            )
        except httpx.HTTPError as exc:
            raise HotLoaderError(f"无法访问 Triton API: {url} ({exc})") from exc

        if tolerate_404 and response.status_code == 404:
            return response

        if response.is_error:
            text = response.text.strip() or "(无响应体)"
            raise HotLoaderError(
                self._format_triton_api_error(method, path, response.status_code, text)
            )
        return response

    @staticmethod
    def _format_triton_api_error(method: str, path: str, status_code: int, response_text: str) -> str:
        message = f"Triton API 调用失败: {method} {path} -> {status_code}\n{response_text}"
        lowered = response_text.lower()

        if _LOAD_UNLOAD_PATH_PATTERN.match(path) and (
            "explicit model load / unload is not allowed if polling is enabled" in lowered
            or "polling is enabled" in lowered
            or "model control mode is not explicit" in lowered
            or "load / unload" in lowered and "not allowed" in lowered
        ):
            return f"{message}\n\n{_EXPLICIT_CONTROL_HINT}"

        return message

    @staticmethod
    def _normalize_metrics_endpoint(url: str) -> str:
        candidate = url.strip()
        if not candidate:
            raise HotLoaderError("Triton metrics 地址不能为空")

        parts = urlsplit(candidate)
        path = parts.path.rstrip("/")
        if not path.endswith("/metrics"):
            path = f"{path}/metrics" if path else "/metrics"

        return urlunsplit(parts._replace(path=path, query="", fragment=""))

    @staticmethod
    def _build_netloc_with_port(parts: Any, port: int) -> str:
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"

        auth = ""
        if parts.username:
            auth = parts.username
            if parts.password:
                auth += f":{parts.password}"
            auth += "@"

        return f"{auth}{host}:{port}"

    def _candidate_metrics_endpoints(self) -> List[str]:
        candidates: List[str] = []

        if self.config.triton_metrics_url:
            candidates.append(self._normalize_metrics_endpoint(self.config.triton_metrics_url))

        candidates.append(self._normalize_metrics_endpoint(self.config.triton_url))

        parts = urlsplit(self.config.triton_url)
        if parts.port is not None:
            metrics_netloc = self._build_netloc_with_port(parts, parts.port + 2)
            candidates.append(
                urlunsplit(parts._replace(netloc=metrics_netloc, path="/metrics", query="", fragment=""))
            )

        deduplicated: List[str] = []
        for candidate in candidates:
            if candidate not in deduplicated:
                deduplicated.append(candidate)
        return deduplicated

    @staticmethod
    def _parse_prometheus_labels(raw_labels: str) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        if not raw_labels:
            return labels

        for match in _PROMETHEUS_LABEL_PATTERN.finditer(raw_labels):
            labels[match.group("key")] = bytes(match.group("value"), "utf-8").decode("unicode_escape")
        return labels

    def _parse_prometheus_gpu_metrics(self, metrics_text: str) -> Dict[str, Any]:
        gpu_metrics: Dict[str, Dict[str, Any]] = {}
        metric_names = {
            "nv_gpu_memory_total_bytes",
            "nv_gpu_memory_used_bytes",
            "nv_gpu_utilization",
            "nv_gpu_power_usage",
        }

        for raw_line in metrics_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            match = _PROMETHEUS_SAMPLE_PATTERN.match(line)
            if not match:
                continue

            metric_name = match.group("name")
            if metric_name not in metric_names:
                continue

            labels = self._parse_prometheus_labels(match.group("labels") or "")
            gpu_key = labels.get("gpu_uuid") or labels.get("gpu") or labels.get("gpu_bus_id") or "unknown"
            entry = gpu_metrics.setdefault(
                gpu_key,
                {
                    "gpu_uuid": labels.get("gpu_uuid") or gpu_key,
                    "gpu_bus_id": labels.get("gpu_bus_id"),
                    "used_bytes": None,
                    "total_bytes": None,
                    "utilization_ratio": None,
                    "power_usage_watts": None,
                },
            )

            value = float(match.group("value"))
            if metric_name == "nv_gpu_memory_total_bytes":
                entry["total_bytes"] = int(round(value))
            elif metric_name == "nv_gpu_memory_used_bytes":
                entry["used_bytes"] = int(round(value))
            elif metric_name == "nv_gpu_utilization":
                entry["utilization_ratio"] = value
            elif metric_name == "nv_gpu_power_usage":
                entry["power_usage_watts"] = value

        gpus: List[Dict[str, Any]] = []
        for index, gpu_key in enumerate(sorted(gpu_metrics)):
            entry = gpu_metrics[gpu_key]
            total_bytes = entry.get("total_bytes")
            used_bytes = entry.get("used_bytes")
            used_ratio = None
            if isinstance(total_bytes, int) and total_bytes > 0 and isinstance(used_bytes, int):
                used_ratio = used_bytes / total_bytes

            gpus.append(
                {
                    "index": index,
                    "label": f"GPU {index}",
                    "gpu_uuid": entry.get("gpu_uuid"),
                    "gpu_bus_id": entry.get("gpu_bus_id"),
                    "used_bytes": used_bytes,
                    "total_bytes": total_bytes,
                    "used_ratio": used_ratio,
                    "used_percent": used_ratio * 100 if used_ratio is not None else None,
                    "utilization_ratio": entry.get("utilization_ratio"),
                    "utilization_percent": entry.get("utilization_ratio") * 100
                    if entry.get("utilization_ratio") is not None
                    else None,
                    "power_usage_watts": entry.get("power_usage_watts"),
                }
            )

        total_used = sum(gpu.get("used_bytes") or 0 for gpu in gpus)
        total_capacity = sum(gpu.get("total_bytes") or 0 for gpu in gpus)
        total_ratio = total_used / total_capacity if total_capacity else None
        utilization_values = [
            gpu["utilization_ratio"]
            for gpu in gpus
            if isinstance(gpu.get("utilization_ratio"), (float, int))
        ]
        power_values = [
            gpu["power_usage_watts"]
            for gpu in gpus
            if isinstance(gpu.get("power_usage_watts"), (float, int))
        ]

        return {
            "summary": {
                "device_count": len(gpus),
                "used_bytes": total_used,
                "total_bytes": total_capacity,
                "used_ratio": total_ratio,
                "used_percent": total_ratio * 100 if total_ratio is not None else None,
                "average_utilization_ratio": sum(utilization_values) / len(utilization_values)
                if utilization_values
                else None,
                "average_utilization_percent": (sum(utilization_values) / len(utilization_values)) * 100
                if utilization_values
                else None,
                "total_power_usage_watts": sum(power_values) if power_values else None,
            },
            "gpus": gpus,
        }

    def get_triton_gpu_metrics(self) -> Dict[str, Any]:
        candidates = self._candidate_metrics_endpoints()
        unavailable_payload = {
            "available": False,
            "url": None,
            "candidate_urls": candidates,
            "detail": "未找到可用的 Triton metrics endpoint",
            "updated_at": self._utc_now(),
            "summary": {
                "device_count": 0,
                "used_bytes": 0,
                "total_bytes": 0,
                "used_ratio": None,
                "used_percent": None,
                "average_utilization_ratio": None,
                "average_utilization_percent": None,
                "total_power_usage_watts": None,
            },
            "gpus": [],
        }

        for candidate_url in candidates:
            try:
                response = httpx.get(candidate_url, timeout=self.config.request_timeout)
            except httpx.HTTPError as exc:
                unavailable_payload["detail"] = f"{candidate_url} 不可达: {exc}"
                continue

            if response.status_code == 404:
                unavailable_payload["detail"] = f"{candidate_url} 返回 404，当前地址未暴露 /metrics"
                continue

            if response.is_error:
                unavailable_payload["detail"] = (
                    f"{candidate_url} 返回 {response.status_code}: {response.text.strip() or '(无响应体)'}"
                )
                continue

            parsed = self._parse_prometheus_gpu_metrics(response.text)
            if not parsed["gpus"]:
                unavailable_payload["detail"] = (
                    f"{candidate_url} 可达，但未发现 GPU metrics；请确认 Triton 开启了 --allow-gpu-metrics"
                )
                continue

            return {
                "available": True,
                "url": candidate_url,
                "candidate_urls": candidates,
                "detail": f"已获取 {len(parsed['gpus'])} 张 GPU 的显存指标",
                "updated_at": self._utc_now(),
                "summary": parsed["summary"],
                "gpus": parsed["gpus"],
            }

        return unavailable_payload

    def triton_ready(self) -> Dict[str, Any]:
        try:
            response = self._triton_request("GET", "/v2/health/ready")
            return {
                "ready": response.status_code == 200,
                "detail": response.text.strip() or "OK",
            }
        except HotLoaderError as exc:
            return {"ready": False, "detail": str(exc)}

    def list_repository_models(self, *, safe: bool = False) -> List[Dict[str, Any]]:
        try:
            response = self._triton_request("POST", "/v2/repository/index", payload={})
        except HotLoaderError:
            if safe:
                return []
            raise

        try:
            payload = response.json()
        except ValueError as exc:
            raise HotLoaderError("Triton repository/index 返回了非 JSON 响应") from exc

        if not isinstance(payload, list):
            raise HotLoaderError("Triton repository/index 返回格式错误")
        return payload

    def _load_model(self, model_name: str) -> None:
        encoded = quote(model_name, safe="")
        self._triton_request("POST", f"/v2/repository/models/{encoded}/load", payload={})

    def _unload_model(self, model_name: str, *, tolerate_missing: bool = True) -> None:
        encoded = quote(model_name, safe="")
        try:
            self._triton_request(
                "POST",
                f"/v2/repository/models/{encoded}/unload",
                payload={},
                tolerate_404=tolerate_missing,
            )
        except HotLoaderError as exc:
            if tolerate_missing:
                lowered = str(exc).lower()
                if "404" in lowered or "not found" in lowered or "unknown model" in lowered:
                    return
            raise

    def _pull_image(self, image_ref: str) -> None:
        self._run_command([self.config.docker_binary, "pull", image_ref])

    def _image_model_path(self, *parts: str) -> str:
        root = self.config.image_model_root.rstrip("/")
        if not root:
            root = "/"

        cleaned_parts = [part.strip("/") for part in parts if part and part.strip("/")]
        if not cleaned_parts:
            return root
        if root == "/":
            return "/" + "/".join(cleaned_parts)
        return f"{root}/{'/'.join(cleaned_parts)}"

    def _stage_image_bundle(self, image_ref: str) -> tuple[Path, Path, List[str]]:
        operation_id = uuid.uuid4().hex[:12]
        stage_dir = self.config.staging_root / operation_id
        bundle_dir = stage_dir / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        self._pull_image(image_ref)

        create_result = self._run_command([self.config.docker_binary, "create", image_ref])
        container_id = create_result.stdout.strip()
        if not container_id:
            raise HotLoaderError(f"无法为镜像创建临时容器: {image_ref}")

        try:
            self._run_command(
                [
                    self.config.docker_binary,
                    "cp",
                    f"{container_id}:{self._image_model_path()}/.",
                    str(bundle_dir),
                ]
            )
        finally:
            self._run_command(
                [self.config.docker_binary, "rm", "-f", container_id],
                allow_failure=True,
            )

        models = self._discover_models(bundle_dir)
        if not models:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise HotLoaderError(
                f"镜像 {image_ref} 中未发现模型目录，请确认 {self.config.image_model_root} 下是 Triton model repository 结构"
            )
        return stage_dir, bundle_dir, models

    def _stage_image_model(self, image_ref: str, model_name: str) -> tuple[Path, Path]:
        if not model_name or not model_name.strip():
            raise HotLoaderError("model_name 不能为空")

        normalized_model_name = model_name.strip()
        operation_id = uuid.uuid4().hex[:12]
        stage_dir = self.config.staging_root / operation_id
        bundle_dir = stage_dir / "bundle"
        model_stage_dir = bundle_dir / normalized_model_name
        model_stage_dir.mkdir(parents=True, exist_ok=True)

        self._pull_image(image_ref)

        create_result = self._run_command([self.config.docker_binary, "create", image_ref])
        container_id = create_result.stdout.strip()
        if not container_id:
            raise HotLoaderError(f"无法为镜像创建临时容器: {image_ref}")

        source_path = self._image_model_path(normalized_model_name)
        try:
            try:
                self._run_command(
                    [
                        self.config.docker_binary,
                        "cp",
                        f"{container_id}:{source_path}/.",
                        str(model_stage_dir),
                    ]
                )
            except HotLoaderError as exc:
                raise HotLoaderError(
                    f"镜像 {image_ref} 中未发现模型目录 {source_path}，请确认镜像结构与当前项目约定一致"
                ) from exc
        finally:
            self._run_command(
                [self.config.docker_binary, "rm", "-f", container_id],
                allow_failure=True,
            )

        if not model_stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise HotLoaderError(
                f"镜像 {image_ref} 中未发现模型目录 {source_path}，请确认镜像结构与当前项目约定一致"
            )
        return stage_dir, bundle_dir

    @staticmethod
    def _discover_models(bundle_dir: Path) -> List[str]:
        if not bundle_dir.exists():
            return []
        models = [path.name for path in bundle_dir.iterdir() if path.is_dir() and not path.name.startswith(".")]
        return sorted(models)

    @staticmethod
    def _sort_versions(versions: Iterable[str]) -> List[str]:
        return sorted({version for version in versions}, key=int)

    def _discover_model_versions(self, model_dir: Path) -> List[str]:
        if not model_dir.exists() or not model_dir.is_dir():
            raise HotLoaderError(f"模型目录不存在: {model_dir}")

        versions = self._sort_versions(
            path.name
            for path in model_dir.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and _TRITON_VERSION_DIR_PATTERN.match(path.name)
        )
        if not versions:
            raise HotLoaderError(
                f"模型目录 {model_dir} 中未发现合法 Triton version 子目录（应为纯数字目录名）"
            )
        return versions

    @staticmethod
    def _select_active_version(versions: Sequence[str]) -> str:
        if not versions:
            raise HotLoaderError("无法从空版本列表中选择激活版本")
        return sorted(versions, key=int)[-1]

    def _collect_model_version_metadata(
        self,
        root_dir: Path,
        model_names: Iterable[str],
    ) -> tuple[Dict[str, List[str]], Dict[str, str]]:
        model_versions: Dict[str, List[str]] = {}
        active_versions: Dict[str, str] = {}
        for model_name in sorted(set(model_names)):
            versions = self._discover_model_versions(root_dir / model_name)
            model_versions[model_name] = versions
            active_versions[model_name] = self._select_active_version(versions)
        return model_versions, active_versions

    @staticmethod
    def _parse_model_version_ref(version_ref: str) -> tuple[str, str]:
        candidate = version_ref.strip()
        match = _MODEL_VERSION_REF_PATTERN.match(candidate)
        if not match:
            raise HotLoaderError(
                f"非法 model version 引用: {version_ref!r}，格式应为 model_name@123"
            )
        return match.group("model"), match.group("version")

    @staticmethod
    def _remove_pbtxt_block(config_text: str, field_name: str) -> str:
        field_pattern = re.compile(rf"(?m)^[ \t]*{re.escape(field_name)}\s*:")
        updated_text = config_text

        while True:
            match = field_pattern.search(updated_text)
            if not match:
                return updated_text

            brace_start = updated_text.find("{", match.end())
            if brace_start == -1:
                return updated_text

            depth = 0
            block_end: int | None = None
            for index in range(brace_start, len(updated_text)):
                char = updated_text[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        block_end = index + 1
                        break

            if block_end is None:
                return updated_text

            line_start = updated_text.rfind("\n", 0, match.start()) + 1
            before = updated_text[:line_start].rstrip()
            after = updated_text[block_end:].lstrip()

            if before and after:
                updated_text = f"{before}\n\n{after}"
            else:
                updated_text = before or after

    def _write_active_version_policy(self, model_dir: Path, active_version: str) -> bool:
        config_path = model_dir / "config.pbtxt"
        if not config_path.exists():
            return False

        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HotLoaderError(f"读取模型配置失败: {config_path} ({exc})") from exc

        cleaned_text = self._remove_pbtxt_block(config_text, "version_policy").strip()
        version_block = "\n".join(
            [
                "version_policy: {",
                "  specific {",
                f"    versions: [ {int(active_version)} ]",
                "  }",
                "}",
            ]
        )
        updated_text = f"{cleaned_text}\n\n{version_block}\n" if cleaned_text else f"{version_block}\n"

        try:
            config_path.write_text(updated_text, encoding="utf-8")
        except OSError as exc:
            raise HotLoaderError(f"写入模型配置失败: {config_path} ({exc})") from exc
        return True

    @staticmethod
    def _extract_specific_version_policy(config_text: str) -> List[str]:
        match = re.search(
            r"version_policy\s*:\s*\{[\s\S]*?specific\s*\{[\s\S]*?versions\s*:\s*\[([^\]]+)\]",
            config_text,
        )
        if not match:
            return []
        versions = [token.strip() for token in match.group(1).split(",")]
        return sorted({version for version in versions if version.isdigit()}, key=int)

    def _read_specific_version_policy(self, model_dir: Path) -> List[str]:
        config_path = model_dir / "config.pbtxt"
        if not config_path.exists():
            return []
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError:
            return []
        return self._extract_specific_version_policy(config_text)

    @staticmethod
    def _copy_model_directory_excluding_versions(
        source_dir: Path,
        target_dir: Path,
        removed_versions: Iterable[str],
    ) -> None:
        excluded_names = set(removed_versions)
        source_dir_str = str(source_dir)

        def ignore(dir_path: str, names: List[str]) -> List[str]:
            if dir_path == source_dir_str:
                return [name for name in names if name in excluded_names]
            return []

        shutil.copytree(source_dir, target_dir, ignore=ignore)

    def _build_model_owner_index(self, state: Dict[str, Any], *, exclude_alias: str | None = None) -> Dict[str, str]:
        owner_index: Dict[str, str] = {}
        for alias, meta in state.get("aliases", {}).items():
            if alias == exclude_alias:
                continue
            for model_name in meta.get("models", []):
                owner_index[model_name] = alias
        return owner_index

    def _prepare_target_directory(
        self,
        source_dir: Path,
        target_dir: Path,
        backup_root: Path,
    ) -> Path | None:
        backup_dir: Path | None = None
        if target_dir.exists():
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_dir = backup_root / target_dir.name
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            target_dir.rename(backup_dir)

        try:
            shutil.copytree(source_dir, target_dir)
        except Exception as exc:
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            if backup_dir and backup_dir.exists():
                backup_dir.rename(target_dir)
            raise HotLoaderError(f"拷贝模型目录失败: {source_dir} -> {target_dir} ({exc})") from exc
        return backup_dir

    @staticmethod
    def _restore_backup(backup_dir: Path | None, target_dir: Path) -> None:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        if backup_dir and backup_dir.exists():
            backup_dir.rename(target_dir)

    @staticmethod
    def _cleanup_backup(backup_dir: Path | None) -> None:
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

    def _delete_model_directory(self, model_name: str) -> None:
        target_dir = self.config.model_repository / model_name
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

    def _hydrate_state_aliases(self, state: Dict[str, Any]) -> Dict[str, Any]:
        aliases = state.setdefault("aliases", {})
        for alias, meta in list(aliases.items()):
            if isinstance(meta, Mapping):
                aliases[alias] = self._hydrate_alias_metadata(meta)
        return state

    def _resolve_active_version(
        self,
        state: Mapping[str, Any],
        model_name: str,
        available_versions: Sequence[str],
        *,
        model_dir: Path,
    ) -> str:
        for meta in state.get("aliases", {}).values():
            if not isinstance(meta, Mapping):
                continue
            active_versions = meta.get("active_versions", {})
            if isinstance(active_versions, Mapping):
                active_version = active_versions.get(model_name)
                if isinstance(active_version, str) and active_version in available_versions:
                    return active_version

        configured_versions = self._read_specific_version_policy(model_dir)
        matching_versions = [version for version in configured_versions if version in available_versions]
        if matching_versions:
            return self._select_active_version(matching_versions)

        return self._select_active_version(available_versions)

    def _update_aliases_for_model_version_change(
        self,
        aliases: Dict[str, Any],
        model_name: str,
        remaining_versions: Sequence[str],
        active_version: str | None,
    ) -> tuple[List[str], List[str]]:
        touched_aliases: List[str] = []
        removed_aliases: List[str] = []

        for alias, meta in list(aliases.items()):
            if not isinstance(meta, Mapping):
                continue
            hydrated = self._hydrate_alias_metadata(meta)
            models = [name for name in hydrated.get("models", []) if isinstance(name, str)]
            if model_name not in models:
                continue

            touched_aliases.append(alias)
            model_versions = dict(hydrated.get("model_versions", {}))
            active_versions = dict(hydrated.get("active_versions", {}))

            if remaining_versions:
                model_versions[model_name] = list(remaining_versions)
                if active_version is not None:
                    active_versions[model_name] = active_version
                hydrated["model_versions"] = model_versions
                hydrated["active_versions"] = active_versions
                hydrated["updated_at"] = self._utc_now()
                aliases[alias] = hydrated
                continue

            hydrated["models"] = [name for name in models if name != model_name]
            model_versions.pop(model_name, None)
            active_versions.pop(model_name, None)
            if model_versions:
                hydrated["model_versions"] = model_versions
            else:
                hydrated.pop("model_versions", None)
            if active_versions:
                hydrated["active_versions"] = active_versions
            else:
                hydrated.pop("active_versions", None)
            hydrated["updated_at"] = self._utc_now()

            if hydrated["models"]:
                aliases[alias] = hydrated
            else:
                removed_aliases.append(alias)
                del aliases[alias]

        return sorted(set(touched_aliases)), sorted(set(removed_aliases))

    def _hydrate_alias_metadata(self, meta: Mapping[str, Any]) -> Dict[str, Any]:
        hydrated = dict(meta)
        models = [model_name for model_name in hydrated.get("models", []) if isinstance(model_name, str)]
        model_versions = {
            model_name: versions
            for model_name, versions in hydrated.get("model_versions", {}).items()
            if isinstance(model_name, str) and isinstance(versions, list)
        } if isinstance(hydrated.get("model_versions"), dict) else {}
        active_versions = {
            model_name: version
            for model_name, version in hydrated.get("active_versions", {}).items()
            if isinstance(model_name, str) and isinstance(version, str)
        } if isinstance(hydrated.get("active_versions"), dict) else {}

        missing_models = [
            model_name
            for model_name in models
            if model_name not in model_versions or model_name not in active_versions
        ]
        for model_name in missing_models:
            model_dir = self.config.model_repository / model_name
            if not model_dir.exists():
                continue
            try:
                versions = self._discover_model_versions(model_dir)
            except HotLoaderError:
                continue
            model_versions.setdefault(model_name, versions)
            active_versions.setdefault(model_name, self._select_active_version(versions))

        if model_versions:
            hydrated["model_versions"] = model_versions
        if active_versions:
            hydrated["active_versions"] = active_versions
        return hydrated

    def get_managed_state(self) -> Dict[str, Any]:
        state = self._hydrate_state_aliases(self._load_state())
        aliases = {
            alias: self._hydrate_alias_metadata(meta)
            for alias, meta in state.get("aliases", {}).items()
            if isinstance(meta, Mapping)
        }
        managed_images = sorted(
            [
                {
                    "id": alias,
                    "image": meta.get("image"),
                    "models": meta.get("models", []),
                    "model_versions": meta.get("model_versions", {}),
                    "active_versions": meta.get("active_versions", {}),
                    "updated_at": meta.get("updated_at"),
                }
                for alias, meta in aliases.items()
            ],
            key=lambda item: item.get("image") or "",
        )
        managed_models = sorted(
            {
                model_name
                for meta in aliases.values()
                for model_name in meta.get("models", [])
            }
        )
        managed_model_versions: Dict[str, List[str]] = {}
        managed_active_versions: Dict[str, str] = {}
        for meta in aliases.values():
            if isinstance(meta.get("model_versions"), dict):
                for model_name, versions in meta["model_versions"].items():
                    if isinstance(versions, list):
                        managed_model_versions[model_name] = versions
            if isinstance(meta.get("active_versions"), dict):
                for model_name, version in meta["active_versions"].items():
                    if isinstance(version, str):
                        managed_active_versions[model_name] = version
        return {
            "config": self.config.to_dict(),
            "updated_at": state.get("updated_at"),
            "aliases": aliases,
            "managed_images": managed_images,
            "managed_alias_count": len(aliases),
            "managed_image_count": len(managed_images),
            "managed_model_count": len(managed_models),
            "managed_models": managed_models,
            "managed_model_versions": managed_model_versions,
            "managed_active_versions": managed_active_versions,
        }

    def get_status(self) -> Dict[str, Any]:
        state = self.get_managed_state()
        ready = self.triton_ready()
        triton_models = self.list_repository_models(safe=True)
        gpu_metrics = self.get_triton_gpu_metrics()
        return {
            "triton": {
                "url": self.config.triton_url,
                "ready": ready["ready"],
                "detail": ready["detail"],
                "metrics": gpu_metrics,
                "repository_models": triton_models,
            },
            "manager": state,
        }

    def apply_config(
        self,
        config_map: Mapping[str, str],
        *,
        prune_missing: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        desired = self._validate_config_map(config_map)
        state = self._hydrate_state_aliases(self._load_state())
        aliases = state.setdefault("aliases", {})
        desired_images = sorted(desired.values())

        result: Dict[str, Any] = {
            "success": True,
            "requested_images": desired_images,
            "prune_missing": prune_missing,
            "force": force,
            "applied": [],
            "skipped": [],
            "removed": [],
            "errors": [],
        }

        if prune_missing:
            missing_aliases = sorted(
                alias
                for alias, meta in aliases.items()
                if isinstance(meta, Mapping) and meta.get("image") not in desired_images
            )
            for alias in missing_aliases:
                try:
                    removal = self.unload_alias(alias)
                    result["removed"].append(removal)
                except HotLoaderError as exc:
                    result["success"] = False
                    result["errors"].append({"alias": alias, "error": str(exc)})

        for image_ref in desired_images:
            alias, current = self._find_state_entry_by_image(aliases, image_ref)
            if current and current.get("image") == image_ref and not force:
                current_meta = self._hydrate_alias_metadata(current)
                result["skipped"].append(
                    {
                        "image": image_ref,
                        "bundle_id": alias,
                        "models": current_meta.get("models", []),
                        "model_versions": current_meta.get("model_versions", {}),
                        "active_versions": current_meta.get("active_versions", {}),
                        "reason": "image unchanged",
                    }
                )
                continue

            try:
                target_alias = alias or self._bundle_key_for_image(image_ref)
                applied = self._apply_alias(target_alias, image_ref)
                result["applied"].append(applied)
            except HotLoaderError as exc:
                result["success"] = False
                result["errors"].append({"bundle_id": alias, "image": image_ref, "error": str(exc)})

        result["state"] = self.get_managed_state()
        return result

    def load_model_from_image(
        self,
        model_name: str,
        image_ref: str,
        *,
        overwrite: bool = True,
        load_after_copy: bool = True,
    ) -> Dict[str, Any]:
        normalized_model_name = model_name.strip()
        normalized_image_ref = image_ref.strip()
        if not normalized_model_name:
            raise HotLoaderError("model_name 不能为空")
        if not normalized_image_ref:
            raise HotLoaderError("image 不能为空")

        state = self._hydrate_state_aliases(self._load_state())
        aliases = state.setdefault("aliases", {})
        current_alias, current_meta = self._find_state_entry_by_model(aliases, normalized_model_name)
        current_image = current_meta.get("image") if isinstance(current_meta, Mapping) else None
        target_dir = self.config.model_repository / normalized_model_name

        if (
            current_image == normalized_image_ref
            and target_dir.exists()
            and isinstance(current_meta, Mapping)
        ):
            hydrated_current = self._hydrate_alias_metadata(current_meta)
            active_versions = hydrated_current.get("active_versions", {})
            active_version = active_versions.get(normalized_model_name) if isinstance(active_versions, dict) else None
            return {
                "success": True,
                "skipped": True,
                "alias": current_alias,
                "image": normalized_image_ref,
                "model_name": normalized_model_name,
                "source_path": self._image_model_path(normalized_model_name),
                "target_path": str(target_dir),
                "model_versions": hydrated_current.get("model_versions", {}).get(normalized_model_name, []),
                "active_version": active_version,
                "load_after_copy": load_after_copy,
                "reason": "model image unchanged",
                "updated_at": hydrated_current.get("updated_at"),
            }

        if target_dir.exists() and not overwrite:
            raise HotLoaderError(f"模型目录已存在，且 overwrite=false: {target_dir}")

        if current_alias is not None:
            if not overwrite:
                raise HotLoaderError(
                    f"模型 {normalized_model_name} 当前已由 {current_alias} 管理，且 overwrite=false"
                )
            self.unload_models([normalized_model_name])

        stage_dir: Path | None = None
        try:
            stage_dir, bundle_dir = self._stage_image_model(normalized_image_ref, normalized_model_name)
            model_versions, active_versions = self._collect_model_version_metadata(
                bundle_dir,
                [normalized_model_name],
            )

            backup_root = stage_dir / "_backup"
            backup_dir = self._prepare_target_directory(
                bundle_dir / normalized_model_name,
                target_dir,
                backup_root,
            )
            try:
                wrote_version_policy = self._write_active_version_policy(
                    target_dir,
                    active_versions[normalized_model_name],
                )
                if load_after_copy:
                    self._load_model(normalized_model_name)
            except Exception as exc:
                self._restore_backup(backup_dir, target_dir)
                raise HotLoaderError(
                    f"复制模型 {normalized_model_name} 后处理失败，已尝试回滚目录: {exc}"
                ) from exc

            updated_state = self._hydrate_state_aliases(self._load_state())
            updated_aliases = updated_state.setdefault("aliases", {})
            alias = self._bundle_key_for_model_image(normalized_model_name, normalized_image_ref)
            updated_aliases[alias] = {
                "image": normalized_image_ref,
                "models": [normalized_model_name],
                "model_versions": model_versions,
                "active_versions": active_versions,
                "updated_at": self._utc_now(),
            }
            updated_state["updated_at"] = self._utc_now()
            self._save_state(updated_state)
            self._cleanup_backup(backup_dir)

            return {
                "success": True,
                "skipped": False,
                "alias": alias,
                "image": normalized_image_ref,
                "model_name": normalized_model_name,
                "source_path": self._image_model_path(normalized_model_name),
                "target_path": str(target_dir),
                "model_versions": model_versions[normalized_model_name],
                "active_version": active_versions[normalized_model_name],
                "load_after_copy": load_after_copy,
                "loaded": load_after_copy,
                "wrote_version_policy": wrote_version_policy,
                "updated_at": updated_aliases[alias]["updated_at"],
            }
        finally:
            if stage_dir and stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)

    def _apply_alias(self, alias: str, image_ref: str) -> Dict[str, Any]:
        self._validate_alias(alias)
        state = self._hydrate_state_aliases(self._load_state())
        aliases = state.setdefault("aliases", {})
        current = aliases.get(alias, {})
        old_models = set(current.get("models", []))

        stage_dir: Path | None = None
        try:
            stage_dir, bundle_dir, new_models = self._stage_image_bundle(image_ref)
            owner_index = self._build_model_owner_index(state, exclude_alias=alias)
            conflicts = {name: owner_index[name] for name in new_models if name in owner_index}
            model_versions, active_versions = self._collect_model_version_metadata(bundle_dir, new_models)
            if conflicts:
                conflict_text = ", ".join(
                    f"{model} -> {owner}" for model, owner in sorted(conflicts.items())
                )
                raise HotLoaderError(
                    f"alias {alias} 试图接管已被其他 alias 管理的模型: {conflict_text}"
                )

            backup_root = stage_dir / "_backup"
            deployed_models: List[str] = []
            backups: Dict[str, Path | None] = {}
            version_policy_models: List[str] = []

            for model_name in new_models:
                source_dir = bundle_dir / model_name
                target_dir = self.config.model_repository / model_name
                backup_dir = self._prepare_target_directory(source_dir, target_dir, backup_root)
                if self._write_active_version_policy(target_dir, active_versions[model_name]):
                    version_policy_models.append(model_name)
                try:
                    self._load_model(model_name)
                except HotLoaderError as exc:
                    self._restore_backup(backup_dir, target_dir)
                    try:
                        if backup_dir and target_dir.exists():
                            self._load_model(model_name)
                    except HotLoaderError:
                        pass
                    raise HotLoaderError(
                        f"按版本重载模型 {model_name} 失败，已尝试回滚目录: {exc}"
                    ) from exc

                backups[model_name] = backup_dir
                deployed_models.append(model_name)

            removed_models = sorted(old_models - set(new_models))
            for model_name in removed_models:
                self._unload_model(model_name, tolerate_missing=True)
                self._delete_model_directory(model_name)

            aliases[alias] = {
                "image": image_ref,
                "models": sorted(new_models),
                "model_versions": model_versions,
                "active_versions": active_versions,
                "updated_at": self._utc_now(),
            }
            state["updated_at"] = self._utc_now()
            self._save_state(state)

            for backup_dir in backups.values():
                self._cleanup_backup(backup_dir)

            return {
                "alias": alias,
                "image": image_ref,
                "models": sorted(new_models),
                "model_versions": model_versions,
                "active_versions": active_versions,
                "reloaded_models": sorted(deployed_models),
                "removed_models": removed_models,
                "version_policy_models": sorted(version_policy_models),
                "version_load_strategy": "load_api_reload_with_specific_version_policy",
                "updated_at": aliases[alias]["updated_at"],
            }
        finally:
            if stage_dir and stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)

    def unload_alias(self, alias: str) -> Dict[str, Any]:
        self._validate_alias(alias)
        state = self._hydrate_state_aliases(self._load_state())
        aliases = state.setdefault("aliases", {})
        current = aliases.get(alias)
        if not current:
            raise HotLoaderError(f"alias 不存在: {alias}")

        models = sorted(set(current.get("models", [])))
        for model_name in models:
            self._unload_model(model_name, tolerate_missing=True)
            self._delete_model_directory(model_name)

        del aliases[alias]
        state["updated_at"] = self._utc_now()
        self._save_state(state)

        return {
            "alias": alias,
            "image": current.get("image"),
            "models": models,
            "updated_at": state["updated_at"],
        }

    def unload_aliases(self, aliases: Iterable[str]) -> Dict[str, Any]:
        removed = []
        errors = []
        for alias in sorted(set(aliases)):
            try:
                removed.append(self.unload_alias(alias))
            except HotLoaderError as exc:
                errors.append({"alias": alias, "error": str(exc)})
        return {
            "success": not errors,
            "removed": removed,
            "errors": errors,
            "state": self.get_managed_state(),
        }

    def unload_model_versions(self, version_refs: Iterable[str]) -> Dict[str, Any]:
        requests: Dict[str, List[str]] = {}
        for version_ref in sorted(set(version_refs)):
            model_name, version = self._parse_model_version_ref(version_ref)
            requests.setdefault(model_name, []).append(version)

        if not requests:
            raise HotLoaderError("请至少提供一个 model@version")

        state = self._hydrate_state_aliases(self._load_state())
        aliases = state.setdefault("aliases", {})
        operation_root = self.config.staging_root / f"version_unload_{uuid.uuid4().hex[:12]}"
        backup_root = operation_root / "backup"

        removed_entries: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        removed_models: List[str] = []
        reloaded_models: List[str] = []
        affected_aliases: List[str] = []
        deleted_aliases: List[str] = []
        switched_active_versions: List[Dict[str, str]] = []

        try:
            for model_name, versions_to_remove in sorted(requests.items()):
                model_dir = self.config.model_repository / model_name
                if not model_dir.exists():
                    errors.append(
                        {
                            "model": model_name,
                            "versions": versions_to_remove,
                            "error": f"模型目录不存在: {model_dir}",
                        }
                    )
                    continue

                try:
                    existing_versions = self._discover_model_versions(model_dir)
                except HotLoaderError as exc:
                    errors.append(
                        {
                            "model": model_name,
                            "versions": versions_to_remove,
                            "error": str(exc),
                        }
                    )
                    continue

                missing_versions = [
                    version for version in versions_to_remove if version not in existing_versions
                ]
                if missing_versions:
                    errors.append(
                        {
                            "model": model_name,
                            "versions": versions_to_remove,
                            "error": f"模型 {model_name} 不存在版本: {', '.join(missing_versions)}",
                        }
                    )
                    continue

                current_active_version = self._resolve_active_version(
                    state,
                    model_name,
                    existing_versions,
                    model_dir=model_dir,
                )
                remaining_versions = [
                    version for version in existing_versions if version not in versions_to_remove
                ]
                backup_dir = backup_root / model_name
                backup_dir.parent.mkdir(parents=True, exist_ok=True)
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                model_dir.rename(backup_dir)

                next_active_version: str | None = None
                wrote_version_policy = False
                try:
                    if remaining_versions:
                        self._copy_model_directory_excluding_versions(
                            backup_dir,
                            model_dir,
                            versions_to_remove,
                        )
                        if (model_dir / "config.pbtxt").exists() and current_active_version in remaining_versions:
                            next_active_version = current_active_version
                        else:
                            next_active_version = self._select_active_version(remaining_versions)
                        wrote_version_policy = self._write_active_version_policy(
                            model_dir,
                            next_active_version,
                        )
                        self._load_model(model_name)
                    else:
                        self._unload_model(model_name, tolerate_missing=True)

                    touched_aliases, removed_alias_list = self._update_aliases_for_model_version_change(
                        aliases,
                        model_name,
                        remaining_versions,
                        next_active_version,
                    )
                    affected_aliases.extend(touched_aliases)
                    deleted_aliases.extend(removed_alias_list)
                    state["updated_at"] = self._utc_now()
                    self._save_state(state)

                    if remaining_versions:
                        reloaded_models.append(model_name)
                        if current_active_version != next_active_version and next_active_version is not None:
                            switched_active_versions.append(
                                {
                                    "model": model_name,
                                    "from": current_active_version,
                                    "to": next_active_version,
                                }
                            )
                    else:
                        removed_models.append(model_name)

                    removed_entries.append(
                        {
                            "model": model_name,
                            "removed_versions": list(versions_to_remove),
                            "remaining_versions": list(remaining_versions),
                            "active_version": next_active_version,
                            "unloaded_model": not remaining_versions,
                            "wrote_version_policy": wrote_version_policy,
                        }
                    )
                    self._cleanup_backup(backup_dir)
                except Exception as exc:
                    self._restore_backup(backup_dir, model_dir)
                    try:
                        if model_dir.exists():
                            if current_active_version in existing_versions:
                                self._write_active_version_policy(model_dir, current_active_version)
                            self._load_model(model_name)
                    except HotLoaderError:
                        pass

                    error_message = str(exc)
                    errors.append(
                        {
                            "model": model_name,
                            "versions": versions_to_remove,
                            "error": error_message,
                        }
                    )

            return {
                "success": not errors,
                "removed_versions": removed_entries,
                "removed_models": sorted(set(removed_models)),
                "reloaded_models": sorted(set(reloaded_models)),
                "affected_aliases": sorted(set(affected_aliases)),
                "deleted_aliases": sorted(set(deleted_aliases)),
                "switched_active_versions": switched_active_versions,
                "errors": errors,
                "state": self.get_managed_state(),
            }
        finally:
            if operation_root.exists():
                shutil.rmtree(operation_root, ignore_errors=True)

    def unload_models(self, model_names: Iterable[str]) -> Dict[str, Any]:
        unique_models = sorted({model_name for model_name in model_names if model_name})
        if not unique_models:
            raise HotLoaderError("请至少提供一个 model name")

        state = self._hydrate_state_aliases(self._load_state())
        aliases = state.setdefault("aliases", {})
        affected_aliases = []

        for model_name in unique_models:
            self._unload_model(model_name, tolerate_missing=True)
            self._delete_model_directory(model_name)

        for alias, meta in list(aliases.items()):
            models = [model_name for model_name in meta.get("models", []) if model_name not in unique_models]
            if len(models) != len(meta.get("models", [])):
                affected_aliases.append(alias)
                if models:
                    meta["models"] = models
                    if isinstance(meta.get("model_versions"), dict):
                        meta["model_versions"] = {
                            model_name: versions
                            for model_name, versions in meta["model_versions"].items()
                            if model_name in models
                        }
                    if isinstance(meta.get("active_versions"), dict):
                        meta["active_versions"] = {
                            model_name: version
                            for model_name, version in meta["active_versions"].items()
                            if model_name in models
                        }
                    meta["updated_at"] = self._utc_now()
                else:
                    del aliases[alias]

        state["updated_at"] = self._utc_now()
        self._save_state(state)

        return {
            "success": True,
            "removed_models": unique_models,
            "affected_aliases": sorted(affected_aliases),
            "state": self.get_managed_state(),
        }

    def reload_models(self, model_names: Iterable[str]) -> Dict[str, Any]:
        unique_models = sorted({model_name for model_name in model_names if model_name})
        if not unique_models:
            raise HotLoaderError("请至少提供一个 model name")

        reloaded = []
        for model_name in unique_models:
            self._load_model(model_name)
            reloaded.append(model_name)

        return {
            "success": True,
            "reloaded_models": reloaded,
            "state": self.get_managed_state(),
        }

    def load_config_file(self, config_file: str | Path) -> Dict[str, str]:
        path = Path(config_file).expanduser().resolve()
        if not path.exists():
            raise HotLoaderError(f"配置文件不存在: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise HotLoaderError(f"配置文件不是合法 JSON: {path}") from exc
        return self._validate_config_map(payload)

    def sample_config(self) -> Dict[str, str]:
        sample_path = Path(__file__).resolve().parent / "sample_config.json"
        if sample_path.exists():
            with sample_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, Mapping):
                raise HotLoaderError(f"示例配置格式错误: {sample_path}")

            sanitized: Dict[str, str] = {}
            for raw_key, image in payload.items():
                if not isinstance(image, str) or not image.strip():
                    continue
                image_ref = image.strip()
                if self._should_skip_config_entry(raw_key, image_ref):
                    continue
                key = raw_key if isinstance(raw_key, str) and raw_key.strip() else self._bundle_key_for_image(image_ref)
                sanitized[key] = image_ref

            if not sanitized:
                raise HotLoaderError("示例配置中没有可用模型镜像")
            return sanitized
        return {}
