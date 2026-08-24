from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import quote, urlsplit, urlunsplit

import httpx


class HotLoaderError(RuntimeError):
    """Raised when a hot-loading operation fails."""


class HotLoaderConflictError(HotLoaderError):
    """Raised when a model already has a different active operation."""


_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_TRITON_VERSION_DIR_PATTERN = re.compile(r"^\d+$")
_MODEL_VERSION_REF_PATTERN = re.compile(r"^(?P<model>[^@\s][^@]*)@(?P<version>\d+)$")
_LOAD_UNLOAD_PATH_PATTERN = re.compile(r"^/v2/repository/models/.+/(load|unload)$")
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
_MODEL_SOURCE_PATH_ENV_NAMES = ("MODEL_SOURCE_PATH",)
_MODEL_TARGET_PATH_ENV_NAMES = ("MODEL_TARGET_PATH",)
_TRITON_REPOSITORY_PVC_ENV_NAMES = ("TRITON_REPOSITORY_PVC",)
_K8S_NAMESPACE_ENV_NAMES = ("K8S_NAMESPACE",)
_MODEL_IMAGE_REGISTRY_PREFIX_ENV_NAMES = ("MODEL_IMAGE_REGISTRY_PREFIX",)
_JOB_TTL_SECONDS_ENV_NAMES = ("JOB_TTL_SECONDS_AFTER_FINISHED",)
_JOB_BACKOFF_LIMIT_ENV_NAMES = ("JOB_BACKOFF_LIMIT",)
_MODEL_COPY_CPU_REQUEST_ENV_NAMES = ("MODEL_COPY_CPU_REQUEST",)
_MODEL_COPY_MEMORY_REQUEST_ENV_NAMES = ("MODEL_COPY_MEMORY_REQUEST",)
_MODEL_COPY_CPU_LIMIT_ENV_NAMES = ("MODEL_COPY_CPU_LIMIT",)
_MODEL_COPY_MEMORY_LIMIT_ENV_NAMES = ("MODEL_COPY_MEMORY_LIMIT",)
_MAX_CONCURRENT_JOBS_ENV_NAMES = ("MAX_CONCURRENT_JOBS",)
_TRITON_RELOAD_MAX_ATTEMPTS_ENV_NAMES = ("TRITON_RELOAD_MAX_ATTEMPTS",)
_TRITON_RELOAD_RETRY_BASE_SECONDS_ENV_NAMES = ("TRITON_RELOAD_RETRY_BASE_SECONDS",)
_TRITON_RELOAD_RETRY_MAX_SECONDS_ENV_NAMES = ("TRITON_RELOAD_RETRY_MAX_SECONDS",)
_TRITON_RELOAD_TIMEOUT_SECONDS_ENV_NAMES = ("TRITON_RELOAD_TIMEOUT_SECONDS",)
_JOB_IMAGE_PULL_POLICY_ENV_NAMES = ("MODEL_COPY_IMAGE_PULL_POLICY",)
_JOB_TOLERATIONS_ENV_NAMES = ("JOB_TOLERATIONS_JSON",)
_REPOSITORY_MAINTENANCE_IMAGE_ENV_NAMES = ("REPOSITORY_MAINTENANCE_IMAGE",)
_CONTROLLER_LABEL = "triton-hot-loader"
_MODEL_NAME_REQUEST_PATTERN = re.compile(r"^[a-z0-9_-]+$")
_IMAGE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]*$")
_K8S_NAME_SANITIZE_PATTERN = re.compile(r"[^a-z0-9-]+")
_IMAGE_TAG_RELEASE_SUFFIX_PATTERN = re.compile(r"(?:[-_])\d{8}(?:[-_]\d{6})?$")
_MODEL_NAME_DERIVE_SANITIZE_PATTERN = re.compile(r"[^a-z0-9_]+")
_REPOSITORY_JOB_TERMINAL_POLL_INTERVAL_SECONDS = 2.0
_REPOSITORY_JOB_TIMEOUT_SECONDS = 300.0
_TRITON_RELOAD_DEFAULT_MAX_ATTEMPTS = 8
_TRITON_RELOAD_DEFAULT_RETRY_BASE_SECONDS = 2.0
_TRITON_RELOAD_DEFAULT_RETRY_MAX_SECONDS = 60.0
_TRITON_RELOAD_DEFAULT_TIMEOUT_SECONDS = 600.0

_EXPLICIT_CONTROL_HINT = (
    "当前 Triton 不允许通过 API 显式执行 load/unload。\n"
    "请确认以下三点：\n"
    "1. Triton 使用 --model-control-mode=EXPLICIT 启动；\n"
    "2. 不要开启 repository polling；如果设置了 --repository-poll-secs，请删除该参数或显式设为 0；\n"
    "3. 修改启动参数后需要重启 Triton。\n"
    "推荐启动方式：\n"
    "tritonserver --model-repository=/repository/trt_models --model-control-mode=EXPLICIT --repository-poll-secs=0"
)

_SYNC_LOAD_SUCCESS_STATUSES = {"MODEL_READY"}
_SYNC_LOAD_FAILURE_STATUSES = {"COPY_FAILED", "TRITON_RELOAD_FAILED"}
_SYNC_LOAD_TERMINAL_STATUSES = _SYNC_LOAD_SUCCESS_STATUSES | _SYNC_LOAD_FAILURE_STATUSES
_SYNC_LOAD_DEFAULT_TIMEOUT_SECONDS = 600.0
_SYNC_LOAD_DEFAULT_POLL_INTERVAL_SECONDS = 2.0
_SYNC_UNLOAD_DEFAULT_TIMEOUT_SECONDS = 120.0
_SYNC_UNLOAD_DEFAULT_POLL_INTERVAL_SECONDS = 1.0
_REPOSITORY_SYNC_VISIBILITY_TIMEOUT_SECONDS = 15.0
_REPOSITORY_SYNC_VISIBILITY_POLL_INTERVAL_SECONDS = 0.5
_ACTIVE_JOB_STATUSES = {
    "JOB_CREATED",
    "SCHEDULING",
    "IMAGE_PULLING",
    "COPY_RUNNING",
    "COPY_SUCCEEDED",
    "TRITON_RELOAD_RUNNING",
}
_STATUS_ACTIVE_JOB_REFRESH_LIMIT = 6

_STATE_LOCKS_GUARD = threading.Lock()
_STATE_LOCKS: Dict[str, threading.RLock] = {}
_MODEL_OPERATION_LOCKS_GUARD = threading.Lock()
_MODEL_OPERATION_LOCKS: Dict[str, threading.RLock] = {}


def _state_lock_for(state_file: Path) -> threading.RLock:
    state_key = str(state_file.expanduser().resolve(strict=False))
    with _STATE_LOCKS_GUARD:
        lock = _STATE_LOCKS.get(state_key)
        if lock is None:
            lock = threading.RLock()
            _STATE_LOCKS[state_key] = lock
    return lock


def _model_operation_lock_for(state_file: Path, model_name: str) -> threading.RLock:
    lock_key = f"{state_file.expanduser().resolve(strict=False)}::{model_name}"
    with _MODEL_OPERATION_LOCKS_GUARD:
        lock = _MODEL_OPERATION_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.RLock()
            _MODEL_OPERATION_LOCKS[lock_key] = lock
        return lock


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


def _normalize_job_tolerations(raw: Any, *, source_name: str) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HotLoaderError(f"{source_name} 必须是 JSON 数组或 Python 列表")

    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise HotLoaderError(f"{source_name}[{index}] 必须是对象")
        normalized.append(dict(item))
    return normalized


def _parse_job_tolerations(raw_text: str | None, *, source_name: str) -> List[Dict[str, Any]]:
    if raw_text is None or not raw_text.strip():
        return []
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HotLoaderError(f"{source_name} 必须是合法 JSON") from exc
    return _normalize_job_tolerations(parsed, source_name=source_name)


def _derive_job_volume_mount_path(model_target_path: str) -> str:
    normalized = model_target_path.strip()
    if not normalized:
        raise HotLoaderError("MODEL_TARGET_PATH 不能为空")

    posix_path = PurePosixPath(normalized)
    parent = str(posix_path.parent)
    return parent if normalized.startswith("/") and parent not in {"", ".", "/"} else str(posix_path)


def _derive_job_staging_root(model_target_path: str) -> str:
    mount_path = _derive_job_volume_mount_path(model_target_path).rstrip("/")
    return f"{mount_path}/.staging" if mount_path else "/.staging"


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
    target_model_repository = _env_default(*_MODEL_TARGET_PATH_ENV_NAMES)
    explicit_state_file = _env_default(*_STATE_FILE_ENV_NAMES)
    explicit_staging_root = _env_default(*_STAGING_ROOT_ENV_NAMES)

    if explicit_runtime_root:
        runtime_root = Path(explicit_runtime_root).expanduser()
        model_repository = (
            Path(explicit_model_repository).expanduser()
            if explicit_model_repository
            else Path(target_model_repository).expanduser()
            if target_model_repository
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

    if explicit_model_repository or target_model_repository:
        model_repository = Path(explicit_model_repository or target_model_repository).expanduser()
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
    model_source_path: str = "/trt_models"
    model_target_path: str = "/repository/trt_models"
    triton_repository_pvc: str = "triton-repository-pvc"
    k8s_namespace: str = "default"
    model_image_registry_prefix: str = "ccr.ccs.tencentyun.com/clobotics/"
    job_ttl_seconds_after_finished: int = 0
    job_backoff_limit: int = 1
    model_copy_cpu_request: str = "100m"
    model_copy_memory_request: str = "256Mi"
    model_copy_cpu_limit: str = "1"
    model_copy_memory_limit: str = "1Gi"
    max_concurrent_jobs: int = 0
    triton_reload_max_attempts: int = _TRITON_RELOAD_DEFAULT_MAX_ATTEMPTS
    triton_reload_retry_base_seconds: float = _TRITON_RELOAD_DEFAULT_RETRY_BASE_SECONDS
    triton_reload_retry_max_seconds: float = _TRITON_RELOAD_DEFAULT_RETRY_MAX_SECONDS
    triton_reload_timeout_seconds: float = _TRITON_RELOAD_DEFAULT_TIMEOUT_SECONDS
    job_image_pull_policy: str = "IfNotPresent"
    job_tolerations: List[Dict[str, Any]] = field(default_factory=list)
    repository_maintenance_image: str | None = None
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        self.triton_url = self.triton_url.rstrip("/")
        if self.triton_metrics_url:
            self.triton_metrics_url = self.triton_metrics_url.rstrip("/")
        self.model_repository = Path(self.model_repository).expanduser().resolve()
        self.state_file = Path(self.state_file).expanduser().resolve()
        self.staging_root = Path(self.staging_root).expanduser().resolve()
        self.repository_maintenance_image = (
            self.repository_maintenance_image.strip() if self.repository_maintenance_image else None
        ) or None
        if self.triton_reload_max_attempts < 1:
            raise HotLoaderError("triton_reload_max_attempts 必须大于等于 1")
        if self.triton_reload_retry_base_seconds <= 0:
            raise HotLoaderError("triton_reload_retry_base_seconds 必须大于 0")
        if self.triton_reload_retry_max_seconds < self.triton_reload_retry_base_seconds:
            raise HotLoaderError("triton_reload_retry_max_seconds 不能小于 triton_reload_retry_base_seconds")
        if self.triton_reload_timeout_seconds <= 0:
            raise HotLoaderError("triton_reload_timeout_seconds 必须大于 0")
        self.job_tolerations = _normalize_job_tolerations(
            self.job_tolerations,
            source_name="job_tolerations",
        )

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
            model_source_path=_env_default(*_MODEL_SOURCE_PATH_ENV_NAMES) or "/trt_models",
            model_target_path=_env_default(*_MODEL_TARGET_PATH_ENV_NAMES) or str(model_repository),
            triton_repository_pvc=_env_default(*_TRITON_REPOSITORY_PVC_ENV_NAMES) or "triton-repository-pvc",
            k8s_namespace=_env_default(*_K8S_NAMESPACE_ENV_NAMES) or "default",
            model_image_registry_prefix=_env_default(*_MODEL_IMAGE_REGISTRY_PREFIX_ENV_NAMES)
            or "ccr.ccs.tencentyun.com/clobotics/",
            job_ttl_seconds_after_finished=int(_env_default(*_JOB_TTL_SECONDS_ENV_NAMES) or "0"),
            job_backoff_limit=int(_env_default(*_JOB_BACKOFF_LIMIT_ENV_NAMES) or "1"),
            model_copy_cpu_request=_env_default(*_MODEL_COPY_CPU_REQUEST_ENV_NAMES) or "100m",
            model_copy_memory_request=_env_default(*_MODEL_COPY_MEMORY_REQUEST_ENV_NAMES) or "256Mi",
            model_copy_cpu_limit=_env_default(*_MODEL_COPY_CPU_LIMIT_ENV_NAMES) or "1",
            model_copy_memory_limit=_env_default(*_MODEL_COPY_MEMORY_LIMIT_ENV_NAMES) or "1Gi",
            max_concurrent_jobs=int(_env_default(*_MAX_CONCURRENT_JOBS_ENV_NAMES) or "0"),
            triton_reload_max_attempts=int(
                _env_default(*_TRITON_RELOAD_MAX_ATTEMPTS_ENV_NAMES)
                or str(_TRITON_RELOAD_DEFAULT_MAX_ATTEMPTS)
            ),
            triton_reload_retry_base_seconds=float(
                _env_default(*_TRITON_RELOAD_RETRY_BASE_SECONDS_ENV_NAMES)
                or str(_TRITON_RELOAD_DEFAULT_RETRY_BASE_SECONDS)
            ),
            triton_reload_retry_max_seconds=float(
                _env_default(*_TRITON_RELOAD_RETRY_MAX_SECONDS_ENV_NAMES)
                or str(_TRITON_RELOAD_DEFAULT_RETRY_MAX_SECONDS)
            ),
            triton_reload_timeout_seconds=float(
                _env_default(*_TRITON_RELOAD_TIMEOUT_SECONDS_ENV_NAMES)
                or str(_TRITON_RELOAD_DEFAULT_TIMEOUT_SECONDS)
            ),
            job_image_pull_policy=_env_default(*_JOB_IMAGE_PULL_POLICY_ENV_NAMES) or "IfNotPresent",
            job_tolerations=_parse_job_tolerations(
                _env_default(*_JOB_TOLERATIONS_ENV_NAMES),
                source_name=_JOB_TOLERATIONS_ENV_NAMES[0],
            ),
            repository_maintenance_image=_env_default(*_REPOSITORY_MAINTENANCE_IMAGE_ENV_NAMES),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triton_url": self.triton_url,
            "triton_metrics_url": self.triton_metrics_url,
            "model_repository": str(self.model_repository),
            "state_file": str(self.state_file),
            "staging_root": str(self.staging_root),
            "model_source_path": self.model_source_path,
            "model_target_path": self.model_target_path,
            "triton_repository_pvc": self.triton_repository_pvc,
            "k8s_namespace": self.k8s_namespace,
            "model_image_registry_prefix": self.model_image_registry_prefix,
            "job_ttl_seconds_after_finished": self.job_ttl_seconds_after_finished,
            "job_backoff_limit": self.job_backoff_limit,
            "model_copy_cpu_request": self.model_copy_cpu_request,
            "model_copy_memory_request": self.model_copy_memory_request,
            "model_copy_cpu_limit": self.model_copy_cpu_limit,
            "model_copy_memory_limit": self.model_copy_memory_limit,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "triton_reload_max_attempts": self.triton_reload_max_attempts,
            "triton_reload_retry_base_seconds": self.triton_reload_retry_base_seconds,
            "triton_reload_retry_max_seconds": self.triton_reload_retry_max_seconds,
            "triton_reload_timeout_seconds": self.triton_reload_timeout_seconds,
            "job_image_pull_policy": self.job_image_pull_policy,
            "job_tolerations": self.job_tolerations,
            "repository_maintenance_image": self.repository_maintenance_image,
            "request_timeout": self.request_timeout,
        }

    def with_updates(self, **updates: Any) -> "HotLoaderConfig":
        payload = self.to_dict()
        payload.update(updates)
        return HotLoaderConfig(**payload)


class TritonHotLoader:
    """Manage model bundles and load or unload them through Triton APIs."""

    def __init__(self, config: HotLoaderConfig | None = None) -> None:
        self.config = config or HotLoaderConfig.default()
        self._state_lock = _state_lock_for(self.config.state_file)
        self._batch_v1_api: Any | None = None
        self._core_v1_api: Any | None = None
        self._ensure_runtime_dirs()

    def _ensure_runtime_dirs(self) -> None:
        self.config.model_repository.mkdir(parents=True, exist_ok=True)
        self.config.staging_root.mkdir(parents=True, exist_ok=True)
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.config.state_file.exists():
            self._save_state(self._empty_state())

    def _controller_can_manage_repository_locally(self) -> bool:
        target = self.config.model_target_path.strip()
        if not target:
            return False

        normalized_target = str(PurePosixPath(target)).rstrip("/") or "/"
        if target.startswith("/") and not normalized_target.startswith("/"):
            normalized_target = f"/{normalized_target.lstrip('/')}"

        local_repository = self.config.model_repository.as_posix().rstrip("/") or "/"
        return local_repository == normalized_target

    def _uses_job_only_repository(self) -> bool:
        return not self._controller_can_manage_repository_locally()

    def _job_repository_path(self) -> Path:
        return Path(self.config.model_target_path).expanduser()

    def _job_repository_mount_path(self) -> Path:
        return Path(_derive_job_volume_mount_path(self.config.model_target_path)).expanduser()

    def _controller_can_access_job_repository_locally(self) -> bool:
        try:
            mount_path = self._job_repository_mount_path()
        except HotLoaderError:
            return False
        return mount_path.exists() and mount_path.is_dir()

    def _model_operation_lock(self, model_name: str) -> threading.RLock:
        return _model_operation_lock_for(self.config.state_file, model_name)

    def _uses_repository_sync_mode(self) -> bool:
        return self._uses_job_only_repository() and self._controller_can_access_job_repository_locally()

    def _can_verify_local_model_repository(self) -> bool:
        return not self._uses_job_only_repository() or self._uses_repository_sync_mode()

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {"aliases": {}, "jobs": {}, "updated_at": None}

    def _load_state(self) -> Dict[str, Any]:
        with self._state_lock:
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

        jobs = data.get("jobs", {})
        if not isinstance(jobs, dict):
            raise HotLoaderError("状态文件 jobs 字段格式错误")

        return {
            "aliases": aliases,
            "jobs": jobs,
            "updated_at": data.get("updated_at"),
        }

    def _save_state(self, state: Dict[str, Any]) -> None:
        with self._state_lock:
            temp_file = self.config.state_file.with_name(
                f"{self.config.state_file.name}.{uuid.uuid4().hex}.tmp"
            )
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

    @staticmethod
    def _validate_model_name(model_name: str) -> str:
        normalized = model_name.strip()
        if not normalized:
            raise HotLoaderError("model_name 不能为空")
        if not _MODEL_NAME_REQUEST_PATTERN.match(normalized):
            raise HotLoaderError("model_name 只允许小写字母、数字、-、_")
        return normalized

    def _validate_image_ref(self, image_ref: str) -> str:
        normalized = image_ref.strip()
        if not normalized:
            raise HotLoaderError("image 不能为空")
        if not _IMAGE_REF_PATTERN.match(normalized):
            raise HotLoaderError("image 包含非法字符")

        prefix = self.config.model_image_registry_prefix.strip()
        if prefix and not normalized.startswith(prefix):
            raise HotLoaderError(
                f"image 必须以允许的 registry 前缀开头: {self.config.model_image_registry_prefix}"
            )
        return normalized

    @staticmethod
    def _extract_model_name_token_from_image_ref(image_ref: str) -> tuple[str, bool]:
        ref_without_digest, _, _ = image_ref.partition("@")
        last_slash_index = ref_without_digest.rfind("/")
        last_colon_index = ref_without_digest.rfind(":")
        if last_colon_index > last_slash_index:
            return ref_without_digest[last_colon_index + 1 :], True
        return ref_without_digest[last_slash_index + 1 :], False

    @classmethod
    def _derive_model_name_from_image_ref(cls, image_ref: str) -> str:
        raw_candidate, extracted_from_tag = cls._extract_model_name_token_from_image_ref(image_ref)
        normalized_candidate = raw_candidate.strip()
        if extracted_from_tag:
            normalized_candidate = _IMAGE_TAG_RELEASE_SUFFIX_PATTERN.sub("", normalized_candidate)

        normalized_candidate = normalized_candidate.lower().replace("-", "_").replace(".", "_")
        normalized_candidate = _MODEL_NAME_DERIVE_SANITIZE_PATTERN.sub("_", normalized_candidate)
        normalized_candidate = re.sub(r"_+", "_", normalized_candidate).strip("_")
        if not normalized_candidate:
            raise HotLoaderError(f"无法从 image 自动提取 model_name: {image_ref}")
        if not _MODEL_NAME_REQUEST_PATTERN.match(normalized_candidate):
            raise HotLoaderError(f"根据 image 提取出的 model_name 非法: {normalized_candidate}")
        return normalized_candidate

    def _resolve_model_name_for_image(self, model_name: str | None, image_ref: str) -> tuple[str, str]:
        normalized_image_ref = self._validate_image_ref(str(image_ref or ""))
        normalized_model_name = str(model_name or "").strip()
        if normalized_model_name:
            return self._validate_model_name(normalized_model_name), normalized_image_ref
        return self._derive_model_name_from_image_ref(normalized_image_ref), normalized_image_ref

    @staticmethod
    def _normalize_k8s_name(value: str, *, limit: int = 63) -> str:
        lowered = value.strip().lower().replace("_", "-")
        lowered = _K8S_NAME_SANITIZE_PATTERN.sub("-", lowered).strip("-")
        lowered = re.sub(r"-{2,}", "-", lowered)
        return lowered[:limit].rstrip("-")

    def _job_name_for_model(self, model_name: str) -> str:
        normalized_model_name = self._normalize_k8s_name(model_name, limit=42)
        if not normalized_model_name:
            raise HotLoaderError("model_name 无法转换为合法的 Kubernetes Job 名称")
        suffix = hashlib.sha1(f"{model_name}:{self._utc_now()}".encode("utf-8")).hexdigest()[:8]
        return f"model-copy-{normalized_model_name}-{suffix}"[:63].rstrip("-")

    @staticmethod
    def _alias_for_model(model_name: str) -> str:
        return f"model_{model_name}"

    @staticmethod
    def _parse_iso_datetime(value: str) -> datetime | None:
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            return None

    def _normalize_callback_config(self, callback: Mapping[str, Any] | None) -> Dict[str, Any] | None:
        if callback is None:
            return None
        if not isinstance(callback, Mapping):
            raise HotLoaderError("callback 必须是对象")

        url = str(callback.get("url") or "").strip()
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise HotLoaderError("callback.url 必须是合法的 http/https URL")

        raw_events = callback.get("events")
        if raw_events is None:
            events = ["terminal"]
        elif isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):
            events = sorted(
                {
                    str(item).strip().lower()
                    for item in raw_events
                    if str(item).strip()
                }
            )
        else:
            raise HotLoaderError("callback.events 必须是字符串数组")

        if not events:
            events = ["terminal"]
        if any(event != "terminal" for event in events):
            raise HotLoaderError("callback.events 目前只支持 terminal")

        token = str(callback.get("token") or "").strip() or None
        return {
            "url": url,
            "events": events,
            "token": token,
            "attempts": 0,
            "delivered_at": None,
            "last_attempt_at": None,
            "last_error": None,
            "last_event_id": None,
            "next_attempt_at": None,
        }

    @staticmethod
    def _sanitize_callback_config(callback: Mapping[str, Any]) -> Dict[str, Any]:
        sanitized = dict(callback)
        sanitized.pop("token", None)
        return sanitized

    def _sanitize_job_metadata(self, meta: Mapping[str, Any]) -> Dict[str, Any]:
        sanitized = dict(meta)
        callback = sanitized.get("callback")
        if isinstance(callback, Mapping):
            sanitized["callback"] = self._sanitize_callback_config(callback)
        return sanitized

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

    def _ensure_k8s_clients(self) -> tuple[Any, Any]:
        if self._batch_v1_api is not None and self._core_v1_api is not None:
            return self._batch_v1_api, self._core_v1_api

        try:
            from kubernetes import client as k8s_client  # type: ignore[import-not-found]
            from kubernetes import config as k8s_config  # type: ignore[import-not-found]
        except ImportError as exc:
            raise HotLoaderError("缺少 kubernetes 依赖，请重新构建镜像后再启动 controller") from exc

        try:
            k8s_config.load_incluster_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception as exc:
                raise HotLoaderError(f"无法加载 Kubernetes 配置: {exc}") from exc

        self._batch_v1_api = k8s_client.BatchV1Api()
        self._core_v1_api = k8s_client.CoreV1Api()
        return self._batch_v1_api, self._core_v1_api

    def _get_batch_v1_api(self) -> Any:
        batch_api, _ = self._ensure_k8s_clients()
        return batch_api

    def _get_core_v1_api(self) -> Any:
        _, core_api = self._ensure_k8s_clients()
        return core_api

    @staticmethod
    def _exception_text(exc: Exception) -> str:
        body = getattr(exc, "body", None)
        reason = getattr(exc, "reason", None)
        if isinstance(body, str) and body.strip():
            return body.strip()
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
        return str(exc)

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

    def _active_job_count(self) -> int:
        batch_api = self._get_batch_v1_api()
        try:
            response = batch_api.list_namespaced_job(
                namespace=self.config.k8s_namespace,
                label_selector=f"app={_CONTROLLER_LABEL},job-role=model-copy",
            )
        except Exception as exc:
            raise HotLoaderError(f"查询 Kubernetes Job 失败: {self._exception_text(exc)}") from exc

        active = 0
        for job in getattr(response, "items", []):
            status = getattr(job, "status", None)
            if (getattr(status, "active", 0) or 0) > 0:
                active += 1
                continue
            if (getattr(status, "succeeded", 0) or 0) == 0 and (getattr(status, "failed", 0) or 0) == 0:
                active += 1
        return active

    def _assert_job_capacity(self) -> None:
        if self.config.max_concurrent_jobs <= 0:
            return
        active = self._active_job_count()
        if active >= self.config.max_concurrent_jobs:
            raise HotLoaderError(
                f"当前运行中的 Job 数量已达到上限 {self.config.max_concurrent_jobs}，请稍后再试"
            )

    def _build_job_manifest(self, job_name: str, model_name: str, image_ref: str) -> Dict[str, Any]:
        model_label = self._normalize_k8s_name(model_name, limit=63) or "unknown-model"
        job_volume_mount_path = _derive_job_volume_mount_path(self.config.model_target_path)
        job_staging_root = _derive_job_staging_root(self.config.model_target_path)
        copy_script = "\n".join(
            [
                "set -eu",
                'echo "MODEL_NAME=${MODEL_NAME}"',
                'echo "MODEL_SOURCE_PATH=${MODEL_SOURCE_PATH}"',
                'echo "MODEL_TARGET_PATH=${MODEL_TARGET_PATH}"',
                'SOURCE_DIR="${MODEL_SOURCE_PATH%/}/${MODEL_NAME}"',
                'if [ -d "${SOURCE_DIR}" ]; then',
                '  COPY_SOURCE="${SOURCE_DIR}"',
                'elif [ -d "${MODEL_SOURCE_PATH}" ]; then',
                '  COPY_SOURCE="${MODEL_SOURCE_PATH}"',
                "else",
                '  echo "source path not found: ${MODEL_SOURCE_PATH} or ${SOURCE_DIR}"',
                "  exit 1",
                "fi",
                'echo "COPY_SOURCE=${COPY_SOURCE}"',
                'TARGET_DIR="${MODEL_TARGET_PATH%/}/${MODEL_NAME}"',
                'STAGING_DIR="${STAGING_ROOT%/}/${MODEL_NAME}/${JOB_NAME}"',
                'BACKUP_DIR="${STAGING_ROOT%/}/${MODEL_NAME}/${JOB_NAME}.backup"',
                'rm -rf "${STAGING_DIR}" "${BACKUP_DIR}"',
                'mkdir -p "$(dirname "${STAGING_DIR}")"',
                'mkdir -p "${STAGING_DIR}"',
                'cp -R "${COPY_SOURCE}/." "${STAGING_DIR}/"',
                'if [ -d "${TARGET_DIR}" ]; then mv "${TARGET_DIR}" "${BACKUP_DIR}"; fi',
                'if mv "${STAGING_DIR}" "${TARGET_DIR}"; then',
                '  rm -rf "${BACKUP_DIR}"',
                "else",
                '  if [ -d "${BACKUP_DIR}" ] && [ ! -d "${TARGET_DIR}" ]; then mv "${BACKUP_DIR}" "${TARGET_DIR}" || true; fi',
                "  exit 1",
                "fi",
                'TARGET_VERSIONS=""',
                'for VERSION_DIR in "${TARGET_DIR}"/*; do',
                '  [ -d "${VERSION_DIR}" ] || continue',
                '  VERSION="${VERSION_DIR##*/}"',
                '  case "${VERSION}" in ""|*[!0-9]*) continue ;; esac',
                '  TARGET_VERSIONS="${TARGET_VERSIONS}${TARGET_VERSIONS:+,}${VERSION}"',
                'done',
                'if [ -z "${TARGET_VERSIONS}" ]; then echo "no numeric Triton model versions found"; exit 1; fi',
                'echo "model copy done target_versions=${TARGET_VERSIONS}"',
            ]
        )

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.config.k8s_namespace,
                "labels": {
                    "app": _CONTROLLER_LABEL,
                    "job-role": "model-copy",
                    "model-name": model_label,
                },
                "annotations": {
                    "hot-loader/model-name": model_name,
                    "hot-loader/image-ref": image_ref,
                },
            },
            "spec": {
                "backoffLimit": self.config.job_backoff_limit,
                "ttlSecondsAfterFinished": self.config.job_ttl_seconds_after_finished,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": _CONTROLLER_LABEL,
                            "job-role": "model-copy",
                            "model-name": model_label,
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "tolerations": self.config.job_tolerations,
                        "containers": [
                            {
                                "name": "model-copy",
                                "image": image_ref,
                                "imagePullPolicy": self.config.job_image_pull_policy,
                                "env": [
                                    {"name": "MODEL_NAME", "value": model_name},
                                    {"name": "MODEL_SOURCE_PATH", "value": self.config.model_source_path},
                                    {"name": "MODEL_TARGET_PATH", "value": self.config.model_target_path},
                                    {"name": "STAGING_ROOT", "value": job_staging_root},
                                    {"name": "JOB_NAME", "value": job_name},
                                ],
                                "command": ["/bin/sh", "-c"],
                                "args": [copy_script],
                                "volumeMounts": [
                                    {
                                        "name": "triton-repository",
                                        "mountPath": job_volume_mount_path,
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": self.config.model_copy_cpu_request,
                                        "memory": self.config.model_copy_memory_request,
                                    },
                                    "limits": {
                                        "cpu": self.config.model_copy_cpu_limit,
                                        "memory": self.config.model_copy_memory_limit,
                                    },
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "triton-repository",
                                "persistentVolumeClaim": {
                                    "claimName": self.config.triton_repository_pvc,
                                },
                            }
                        ],
                    },
                },
            },
        }

    def _build_repository_cleanup_job_manifest(self, job_name: str, model_name: str) -> Dict[str, Any]:
        if not self.config.repository_maintenance_image:
            raise HotLoaderError(
                "当前部署使用 Job-only repository 模式；请配置 REPOSITORY_MAINTENANCE_IMAGE 以便卸载时清理 PVC 中的模型目录"
            )

        model_label = self._normalize_k8s_name(model_name, limit=63) or "unknown-model"
        job_volume_mount_path = _derive_job_volume_mount_path(self.config.model_target_path)
        cleanup_script = "\n".join(
            [
                "set -eu",
                'TARGET_DIR="${MODEL_TARGET_PATH%/}/${MODEL_NAME}"',
                'echo "cleanup target: ${TARGET_DIR}"',
                'rm -rf "${TARGET_DIR}"',
                'echo "repository cleanup done"',
            ]
        )

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.config.k8s_namespace,
                "labels": {
                    "app": _CONTROLLER_LABEL,
                    "job-role": "repository-cleanup",
                    "model-name": model_label,
                },
                "annotations": {
                    "hot-loader/model-name": model_name,
                },
            },
            "spec": {
                "backoffLimit": self.config.job_backoff_limit,
                "ttlSecondsAfterFinished": max(self.config.job_ttl_seconds_after_finished, 60),
                "template": {
                    "metadata": {
                        "labels": {
                            "app": _CONTROLLER_LABEL,
                            "job-role": "repository-cleanup",
                            "model-name": model_label,
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "tolerations": self.config.job_tolerations,
                        "containers": [
                            {
                                "name": "repository-cleanup",
                                "image": self.config.repository_maintenance_image,
                                "imagePullPolicy": self.config.job_image_pull_policy,
                                "env": [
                                    {"name": "MODEL_NAME", "value": model_name},
                                    {"name": "MODEL_TARGET_PATH", "value": self.config.model_target_path},
                                ],
                                "command": ["/bin/sh", "-c"],
                                "args": [cleanup_script],
                                "volumeMounts": [
                                    {
                                        "name": "triton-repository",
                                        "mountPath": job_volume_mount_path,
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": self.config.model_copy_cpu_request,
                                        "memory": self.config.model_copy_memory_request,
                                    },
                                    "limits": {
                                        "cpu": self.config.model_copy_cpu_limit,
                                        "memory": self.config.model_copy_memory_limit,
                                    },
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "triton-repository",
                                "persistentVolumeClaim": {
                                    "claimName": self.config.triton_repository_pvc,
                                },
                            }
                        ],
                    },
                },
            },
        }

    def _wait_for_repository_job_completion(
        self,
        job_name: str,
        *,
        timeout_seconds: float = _REPOSITORY_JOB_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _REPOSITORY_JOB_TERMINAL_POLL_INTERVAL_SECONDS,
    ) -> None:
        batch_api = self._get_batch_v1_api()
        deadline = time.monotonic() + timeout_seconds

        while True:
            try:
                job = batch_api.read_namespaced_job(
                    name=job_name,
                    namespace=self.config.k8s_namespace,
                )
            except Exception as exc:
                raise HotLoaderError(f"查询 repository cleanup Job 失败: {self._exception_text(exc)}") from exc

            job_status = getattr(job, "status", None)
            if (getattr(job_status, "succeeded", 0) or 0) > 0:
                return
            if (getattr(job_status, "failed", 0) or 0) > 0:
                pods = self._list_job_pods(job_name)
                pod_name = getattr(getattr(pods[0], "metadata", None), "name", None) if pods else None
                logs = self._read_pod_logs(pod_name)
                detail = logs or "repository cleanup Job 执行失败"
                raise HotLoaderError(detail)

            if time.monotonic() >= deadline:
                raise HotLoaderError(
                    f"等待 repository cleanup Job 超时 ({int(timeout_seconds)}s): {job_name}"
                )

            time.sleep(max(poll_interval_seconds, 0))

    def _delete_model_directory_via_job(self, model_name: str) -> None:
        job_suffix = uuid.uuid4().hex[:8]
        model_label = self._normalize_k8s_name(model_name, limit=40) or "unknown-model"
        job_name = f"{model_label}-cleanup-{job_suffix}"
        manifest = self._build_repository_cleanup_job_manifest(job_name, model_name)
        batch_api = self._get_batch_v1_api()

        try:
            batch_api.create_namespaced_job(
                namespace=self.config.k8s_namespace,
                body=manifest,
            )
        except Exception as exc:
            raise HotLoaderError(f"创建 repository cleanup Job 失败: {self._exception_text(exc)}") from exc

        self._wait_for_repository_job_completion(job_name)

    @staticmethod
    def _model_bundle_ready(model_dir: Path) -> bool:
        if not model_dir.exists() or not model_dir.is_dir():
            return False

        visible_entries = [path for path in model_dir.iterdir() if not path.name.startswith(".")]
        if not visible_entries:
            return False

        version_dirs = [
            path
            for path in visible_entries
            if path.is_dir() and _TRITON_VERSION_DIR_PATTERN.match(path.name)
        ]
        if not version_dirs:
            return any(path.is_file() for path in visible_entries)

        for version_dir in version_dirs:
            if not any(not child.name.startswith(".") for child in version_dir.iterdir()):
                return False
        return True

    def _wait_for_model_bundle_ready(
        self,
        model_dir: Path,
        *,
        timeout_seconds: float = _REPOSITORY_SYNC_VISIBILITY_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _REPOSITORY_SYNC_VISIBILITY_POLL_INTERVAL_SECONDS,
        missing_hint: str = "请检查 Job 是否把文件复制到了 PVC",
    ) -> None:
        deadline = time.monotonic() + timeout_seconds

        while True:
            if self._model_bundle_ready(model_dir):
                return
            if time.monotonic() >= deadline:
                if model_dir.exists():
                    raise HotLoaderError(
                        f"模型目录尚未准备完成: {model_dir}，{missing_hint}"
                    )
                raise HotLoaderError(
                    f"模型目录不存在: {model_dir}，{missing_hint}"
                )
            time.sleep(max(poll_interval_seconds, 0))

    def _sync_model_from_job_repository(self, model_name: str) -> None:
        source_dir = self._job_repository_path() / model_name
        self._wait_for_model_bundle_ready(source_dir)

        target_dir = self.config.model_repository / model_name
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        sync_root = self.config.staging_root / model_name
        sync_root.mkdir(parents=True, exist_ok=True)
        sync_token = uuid.uuid4().hex
        staging_dir = sync_root / f"sync-{sync_token}"
        backup_dir = sync_root / f"sync-{sync_token}.backup"

        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)

        try:
            shutil.copytree(source_dir, staging_dir)
            if target_dir.exists():
                shutil.move(str(target_dir), str(backup_dir))
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.move(str(staging_dir), str(target_dir))
            self._wait_for_model_bundle_ready(
                target_dir,
                missing_hint="请检查控制器临时目录是否可写",
            )
        except Exception as exc:
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            if backup_dir.exists() and not target_dir.exists():
                shutil.move(str(backup_dir), str(target_dir))
            raise HotLoaderError(
                f"从 Job repository 同步模型到本地临时目录失败: {source_dir} -> {target_dir} ({exc})"
            ) from exc
        else:
            shutil.rmtree(backup_dir, ignore_errors=True)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _delete_model_directory_from_job_repository(self, model_name: str) -> None:
        target_dir = self._job_repository_path() / model_name
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

    def _update_job_state(self, job_name: str, *, touch_updated_at: bool = True, **updates: Any) -> Dict[str, Any]:
        with self._state_lock:
            state = self._hydrate_state_aliases(self._load_state())
            jobs = state.setdefault("jobs", {})
            current = dict(jobs.get(job_name, {})) if isinstance(jobs.get(job_name), Mapping) else {}
            current.update(updates)
            current["job_name"] = job_name
            timestamp = self._utc_now()
            if touch_updated_at or not current.get("updated_at"):
                current["updated_at"] = timestamp
                state["updated_at"] = current["updated_at"]
            elif not state.get("updated_at"):
                state["updated_at"] = current["updated_at"]
            jobs[job_name] = current
            self._save_state(state)
            return current

    def _drop_model_from_aliases(self, aliases: Dict[str, Any], model_name: str) -> None:
        for alias, meta in list(aliases.items()):
            if not isinstance(meta, Mapping):
                continue
            hydrated = self._hydrate_alias_metadata(meta)
            models = [name for name in hydrated.get("models", []) if name != model_name]
            if len(models) == len(hydrated.get("models", [])):
                continue

            if models:
                hydrated["models"] = models
                hydrated["updated_at"] = self._utc_now()
                aliases[alias] = hydrated
            else:
                del aliases[alias]

    def _register_loaded_model(self, model_name: str, image_ref: str) -> Dict[str, Any]:
        if self._can_verify_local_model_repository() and not (self.config.model_repository / model_name).exists():
            raise HotLoaderError(
                f"模型目录不存在: {self.config.model_repository / model_name}，请检查 Job 是否把文件复制到了 PVC"
            )

        with self._state_lock:
            state = self._hydrate_state_aliases(self._load_state())
            aliases = state.setdefault("aliases", {})
            alias = self._alias_for_model(model_name)
            touched_aliases = {
                current_alias
                for current_alias, meta in aliases.items()
                if isinstance(meta, Mapping) and model_name in meta.get("models", [])
            }
            touched_aliases.add(alias)
            previous_aliases = {
                current_alias: json.loads(json.dumps(aliases[current_alias], ensure_ascii=False))
                for current_alias in touched_aliases
                if current_alias in aliases
            }

            self._drop_model_from_aliases(aliases, model_name)
            aliases[alias] = {
                "image": image_ref,
                "models": [model_name],
                "updated_at": self._utc_now(),
            }
            state["updated_at"] = self._utc_now()
            self._save_state(state)

        return {
            "alias": alias,
            "image": image_ref,
            "model_name": model_name,
            "previous_aliases": previous_aliases,
        }

    def _rollback_loaded_model_registration(self, registration: Mapping[str, Any]) -> None:
        alias = str(registration.get("alias") or "")
        image_ref = str(registration.get("image") or "")
        model_name = str(registration.get("model_name") or "")
        previous_aliases = registration.get("previous_aliases", {})
        if not alias or not image_ref or not model_name or not isinstance(previous_aliases, Mapping):
            return

        with self._state_lock:
            state = self._hydrate_state_aliases(self._load_state())
            aliases = state.setdefault("aliases", {})
            current = aliases.get(alias)
            if not isinstance(current, Mapping):
                return
            if current.get("image") != image_ref or current.get("models") != [model_name]:
                return
            aliases.pop(alias, None)
            for previous_alias, previous_meta in previous_aliases.items():
                if isinstance(previous_alias, str) and isinstance(previous_meta, Mapping):
                    aliases[previous_alias] = dict(previous_meta)
            state["updated_at"] = self._utc_now()
            self._save_state(state)

    def _list_job_pods(self, job_name: str) -> List[Any]:
        core_api = self._get_core_v1_api()
        try:
            response = core_api.list_namespaced_pod(
                namespace=self.config.k8s_namespace,
                label_selector=f"job-name={job_name}",
            )
        except Exception as exc:
            raise HotLoaderError(f"查询 Job Pod 失败: {self._exception_text(exc)}") from exc
        return list(getattr(response, "items", []) or [])

    def _read_pod_logs(self, pod_name: str | None) -> str | None:
        if not pod_name:
            return None

        core_api = self._get_core_v1_api()
        try:
            logs = core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.config.k8s_namespace,
                tail_lines=200,
            )
        except Exception:
            return None
        return logs.strip() if isinstance(logs, str) and logs.strip() else None

    def _read_pod_events(self, pod_name: str | None) -> List[Dict[str, Any]]:
        if not pod_name:
            return []

        core_api = self._get_core_v1_api()
        try:
            response = core_api.list_namespaced_event(
                namespace=self.config.k8s_namespace,
                field_selector=(
                    f"involvedObject.name={pod_name},"
                    f"involvedObject.namespace={self.config.k8s_namespace}"
                ),
            )
        except Exception:
            return []

        events: List[Dict[str, Any]] = []
        for item in getattr(response, "items", []) or []:
            events.append(
                {
                    "type": getattr(item, "type", None),
                    "reason": getattr(item, "reason", None),
                    "message": getattr(item, "message", None),
                    "count": getattr(item, "count", None),
                }
            )
        return events

    @staticmethod
    def _pod_waiting_state(pod: Any) -> tuple[str | None, str | None]:
        statuses = getattr(getattr(pod, "status", None), "container_statuses", None) or []
        for status in statuses:
            waiting = getattr(getattr(status, "state", None), "waiting", None)
            if waiting is not None:
                return getattr(waiting, "reason", None), getattr(waiting, "message", None)
        return None, None

    @staticmethod
    def _pod_scheduling_state(pod: Any) -> tuple[str | None, str | None]:
        conditions = getattr(getattr(pod, "status", None), "conditions", None) or []
        for condition in conditions:
            if getattr(condition, "type", None) != "PodScheduled":
                continue
            if str(getattr(condition, "status", "")).lower() != "false":
                continue
            return getattr(condition, "reason", None), getattr(condition, "message", None)
        return None, None

    @staticmethod
    def _latest_event_detail(events: Sequence[Mapping[str, Any]]) -> str | None:
        for item in reversed(list(events)):
            message = str(item.get("message") or "").strip()
            if message:
                return message
            reason = str(item.get("reason") or "").strip()
            if reason:
                return reason
        return None

    def _determine_job_status(
        self,
        job: Any,
        pods: Sequence[Any],
        events: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str]:
        job_status = getattr(job, "status", None)
        if (getattr(job_status, "succeeded", 0) or 0) > 0:
            return "COPY_SUCCEEDED", "模型文件复制已完成"
        if (getattr(job_status, "failed", 0) or 0) > 0:
            return "COPY_FAILED", "Job 已失败，请检查 Pod 日志"

        if pods:
            pod = pods[0]
            phase = getattr(getattr(pod, "status", None), "phase", None) or "Unknown"
            waiting_reason, waiting_message = self._pod_waiting_state(pod)
            scheduling_reason, scheduling_message = self._pod_scheduling_state(pod)
            event_detail = self._latest_event_detail(events)
            if waiting_reason in {"ErrImagePull", "ImagePullBackOff"}:
                detail = waiting_message or waiting_reason
                return "COPY_FAILED", f"镜像拉取失败: {detail}"
            if phase == "Running":
                return "COPY_RUNNING", "模型复制容器正在运行"
            if phase == "Pending":
                if scheduling_reason == "Unschedulable":
                    detail = scheduling_message or event_detail or "Pod 调度失败"
                    return "SCHEDULING", detail
                detail = waiting_message or waiting_reason or event_detail or "等待调度或拉取镜像"
                return "IMAGE_PULLING", detail
            if phase == "Failed":
                return "COPY_FAILED", waiting_message or event_detail or "Pod 运行失败"

        return "JOB_CREATED", self._latest_event_detail(events) or "Job 已创建，等待 Kubernetes 调度"

    @staticmethod
    def _normalize_target_versions(versions: Iterable[Any] | None) -> List[str]:
        if versions is None:
            return []
        return sorted({str(version) for version in versions if str(version).isdigit()}, key=int)

    @staticmethod
    def _target_versions_from_copy_logs(logs: str | None) -> List[str]:
        if not logs:
            return []
        matches = re.findall(r"(?:^|\s)target_versions=([0-9][0-9,]*)(?:\s|$)", logs)
        if not matches:
            return []
        return TritonHotLoader._normalize_target_versions(matches[-1].split(","))

    @staticmethod
    def _target_versions_from_image(image_ref: str) -> List[str]:
        tag = image_ref.rsplit(":", 1)[-1]
        match = re.search(r"(?:^|[-_])(\d{8})(?:[-_]\d{6})?$", tag)
        return [match.group(1)] if match else []

    def _model_ready_in_triton(
        self,
        model_name: str,
        *,
        target_versions: Iterable[Any] | None = None,
    ) -> bool:
        expected_versions = set(self._normalize_target_versions(target_versions))
        for item in self.list_repository_models(safe=True):
            if item.get("name") != model_name:
                continue
            if expected_versions and str(item.get("version") or "") not in expected_versions:
                continue
            state_text = str(item.get("state") or "").upper()
            if state_text == "READY":
                return True
        return False

    def _model_unloaded_in_triton(self, model_name: str) -> bool:
        matched = False
        for item in self.list_repository_models():
            if item.get("name") != model_name:
                continue
            matched = True
            state_text = str(item.get("state") or "").upper()
            if state_text == "READY":
                return False
        return True if matched else True

    def _wait_for_model_unloaded(
        self,
        model_name: str,
        *,
        timeout_seconds: float = _SYNC_UNLOAD_DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _SYNC_UNLOAD_DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds

        while True:
            if self._model_unloaded_in_triton(model_name):
                return

            if time.monotonic() >= deadline:
                raise HotLoaderError(
                    f"等待 Triton 完成 unload 超时 ({int(timeout_seconds)}s): {model_name}"
                )

            time.sleep(max(poll_interval_seconds, 0))

    def _retry_delay_seconds(self, attempts: int) -> float:
        exponent = max(attempts - 1, 0)
        return min(
            self.config.triton_reload_retry_max_seconds,
            self.config.triton_reload_retry_base_seconds * (2 ** exponent),
        )

    @staticmethod
    def _parse_utc_timestamp(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _reload_deadline_exceeded(self, cached_state: Mapping[str, Any], now: datetime) -> bool:
        started_at = self._parse_utc_timestamp(cached_state.get("triton_reload_started_at"))
        return started_at is not None and now >= started_at + timedelta(
            seconds=self.config.triton_reload_timeout_seconds
        )

    def _update_ready_job(
        self,
        job_name: str,
        model_name: str,
        image_ref: str,
        *,
        target_versions: Sequence[str],
        alias: str | None = None,
        triton_reload_attempts: int | None = None,
    ) -> Dict[str, Any]:
        updates: Dict[str, Any] = {
            "status": "MODEL_READY",
            "model_name": model_name,
            "image": image_ref,
            "target_versions": list(target_versions),
            "detail": "Triton 已确认本次交付的目标版本 READY",
            "triton_ready": True,
            "triton_reload_next_attempt_at": None,
            "error": None,
        }
        if alias is not None:
            updates["alias"] = alias
        if triton_reload_attempts is not None:
            updates["triton_reload_attempts"] = triton_reload_attempts
        return self._update_job_state(job_name, **updates)

    def _update_reload_failed_job(
        self,
        job_name: str,
        model_name: str,
        image_ref: str,
        *,
        target_versions: Sequence[str],
        detail: str,
    ) -> Dict[str, Any]:
        return self._update_job_state(
            job_name,
            status="TRITON_RELOAD_FAILED",
            model_name=model_name,
            image=image_ref,
            target_versions=list(target_versions),
            detail=detail,
            triton_ready=False,
            triton_reload_next_attempt_at=None,
            error=detail,
        )

    def _schedule_reload_retry(
        self,
        job_name: str,
        model_name: str,
        image_ref: str,
        *,
        target_versions: Sequence[str],
        attempts: int,
        started_at: str,
        last_attempt_at: str,
        error: str | None = None,
    ) -> Dict[str, Any]:
        next_attempt_at = (datetime.now(timezone.utc) + timedelta(seconds=self._retry_delay_seconds(attempts))).isoformat()
        detail = (
            f"Triton reload 第 {attempts}/{self.config.triton_reload_max_attempts} 次未就绪；"
            f"将在 {next_attempt_at} 后重试"
        )
        if error:
            detail = f"{detail}: {error}"
        return self._update_job_state(
            job_name,
            status="TRITON_RELOAD_RUNNING",
            model_name=model_name,
            image=image_ref,
            target_versions=list(target_versions),
            detail=detail,
            triton_ready=False,
            triton_reload_attempts=attempts,
            triton_reload_started_at=started_at,
            triton_reload_last_attempt_at=last_attempt_at,
            triton_reload_next_attempt_at=next_attempt_at,
            error=error,
        )

    def _finalize_successful_job(
        self,
        job_name: str,
        model_name: str,
        image_ref: str,
        *,
        target_versions: Iterable[Any] | None = None,
    ) -> Dict[str, Any]:
        with self._model_operation_lock(model_name):
            cached_state = self._load_state().get("jobs", {}).get(job_name, {})
            resolved_target_versions = self._normalize_target_versions(target_versions)
            if isinstance(cached_state, Mapping):
                final_status = str(cached_state.get("status") or "").upper()
                if not resolved_target_versions:
                    resolved_target_versions = self._normalize_target_versions(cached_state.get("target_versions"))
                if final_status in {"MODEL_READY", "TRITON_RELOAD_FAILED"}:
                    return dict(cached_state)
                if final_status == "TRITON_RELOAD_RUNNING":
                    if self._model_ready_in_triton(model_name, target_versions=resolved_target_versions):
                        return self._update_ready_job(
                            job_name,
                            model_name,
                            image_ref,
                            target_versions=resolved_target_versions,
                            triton_reload_attempts=int(cached_state.get("triton_reload_attempts") or 0),
                        )
                    now = datetime.now(timezone.utc)
                    attempts = int(cached_state.get("triton_reload_attempts") or 0)
                    if attempts >= self.config.triton_reload_max_attempts:
                        return self._update_reload_failed_job(
                            job_name,
                            model_name,
                            image_ref,
                            target_versions=resolved_target_versions,
                            detail=f"Triton reload 超过最大尝试次数 {self.config.triton_reload_max_attempts}",
                        )
                    if self._reload_deadline_exceeded(cached_state, now):
                        return self._update_reload_failed_job(
                            job_name,
                            model_name,
                            image_ref,
                            target_versions=resolved_target_versions,
                            detail=f"等待 Triton 目标版本 READY 超时 ({int(self.config.triton_reload_timeout_seconds)}s)",
                        )
                    next_attempt_at = self._parse_utc_timestamp(cached_state.get("triton_reload_next_attempt_at"))
                    if next_attempt_at is not None and now < next_attempt_at:
                        return dict(cached_state)
                    reload_attempts = attempts + 1
                    reload_attempted_at = now.isoformat()
                    started_at = str(cached_state.get("triton_reload_started_at") or reload_attempted_at)
                    try:
                        self._load_model(model_name)
                        ready = self._model_ready_in_triton(model_name, target_versions=resolved_target_versions)
                    except Exception as exc:
                        if reload_attempts >= self.config.triton_reload_max_attempts:
                            return self._update_reload_failed_job(
                                job_name,
                                model_name,
                                image_ref,
                                target_versions=resolved_target_versions,
                                detail=f"Triton reload 第 {reload_attempts} 次失败: {exc}",
                            )
                        return self._schedule_reload_retry(
                            job_name,
                            model_name,
                            image_ref,
                            target_versions=resolved_target_versions,
                            attempts=reload_attempts,
                            started_at=started_at,
                            last_attempt_at=reload_attempted_at,
                            error=str(exc),
                        )
                    if ready:
                        return self._update_ready_job(
                            job_name,
                            model_name,
                            image_ref,
                            target_versions=resolved_target_versions,
                            triton_reload_attempts=reload_attempts,
                        )
                    if reload_attempts >= self.config.triton_reload_max_attempts:
                        return self._update_reload_failed_job(
                            job_name,
                            model_name,
                            image_ref,
                            target_versions=resolved_target_versions,
                            detail=f"Triton reload 第 {reload_attempts} 次后目标版本仍未 READY",
                        )
                    return self._schedule_reload_retry(
                        job_name,
                        model_name,
                        image_ref,
                        target_versions=resolved_target_versions,
                        attempts=reload_attempts,
                        started_at=started_at,
                        last_attempt_at=reload_attempted_at,
                    )

            if self._uses_repository_sync_mode():
                self._sync_model_from_job_repository(model_name)

            registration = self._register_loaded_model(model_name, image_ref)
            self._update_job_state(
                job_name,
                status="TRITON_RELOAD_RUNNING",
                model_name=model_name,
                image=image_ref,
                alias=registration["alias"],
                detail="模型文件复制完成，正在请求 Triton load",
                triton_ready=False,
                error=None,
            )
            try:
                self._load_model(model_name)
                ready = self._model_ready_in_triton(model_name, target_versions=resolved_target_versions)
            except Exception as exc:
                attempted_at = self._utc_now()
                return self._schedule_reload_retry(
                    job_name,
                    model_name,
                    image_ref,
                    target_versions=resolved_target_versions,
                    attempts=1,
                    started_at=attempted_at,
                    last_attempt_at=attempted_at,
                    error=str(exc),
                )

            if ready:
                return self._update_ready_job(
                    job_name,
                    model_name,
                    image_ref,
                    target_versions=resolved_target_versions,
                    alias=registration["alias"],
                    triton_reload_attempts=1,
                )
            attempted_at = self._utc_now()
            return self._schedule_reload_retry(
                job_name,
                model_name,
                image_ref,
                target_versions=resolved_target_versions,
                attempts=1,
                started_at=attempted_at,
                last_attempt_at=attempted_at,
            )

    def create_model_copy_job(
        self,
        model_name: str | None,
        image_ref: str,
        *,
        callback: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized_model_name, normalized_image_ref = self._resolve_model_name_for_image(model_name, image_ref)
        callback_config = self._normalize_callback_config(callback)
        with self._model_operation_lock(normalized_model_name):
            state = self._load_state()
            for existing_job_name, existing in state.get("jobs", {}).items():
                if not isinstance(existing, Mapping):
                    continue
                if existing.get("model_name") != normalized_model_name:
                    continue
                if str(existing.get("status") or "").upper() not in _ACTIVE_JOB_STATUSES:
                    continue
                if existing.get("image") != normalized_image_ref:
                    raise HotLoaderConflictError(
                        f"模型 {normalized_model_name} 已有活跃 operation {existing_job_name}，"
                        "其镜像不同；请等待当前 operation 进入终态后再提交"
                    )
                return {
                    "success": True,
                    "job_name": existing_job_name,
                    "model_name": normalized_model_name,
                    "image": normalized_image_ref,
                    "status": existing.get("status"),
                    "callback_registered": bool(existing.get("callback")),
                    "target_versions": self._normalize_target_versions(existing.get("target_versions")),
                    "reused": True,
                }

            self._assert_job_capacity()
            job_name = self._job_name_for_model(normalized_model_name)
            manifest = self._build_job_manifest(job_name, normalized_model_name, normalized_image_ref)
            batch_api = self._get_batch_v1_api()

            try:
                created = batch_api.create_namespaced_job(
                    namespace=self.config.k8s_namespace,
                    body=manifest,
                )
            except Exception as exc:
                raise HotLoaderError(f"创建 Kubernetes Job 失败: {self._exception_text(exc)}") from exc

            metadata = getattr(created, "metadata", None)
            target_versions = self._target_versions_from_image(normalized_image_ref)
            self._update_job_state(
                job_name,
                status="JOB_CREATED",
                model_name=normalized_model_name,
                image=normalized_image_ref,
                callback=callback_config,
                namespace=self.config.k8s_namespace,
                job_uid=getattr(metadata, "uid", None),
                created_at=self._utc_now(),
                target_versions=target_versions,
            )

            return {
                "success": True,
                "job_name": job_name,
                "model_name": normalized_model_name,
                "image": normalized_image_ref,
                "status": "JOB_CREATED",
                "callback_registered": bool(callback_config),
                "target_versions": target_versions,
                "reused": False,
            }

    def wait_for_job_terminal_state(
        self,
        job_name: str,
        *,
        timeout_seconds: float = _SYNC_LOAD_DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _SYNC_LOAD_DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_payload: Dict[str, Any] | None = None

        while True:
            payload = self.get_job_status(job_name)
            last_payload = payload
            status = str(payload.get("status") or "").upper()
            if status in _SYNC_LOAD_TERMINAL_STATUSES:
                return payload

            if time.monotonic() >= deadline:
                detail = str(payload.get("detail") or status or "等待超时")
                raise HotLoaderError(
                    f"等待模型加载超时 ({int(timeout_seconds)}s): {job_name} 当前状态 {status or '-'} - {detail}"
                )

            time.sleep(max(poll_interval_seconds, 0))

    def create_model_copy_job_and_wait(
        self,
        model_name: str | None,
        image_ref: str,
        *,
        callback: Mapping[str, Any] | None = None,
        timeout_seconds: float = _SYNC_LOAD_DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _SYNC_LOAD_DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> Dict[str, Any]:
        submitted = self.create_model_copy_job(model_name, image_ref, callback=callback)
        final_payload = self.wait_for_job_terminal_state(
            submitted["job_name"],
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        final_status = str(final_payload.get("status") or "").upper()
        success = final_status in _SYNC_LOAD_SUCCESS_STATUSES
        return {
            **submitted,
            **final_payload,
            "submitted_status": submitted["status"],
            "success": success,
        }

    def load_models_from_images(
        self,
        models: Iterable[Mapping[str, str]],
        *,
        callback: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        submitted: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for item in models:
            raw_model_name = item.get("model_name", "") if isinstance(item, Mapping) else ""
            raw_image_ref = item.get("image", "") if isinstance(item, Mapping) else ""
            model_name = str(raw_model_name or "")
            image_ref = str(raw_image_ref or "")
            resolved_model_name = model_name
            try:
                resolved_model_name, normalized_image_ref = self._resolve_model_name_for_image(model_name, image_ref)
                submitted.append(
                    self.create_model_copy_job(
                        resolved_model_name,
                        normalized_image_ref,
                        callback=callback,
                    )
                )
            except HotLoaderError as exc:
                errors.append(
                    {
                        "model_name": resolved_model_name,
                        "image": image_ref,
                        "error": str(exc),
                    }
                )

        return {
            "success": not errors,
            "submitted": submitted,
            "errors": errors,
            "state": self.get_managed_state(),
        }

    def load_models_from_images_sync(
        self,
        models: Iterable[Mapping[str, str]],
        *,
        callback: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        completed: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for item in models:
            raw_model_name = item.get("model_name", "") if isinstance(item, Mapping) else ""
            raw_image_ref = item.get("image", "") if isinstance(item, Mapping) else ""
            model_name = str(raw_model_name or "")
            image_ref = str(raw_image_ref or "")
            resolved_model_name = model_name
            try:
                resolved_model_name, normalized_image_ref = self._resolve_model_name_for_image(model_name, image_ref)
                result = self.create_model_copy_job_and_wait(
                    resolved_model_name,
                    normalized_image_ref,
                    callback=callback,
                )
            except HotLoaderError as exc:
                errors.append(
                    {
                        "model_name": resolved_model_name,
                        "image": image_ref,
                        "error": str(exc),
                    }
                )
                continue

            if result.get("success"):
                completed.append(result)
            else:
                errors.append(
                    {
                        "model_name": str(result.get("model_name") or resolved_model_name),
                        "image": image_ref,
                        "error": str(result.get("detail") or result.get("status") or "加载失败"),
                        "job_name": result.get("job_name"),
                        "status": result.get("status"),
                    }
                )

        return {
            "success": not errors,
            "completed": completed,
            "errors": errors,
            "state": self.get_managed_state(),
        }

    def get_job_status(self, job_name: str, *, include_logs: bool = True) -> Dict[str, Any]:
        batch_api = self._get_batch_v1_api()
        cached_jobs = self._load_state().get("jobs", {})
        cached = dict(cached_jobs.get(job_name, {})) if isinstance(cached_jobs.get(job_name), Mapping) else {}

        try:
            job = batch_api.read_namespaced_job(
                name=job_name,
                namespace=self.config.k8s_namespace,
            )
        except Exception as exc:
            if cached:
                model_name = str(cached.get("model_name") or "")
                image_ref = str(cached.get("image") or "")
                cached_status = str(cached.get("status") or "").upper()
                model_dir = self.config.model_repository / model_name if model_name else None
                source_model_dir = self._job_repository_path() / model_name if model_name and self._uses_repository_sync_mode() else None
                should_attempt_finalize = (
                    model_name
                    and image_ref
                    and cached_status in _ACTIVE_JOB_STATUSES
                    and (
                        self._uses_job_only_repository()
                        or source_model_dir is not None
                        or (model_dir is not None and model_dir.exists())
                    )
                )
                if should_attempt_finalize:
                    try:
                        finalized = self._finalize_successful_job(job_name, model_name, image_ref)
                    except HotLoaderError as finalize_exc:
                        finalized = self._update_job_state(
                            job_name,
                            status="TRITON_RELOAD_FAILED",
                            model_name=model_name,
                            image=image_ref,
                            detail=str(finalize_exc),
                            error=str(finalize_exc),
                        )
                    if str(finalized.get("status") or "").upper() == "TRITON_RELOAD_RUNNING":
                        finalized = self._update_job_state(
                            job_name,
                            touch_updated_at=False,
                            detail=(
                                f"{str(finalized.get('detail') or '').strip()}；"
                                "原始 Job 已不可读，按已落盘模型继续自动 reload"
                            ).strip("；"),
                        )
                    return self._sanitize_job_metadata(finalized)

                if cached_status in _SYNC_LOAD_TERMINAL_STATUSES:
                    return self._sanitize_job_metadata(cached)

                cached["detail"] = f"Job 当前不可读，可能已被 TTL 清理: {self._exception_text(exc)}"
                return self._sanitize_job_metadata(cached)
            raise HotLoaderError(f"查询 Kubernetes Job 失败: {self._exception_text(exc)}") from exc

        pods = self._list_job_pods(job_name)
        pod_name = getattr(getattr(pods[0], "metadata", None), "name", None) if pods else None
        events = self._read_pod_events(pod_name)
        controller_status, detail = self._determine_job_status(job, pods, events)
        model_name = (
            cached.get("model_name")
            or getattr(getattr(job, "metadata", None), "annotations", {}).get("hot-loader/model-name")
            or ""
        )
        image_ref = (
            cached.get("image")
            or getattr(getattr(job, "metadata", None), "annotations", {}).get("hot-loader/image-ref")
            or ""
        )
        logs = (
            self._read_pod_logs(pod_name)
            if include_logs or controller_status == "COPY_SUCCEEDED"
            else cached.get("logs")
        )

        if controller_status == "COPY_SUCCEEDED" and model_name and image_ref:
            target_versions = self._target_versions_from_copy_logs(logs)
            if target_versions:
                self._update_job_state(
                    job_name,
                    target_versions=target_versions,
                    model_name=model_name,
                    image=image_ref,
                )
            try:
                finalized = self._finalize_successful_job(
                    job_name,
                    model_name,
                    image_ref,
                    target_versions=target_versions,
                )
                controller_status = str(finalized.get("status") or controller_status)
                detail = str(finalized.get("detail") or detail)
            except HotLoaderError as exc:
                controller_status = "TRITON_RELOAD_FAILED"
                detail = str(exc)

        updates: Dict[str, Any] = {
            "status": controller_status,
            "detail": detail,
            "model_name": model_name,
            "image": image_ref,
            "pod_name": pod_name,
            "events": events,
        }
        if include_logs:
            updates["logs"] = logs

        payload = self._update_job_state(job_name, **updates)
        payload["job_name"] = job_name
        payload["model_name"] = model_name
        payload["pod_name"] = pod_name
        payload["logs"] = logs
        payload["events"] = events
        return self._sanitize_job_metadata(payload)

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

        if self._uses_repository_sync_mode():
            self._delete_model_directory_from_job_repository(model_name)
            return

        if self._uses_job_only_repository():
            self._delete_model_directory_via_job(model_name)

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
        hydrated["models"] = models
        hydrated.pop("model_versions", None)
        hydrated.pop("active_versions", None)
        return hydrated

    def get_managed_state(self) -> Dict[str, Any]:
        state = self._hydrate_state_aliases(self._load_state())
        aliases = {
            alias: self._hydrate_alias_metadata(meta)
            for alias, meta in state.get("aliases", {}).items()
            if isinstance(meta, Mapping)
        }
        jobs = {
            job_name: self._sanitize_job_metadata(meta)
            for job_name, meta in state.get("jobs", {}).items()
            if isinstance(meta, Mapping)
        }
        managed_images = sorted(
            [
                {
                    "id": alias,
                    "image": meta.get("image"),
                    "models": meta.get("models", []),
                    "updated_at": meta.get("updated_at"),
                }
                for alias, meta in aliases.items()
            ],
            key=lambda item: (item.get("models") or [""])[0],
        )
        managed_models = sorted(
            {
                model_name
                for meta in aliases.values()
                for model_name in meta.get("models", [])
            }
        )
        return {
            "config": self.config.to_dict(),
            "updated_at": state.get("updated_at"),
            "aliases": aliases,
            "managed_images": managed_images,
            "managed_alias_count": len(aliases),
            "managed_image_count": len(managed_images),
            "managed_model_count": len(managed_models),
            "managed_models": managed_models,
            "jobs": jobs,
            "job_count": len(jobs),
            "active_jobs": sorted(
                [
                    job_name
                    for job_name, meta in jobs.items()
                    if str(meta.get("status") or "")
                    in _ACTIVE_JOB_STATUSES
                ]
            ),
        }

    def list_pending_terminal_callbacks(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        state = self._load_state()
        jobs = state.get("jobs", {})
        if not isinstance(jobs, Mapping):
            return []

        pending: List[Dict[str, Any]] = []
        for job_name, meta in jobs.items():
            if not isinstance(meta, Mapping):
                continue
            callback = meta.get("callback")
            if not isinstance(callback, Mapping):
                continue
            if "terminal" not in callback.get("events", []):
                continue
            if str(meta.get("status") or "").upper() not in _SYNC_LOAD_TERMINAL_STATUSES:
                continue
            if str(callback.get("delivered_at") or "").strip():
                continue

            next_attempt_at = self._parse_iso_datetime(str(callback.get("next_attempt_at") or ""))
            if next_attempt_at and next_attempt_at > now:
                continue

            pending.append({"job_name": job_name, **dict(meta)})

        pending.sort(
            key=lambda item: (
                str(item.get("callback", {}).get("next_attempt_at") or ""),
                str(item.get("updated_at") or ""),
                str(item.get("job_name") or ""),
            )
        )
        if limit > 0:
            pending = pending[:limit]
        return pending

    def record_terminal_callback_result(
        self,
        job_name: str,
        *,
        delivered: bool,
        event_id: str,
        error: str | None = None,
        retry_delay_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        with self._state_lock:
            state = self._hydrate_state_aliases(self._load_state())
            jobs = state.setdefault("jobs", {})
            current = dict(jobs.get(job_name, {})) if isinstance(jobs.get(job_name), Mapping) else {}
            callback = dict(current.get("callback", {})) if isinstance(current.get("callback"), Mapping) else {}
            if not callback:
                raise HotLoaderError(f"Job 未注册 callback: {job_name}")

            now = datetime.now(timezone.utc)
            callback["attempts"] = int(callback.get("attempts") or 0) + 1
            callback["last_attempt_at"] = now.isoformat()
            callback["last_event_id"] = event_id
            callback["last_error"] = error
            callback["next_attempt_at"] = None
            if delivered:
                callback["delivered_at"] = now.isoformat()
            else:
                callback["delivered_at"] = None
                callback["next_attempt_at"] = (now + timedelta(seconds=max(retry_delay_seconds, 0.0))).isoformat()

            current["callback"] = callback
            jobs[job_name] = current
            self._save_state(state)
            return self._sanitize_job_metadata(current)

    def refresh_active_job_statuses(
        self,
        *,
        limit: int = _STATUS_ACTIVE_JOB_REFRESH_LIMIT,
        include_logs: bool = False,
    ) -> List[Dict[str, Any]]:
        state = self._load_state()
        jobs = state.get("jobs", {})
        if not isinstance(jobs, Mapping):
            return []

        active_entries = [
            (job_name, dict(meta))
            for job_name, meta in jobs.items()
            if isinstance(meta, Mapping) and str(meta.get("status") or "") in _ACTIVE_JOB_STATUSES
        ]
        unchecked_entries = [
            item for item in active_entries if not str(item[1].get("status_checked_at") or "").strip()
        ]
        checked_entries = [
            item for item in active_entries if str(item[1].get("status_checked_at") or "").strip()
        ]
        unchecked_entries.sort(
            key=lambda item: (str(item[1].get("updated_at") or ""), item[0]),
            reverse=True,
        )
        checked_entries.sort(
            key=lambda item: (
                str(item[1].get("status_checked_at") or ""),
                str(item[1].get("updated_at") or ""),
                item[0],
            )
        )
        active_entries = unchecked_entries + checked_entries
        if limit > 0:
            active_entries = active_entries[:limit]

        refreshed: List[Dict[str, Any]] = []
        for job_name, _ in active_entries:
            try:
                payload = self.get_job_status(job_name, include_logs=include_logs)
                payload = self._update_job_state(
                    job_name,
                    touch_updated_at=False,
                    status_checked_at=self._utc_now(),
                )
                refreshed.append(payload)
            except HotLoaderError:
                continue
        return refreshed

    def get_status(self) -> Dict[str, Any]:
        self.refresh_active_job_statuses(limit=_STATUS_ACTIVE_JOB_REFRESH_LIMIT, include_logs=False)
        state = self.get_managed_state()
        with ThreadPoolExecutor(max_workers=3) as executor:
            ready_future = executor.submit(self.triton_ready)
            triton_models_future = executor.submit(self.list_repository_models, safe=True)
            gpu_metrics_future = executor.submit(self.get_triton_gpu_metrics)
            ready = ready_future.result()
            triton_models = triton_models_future.result()
            gpu_metrics = gpu_metrics_future.result()
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

    def unload_alias(self, alias: str) -> Dict[str, Any]:
        self._validate_alias(alias)
        with self._state_lock:
            state = self._hydrate_state_aliases(self._load_state())
            aliases = state.setdefault("aliases", {})
            current = aliases.get(alias)
            if not current:
                raise HotLoaderError(f"alias 不存在: {alias}")
            current = dict(current)
            models = sorted(set(current.get("models", [])))

        for model_name in models:
            with self._model_operation_lock(model_name):
                self._unload_model(model_name, tolerate_missing=True)
                self._wait_for_model_unloaded(model_name)

        return {
            "alias": alias,
            "image": current.get("image"),
            "models": models,
            "unloaded_models": models,
            "detail": "仅卸载 Triton 运行态；Repository 文件和管理映射已保留，可直接 reload",
            "updated_at": self._utc_now(),
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
        if any(str(version_ref or "").strip() for version_ref in version_refs):
            raise HotLoaderError("同名模型已取消版本管理，请按 model_name 或 alias 卸载")
        raise HotLoaderError("请至少提供一个 model name 或 alias")

    def unload_models(self, model_names: Iterable[str]) -> Dict[str, Any]:
        unique_models = sorted({model_name for model_name in model_names if model_name})
        if not unique_models:
            raise HotLoaderError("请至少提供一个 model name")

        for model_name in unique_models:
            with self._model_operation_lock(model_name):
                self._unload_model(model_name, tolerate_missing=True)
                self._wait_for_model_unloaded(model_name)

        return {
            "success": True,
            "unloaded_models": unique_models,
            "affected_aliases": [],
            "detail": "仅卸载 Triton 运行态；Repository 文件和管理映射已保留，可直接 reload",
            "state": self.get_managed_state(),
        }

    def reload_models(self, model_names: Iterable[str]) -> Dict[str, Any]:
        unique_models = sorted({model_name for model_name in model_names if model_name})
        if not unique_models:
            raise HotLoaderError("请至少提供一个 model name")

        reloaded = []
        for model_name in unique_models:
            with self._model_operation_lock(model_name):
                self._load_model(model_name)
                reloaded.append(model_name)

        return {
            "success": True,
            "reloaded_models": reloaded,
            "state": self.get_managed_state(),
        }

    def get_models_overview(self) -> Dict[str, Any]:
        return {
            "managed": self.get_managed_state(),
            "triton_models": self.list_repository_models(safe=True),
        }
