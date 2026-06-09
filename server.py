from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from urllib.parse import urlsplit, urlunsplit

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hot_loader import HotLoaderConfig, HotLoaderError, TritonHotLoader


TRITON_URL_OVERRIDE_HEADER = "x-hot-triton-url"
TRITON_METRICS_PORT_OVERRIDE_HEADER = "x-hot-triton-metrics-port"


class ModelLoadRequest(BaseModel):
    image: str = Field(min_length=1)
    model_name: str | None = Field(default=None)


class ModelLoadBatchRequest(BaseModel):
    models: List[ModelLoadRequest] = Field(default_factory=list)


class ModelActionRequest(BaseModel):
    model_name: str = Field(min_length=1)


class UnloadRequest(BaseModel):
    aliases: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    versions: List[str] = Field(default_factory=list)


class ReloadRequest(BaseModel):
    models: List[str] = Field(default_factory=list)


def _build_netloc_with_port(parts, port: int) -> str:
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


def _build_metrics_url_from_port(triton_url: str, port_text: str) -> str:
    normalized_port = port_text.strip()
    if not normalized_port.isdigit():
        raise HotLoaderError("Metrics 端口必须是 1-65535 的整数")

    port = int(normalized_port)
    if port < 1 or port > 65535:
        raise HotLoaderError("Metrics 端口必须在 1-65535 之间")

    parts = urlsplit(triton_url)
    if not parts.scheme or not parts.netloc:
        raise HotLoaderError("当前 Triton URL 无法推导 Metrics endpoint")

    return urlunsplit(
        parts._replace(
            netloc=_build_netloc_with_port(parts, port),
            path="/metrics",
            query="",
            fragment="",
        )
    )


def _bytes_to_mb(value):
    if not isinstance(value, (int, float)):
        return None
    return int(round(value / (1024 * 1024)))


def _format_runtime_gpu_status(metrics: Dict[str, object]) -> Dict[str, object]:
    gpus = []
    for gpu in metrics.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        total_bytes = gpu.get("total_bytes")
        used_bytes = gpu.get("used_bytes")
        free_bytes = None
        if isinstance(total_bytes, (int, float)) and isinstance(used_bytes, (int, float)):
            free_bytes = total_bytes - used_bytes

        gpus.append(
            {
                "gpu_index": gpu.get("index"),
                "gpu_uuid": gpu.get("gpu_uuid"),
                "gpu_bus_id": gpu.get("gpu_bus_id"),
                "memory_total_mb": _bytes_to_mb(total_bytes),
                "memory_used_mb": _bytes_to_mb(used_bytes),
                "memory_free_mb": _bytes_to_mb(free_bytes),
                "memory_used_percent": gpu.get("used_percent"),
                "gpu_utilization_percent": gpu.get("utilization_percent"),
                "power_draw_w": gpu.get("power_usage_watts"),
                "power_limit_w": None,
                "temperature_c": None,
            }
        )

    status = "OK" if metrics.get("available") else "UNAVAILABLE"
    return {
        "status": status,
        "detail": metrics.get("detail"),
        "updated_at": metrics.get("updated_at"),
        "source_url": metrics.get("url"),
        "summary": metrics.get("summary"),
        "gpus": gpus,
    }


def _get_request_loader(request: Request) -> TritonHotLoader:
    base_loader = request.app.state.loader
    override_url = request.headers.get(TRITON_URL_OVERRIDE_HEADER, "").strip()
    override_metrics_port = request.headers.get(TRITON_METRICS_PORT_OVERRIDE_HEADER, "").strip()
    if not override_url and not override_metrics_port:
        return base_loader

    effective_triton_url = override_url or base_loader.config.triton_url
    effective_metrics_url = base_loader.config.triton_metrics_url
    if override_metrics_port:
        effective_metrics_url = _build_metrics_url_from_port(effective_triton_url, override_metrics_port)
    elif override_url:
        effective_metrics_url = None

    return TritonHotLoader(
        base_loader.config.with_updates(
            triton_url=effective_triton_url,
            triton_metrics_url=effective_metrics_url,
        )
    )


def create_app(loader: TritonHotLoader | None = None) -> FastAPI:
    project_root = Path(__file__).resolve().parent
    static_dir = project_root / "static"
    index_file = static_dir / "index.html"

    app = FastAPI(
        title="Triton Hot Loader Controller",
        description="Kubernetes Job based Triton model loading controller.",
        version="2.0.0",
    )
    app.state.loader = loader or TritonHotLoader(HotLoaderConfig.default())

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.exception_handler(HotLoaderError)
    async def hot_loader_exception_handler(_: Request, exc: HotLoaderError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"success": False, "detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(index_file)

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/runtime/health")
    async def runtime_health(request: Request) -> Dict[str, object]:
        loader = _get_request_loader(request)
        ready = loader.triton_ready()
        return {
            "status": "ok" if ready["ready"] else "degraded",
            "triton_ready": ready["ready"],
            "triton_url": loader.config.triton_url,
            "detail": ready["detail"],
        }

    def _runtime_gpu_status_payload(loader: TritonHotLoader) -> Dict[str, object]:
        metrics = loader.get_triton_gpu_metrics()
        return _format_runtime_gpu_status(metrics)

    def _load_model_sync(loader: TritonHotLoader, payload: ModelLoadRequest) -> Dict[str, object]:
        result = loader.create_model_copy_job_and_wait(payload.model_name, payload.image)
        if not result.get("success"):
            raise HotLoaderError(str(result.get("detail") or result.get("status") or "模型加载失败"))
        return result

    def _load_model_batch_sync(loader: TritonHotLoader, payload: ModelLoadBatchRequest) -> Dict[str, object]:
        if not payload.models:
            raise HotLoaderError("models 不能为空")
        result = loader.load_models_from_images_sync([item.model_dump(exclude_none=True) for item in payload.models])
        if not result.get("success"):
            error_messages = [
                str(item.get("error") or item.get("status") or "加载失败")
                for item in result.get("errors", [])
                if isinstance(item, dict)
            ]
            raise HotLoaderError("；".join(error_messages) or "批量加载失败")
        return result

    def _unload_payload(loader: TritonHotLoader, payload: UnloadRequest) -> Dict[str, object]:
        if not payload.aliases and not payload.models and not payload.versions:
            raise HotLoaderError("请至少选择一个 alias、model 或 model@version")

        alias_result = None
        model_result = None
        version_result = None
        success = True
        errors: List[Dict[str, object]] = []

        if payload.aliases:
            alias_result = loader.unload_aliases(payload.aliases)
            success = success and alias_result["success"]
            errors.extend(alias_result.get("errors", []))

        if payload.models:
            model_result = loader.unload_models(payload.models)
            success = success and model_result["success"]

        if payload.versions:
            version_result = loader.unload_model_versions(payload.versions)
            success = success and version_result["success"]
            errors.extend(version_result.get("errors", []))

        return {
            "success": success and not errors,
            "alias_result": alias_result,
            "model_result": model_result,
            "version_result": version_result,
            "errors": errors,
            "state": loader.get_managed_state(),
        }

    @app.get("/api/status")
    async def status(request: Request) -> Dict[str, object]:
        return _get_request_loader(request).get_status()

    @app.get("/api/models")
    async def api_models(request: Request) -> Dict[str, object]:
        return _get_request_loader(request).get_models_overview()

    @app.get("/api/state")
    async def api_state(request: Request) -> Dict[str, object]:
        return _get_request_loader(request).get_managed_state()

    @app.get("/api/gpu-metrics")
    async def api_gpu_metrics(request: Request) -> Dict[str, object]:
        return _get_request_loader(request).get_triton_gpu_metrics()

    @app.get("/api/gpu-status")
    async def api_gpu_status(request: Request) -> Dict[str, object]:
        return _runtime_gpu_status_payload(_get_request_loader(request))

    @app.get("/runtime/gpu-status")
    async def runtime_gpu_status(request: Request) -> Dict[str, object]:
        return _runtime_gpu_status_payload(_get_request_loader(request))

    @app.post("/api/models/load")
    async def api_load_model(payload: ModelLoadRequest, request: Request) -> Dict[str, object]:
        return _load_model_sync(_get_request_loader(request), payload)

    @app.post("/api/models/load-batch")
    async def api_load_model_batch(payload: ModelLoadBatchRequest, request: Request) -> Dict[str, object]:
        return _load_model_batch_sync(_get_request_loader(request), payload)

    @app.get("/api/jobs/{job_name}")
    async def api_job_status(job_name: str, request: Request) -> Dict[str, object]:
        return _get_request_loader(request).get_job_status(job_name)

    @app.post("/api/models/unload")
    async def api_unload_model(payload: ModelActionRequest, request: Request) -> Dict[str, object]:
        return _get_request_loader(request).unload_models([payload.model_name])

    @app.post("/api/models/reload")
    async def api_reload_model(payload: ModelActionRequest, request: Request) -> Dict[str, object]:
        return _get_request_loader(request).reload_models([payload.model_name])

    @app.post("/api/models/unload-batch")
    async def api_unload_batch(payload: UnloadRequest, request: Request) -> Dict[str, object]:
        return _unload_payload(_get_request_loader(request), payload)

    @app.post("/models/load")
    async def load_model(payload: ModelLoadRequest, request: Request) -> Dict[str, object]:
        return _load_model_sync(_get_request_loader(request), payload)

    @app.post("/models/load-batch")
    async def load_model_batch(payload: ModelLoadBatchRequest, request: Request) -> Dict[str, object]:
        return _load_model_batch_sync(_get_request_loader(request), payload)

    @app.get("/models/jobs/{job_name}")
    async def job_status(job_name: str, request: Request) -> Dict[str, object]:
        return _get_request_loader(request).get_job_status(job_name)

    @app.get("/models")
    async def models(request: Request) -> Dict[str, object]:
        return _get_request_loader(request).get_models_overview()

    @app.post("/models/unload")
    async def unload_model(payload: ModelActionRequest, request: Request) -> Dict[str, object]:
        return _get_request_loader(request).unload_models([payload.model_name])

    @app.post("/models/reload")
    async def reload_model(payload: ModelActionRequest, request: Request) -> Dict[str, object]:
        return _get_request_loader(request).reload_models([payload.model_name])

    @app.get("/metrics/gpu")
    async def gpu_metrics(request: Request) -> Dict[str, object]:
        loader = _get_request_loader(request)
        return {
            "gpu": _format_runtime_gpu_status(loader.get_triton_gpu_metrics()),
            "triton": {
                "url": loader.config.triton_url,
                "ready": loader.triton_ready()["ready"],
                "models": loader.list_repository_models(safe=True),
            },
            "manager": loader.get_managed_state(),
        }

    @app.post("/api/unload")
    async def unload_compat(payload: UnloadRequest, request: Request) -> Dict[str, object]:
        return _unload_payload(_get_request_loader(request), payload)

    @app.post("/api/reload")
    async def reload_compat(payload: ReloadRequest, request: Request) -> Dict[str, object]:
        return _get_request_loader(request).reload_models(payload.models)

    return app


def start_server(
    config: HotLoaderConfig | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8090,
) -> None:
    app = create_app(TritonHotLoader(config or HotLoaderConfig.default()))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
