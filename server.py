from __future__ import annotations

import hmac
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlsplit, urlunsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hot_loader import HotLoaderConfig, HotLoaderError, TritonHotLoader


TRITON_URL_OVERRIDE_HEADER = "x-hot-triton-url"
TRITON_METRICS_PORT_OVERRIDE_HEADER = "x-hot-triton-metrics-port"
_CALLBACK_WATCH_INTERVAL_SECONDS = 2.0
_CALLBACK_RETRY_BASE_SECONDS = 5.0
_CALLBACK_RETRY_MAX_SECONDS = 60.0


class CallbackRequest(BaseModel):
    url: str = Field(min_length=1)
    events: List[str] = Field(default_factory=lambda: ["terminal"])
    token: str | None = Field(default=None)


class ModelImageRequest(BaseModel):
    image: str = Field(min_length=1)
    model_name: str | None = Field(default=None)


class ModelLoadRequest(ModelImageRequest):
    wait_for_ready: bool = Field(default=False)
    callback: CallbackRequest | None = Field(default=None)


class ModelLoadBatchRequest(BaseModel):
    models: List[ModelImageRequest] = Field(default_factory=list)
    wait_for_ready: bool = Field(default=False)
    callback: CallbackRequest | None = Field(default=None)


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


def _callback_retry_delay_seconds(attempts: int) -> float:
    exponent = max(attempts - 1, 0)
    return min(_CALLBACK_RETRY_MAX_SECONDS, _CALLBACK_RETRY_BASE_SECONDS * (2 ** exponent))


def _callback_payload_dict(callback: CallbackRequest | None) -> Dict[str, object] | None:
    return callback.model_dump(exclude_none=True) if callback else None


def _callback_event_body(job: Dict[str, object], *, event_id: str, attempt: int) -> bytes:
    payload = {
        "event_id": event_id,
        "event_type": "job.status.changed",
        "job_name": job.get("job_name"),
        "model_name": job.get("model_name"),
        "image": job.get("image"),
        "status": job.get("status"),
        "detail": job.get("detail"),
        "terminal": True,
        "triton_ready": job.get("triton_ready"),
        "updated_at": job.get("updated_at"),
        "callback_attempt": attempt,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _deliver_terminal_callback(loader: TritonHotLoader, job: Dict[str, object]) -> None:
    callback = job.get("callback")
    if not isinstance(callback, dict):
        return

    job_name = str(job.get("job_name") or "")
    if not job_name:
        return

    event_id = str(callback.get("last_event_id") or uuid.uuid4())
    attempt = int(callback.get("attempts") or 0) + 1
    body = _callback_event_body(job, event_id=event_id, attempt=attempt)
    timestamp = str(int(time.time()))
    headers = {
        "Content-Type": "application/json",
        "X-Hot-Loader-Event": "job.status.changed",
        "X-Hot-Loader-Job-Name": job_name,
        "X-Hot-Loader-Timestamp": timestamp,
    }

    token = str(callback.get("token") or "").strip()
    if token:
        signature = hmac.new(token.encode("utf-8"), f"{timestamp}.".encode("utf-8") + body, "sha256").hexdigest()
        headers["X-Hot-Loader-Signature"] = f"sha256={signature}"

    try:
        response = httpx.post(str(callback["url"]), content=body, headers=headers, timeout=loader.config.request_timeout)
        if response.is_error:
            detail = response.text.strip() or f"HTTP {response.status_code}"
            raise HotLoaderError(f"callback 返回非 2xx: {detail}")
        loader.record_terminal_callback_result(job_name, delivered=True, event_id=event_id)
    except Exception as exc:
        loader.record_terminal_callback_result(
            job_name,
            delivered=False,
            event_id=event_id,
            error=str(exc),
            retry_delay_seconds=_callback_retry_delay_seconds(attempt),
        )


def _background_watch_loop(loader: TritonHotLoader, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            loader.refresh_active_job_statuses(include_logs=False)
            for job in loader.list_pending_terminal_callbacks():
                _deliver_terminal_callback(loader, job)
        except Exception:
            pass
        stop_event.wait(_CALLBACK_WATCH_INTERVAL_SECONDS)


def create_app(loader: TritonHotLoader | None = None, *, enable_background_worker: bool = True) -> FastAPI:
    project_root = Path(__file__).resolve().parent
    static_dir = project_root / "static"
    index_file = static_dir / "index.html"

    app = FastAPI(
        title="Triton Hot Loader Controller",
        description="Kubernetes Job based Triton model loading controller.",
        version="2.0.0",
    )
    app.state.loader = loader or TritonHotLoader(HotLoaderConfig.default())

    if enable_background_worker:
        worker_stop_event = threading.Event()

        @app.on_event("startup")
        async def start_background_worker() -> None:
            if getattr(app.state, "callback_worker", None):
                return
            worker_stop_event.clear()
            worker = threading.Thread(
                target=_background_watch_loop,
                args=(app.state.loader, worker_stop_event),
                daemon=True,
                name="hot-loader-callback-worker",
            )
            app.state.callback_worker = worker
            app.state.callback_worker_stop_event = worker_stop_event
            worker.start()

        @app.on_event("shutdown")
        async def stop_background_worker() -> None:
            worker_stop_event.set()
            worker = getattr(app.state, "callback_worker", None)
            if isinstance(worker, threading.Thread):
                worker.join(timeout=5.0)
            app.state.callback_worker = None

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
        ready = await run_in_threadpool(loader.triton_ready)
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
        callback = _callback_payload_dict(payload.callback)
        kwargs = {"callback": callback} if callback else {}
        result = loader.create_model_copy_job_and_wait(payload.model_name, payload.image, **kwargs)
        if not result.get("success"):
            raise HotLoaderError(str(result.get("detail") or result.get("status") or "模型加载失败"))
        return result

    def _load_model_async(loader: TritonHotLoader, payload: ModelLoadRequest) -> Dict[str, object]:
        callback = _callback_payload_dict(payload.callback)
        kwargs = {"callback": callback} if callback else {}
        return loader.create_model_copy_job(payload.model_name, payload.image, **kwargs)

    def _load_model_batch_sync(loader: TritonHotLoader, payload: ModelLoadBatchRequest) -> Dict[str, object]:
        if not payload.models:
            raise HotLoaderError("models 不能为空")
        callback = _callback_payload_dict(payload.callback)
        kwargs = {"callback": callback} if callback else {}
        result = loader.load_models_from_images_sync([item.model_dump(exclude_none=True) for item in payload.models], **kwargs)
        if not result.get("success"):
            error_messages = [
                str(item.get("error") or item.get("status") or "加载失败")
                for item in result.get("errors", [])
                if isinstance(item, dict)
            ]
            raise HotLoaderError("；".join(error_messages) or "批量加载失败")
        return result

    def _load_model_batch_async(loader: TritonHotLoader, payload: ModelLoadBatchRequest) -> Dict[str, object]:
        if not payload.models:
            raise HotLoaderError("models 不能为空")
        callback = _callback_payload_dict(payload.callback)
        kwargs = {"callback": callback} if callback else {}
        return loader.load_models_from_images([item.model_dump(exclude_none=True) for item in payload.models], **kwargs)

    def _unload_payload(loader: TritonHotLoader, payload: UnloadRequest) -> Dict[str, object]:
        if not payload.aliases and not payload.models and not payload.versions:
            raise HotLoaderError("请至少选择一个 alias 或 model")
        if payload.versions:
            raise HotLoaderError("同名模型已取消版本管理，请按 model_name 或 alias 卸载")

        alias_result = None
        model_result = None
        success = True
        errors: List[Dict[str, object]] = []

        if payload.aliases:
            alias_result = loader.unload_aliases(payload.aliases)
            success = success and alias_result["success"]
            errors.extend(alias_result.get("errors", []))

        if payload.models:
            model_result = loader.unload_models(payload.models)
            success = success and model_result["success"]

        return {
            "success": success and not errors,
            "alias_result": alias_result,
            "model_result": model_result,
            "errors": errors,
            "state": loader.get_managed_state(),
        }

    @app.get("/api/status")
    async def status(request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).get_status)

    @app.get("/api/models")
    async def api_models(request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).get_models_overview)

    @app.get("/api/state")
    async def api_state(request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).get_managed_state)

    @app.get("/api/gpu-metrics")
    async def api_gpu_metrics(request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).get_triton_gpu_metrics)

    @app.get("/api/gpu-status")
    async def api_gpu_status(request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_runtime_gpu_status_payload, _get_request_loader(request))

    @app.get("/runtime/gpu-status")
    async def runtime_gpu_status(request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_runtime_gpu_status_payload, _get_request_loader(request))

    @app.post("/api/models/load")
    async def api_load_model(payload: ModelLoadRequest, request: Request) -> Dict[str, object]:
        handler = _load_model_sync if payload.wait_for_ready else _load_model_async
        return await run_in_threadpool(handler, _get_request_loader(request), payload)

    @app.post("/api/models/load-batch")
    async def api_load_model_batch(payload: ModelLoadBatchRequest, request: Request) -> Dict[str, object]:
        handler = _load_model_batch_sync if payload.wait_for_ready else _load_model_batch_async
        return await run_in_threadpool(handler, _get_request_loader(request), payload)

    @app.get("/api/jobs/{job_name}")
    async def api_job_status(job_name: str, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).get_job_status, job_name)

    @app.post("/api/models/unload")
    async def api_unload_model(payload: ModelActionRequest, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).unload_models, [payload.model_name])

    @app.post("/api/models/reload")
    async def api_reload_model(payload: ModelActionRequest, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).reload_models, [payload.model_name])

    @app.post("/api/models/unload-batch")
    async def api_unload_batch(payload: UnloadRequest, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_unload_payload, _get_request_loader(request), payload)

    @app.post("/models/load")
    async def load_model(payload: ModelLoadRequest, request: Request) -> Dict[str, object]:
        handler = _load_model_sync if payload.wait_for_ready else _load_model_async
        return await run_in_threadpool(handler, _get_request_loader(request), payload)

    @app.post("/models/load-batch")
    async def load_model_batch(payload: ModelLoadBatchRequest, request: Request) -> Dict[str, object]:
        handler = _load_model_batch_sync if payload.wait_for_ready else _load_model_batch_async
        return await run_in_threadpool(handler, _get_request_loader(request), payload)

    @app.get("/models/jobs/{job_name}")
    async def job_status(job_name: str, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).get_job_status, job_name)

    @app.get("/models")
    async def models(request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).get_models_overview)

    @app.post("/models/unload")
    async def unload_model(payload: ModelActionRequest, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).unload_models, [payload.model_name])

    @app.post("/models/reload")
    async def reload_model(payload: ModelActionRequest, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).reload_models, [payload.model_name])

    @app.get("/metrics/gpu")
    async def gpu_metrics(request: Request) -> Dict[str, object]:
        loader = _get_request_loader(request)
        gpu_payload, triton_ready_payload, triton_models_payload, manager_payload = await run_in_threadpool(
            lambda: (
                _format_runtime_gpu_status(loader.get_triton_gpu_metrics()),
                loader.triton_ready(),
                loader.list_repository_models(safe=True),
                loader.get_managed_state(),
            )
        )
        return {
            "gpu": gpu_payload,
            "triton": {
                "url": loader.config.triton_url,
                "ready": triton_ready_payload["ready"],
                "models": triton_models_payload,
            },
            "manager": manager_payload,
        }

    @app.post("/api/unload")
    async def unload_compat(payload: UnloadRequest, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_unload_payload, _get_request_loader(request), payload)

    @app.post("/api/reload")
    async def reload_compat(payload: ReloadRequest, request: Request) -> Dict[str, object]:
        return await run_in_threadpool(_get_request_loader(request).reload_models, payload.models)

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
