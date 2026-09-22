"""Durable controller state machine for remove-version-and-reload operations."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

import version_transaction as files

TERMINAL = {"SUCCEEDED", "FAILED_RESTORED"}


class PendingIO(Exception):
    """An external operation may still be running; never restore speculatively."""


class VersionOperations:
    def __init__(self, loader):
        self.loader = loader

    def update(self, operation_id, **updates):
        with self.loader._state_lock:
            state = self.loader._load_state()
            operation = state["version_operations"][operation_id]
            operation.update(updates, updated_at=self.loader._utc_now())
            state["updated_at"] = operation["updated_at"]
            self.loader._save_state(state)
            return dict(operation)

    def phase(self, operation, phase, **updates):
        return self.update(operation["id"], phase=phase, status=phase, **updates)

    @staticmethod
    def public(operation):
        result = {k: v for k, v in operation.items() if k not in {"plans", "remote_results"}}
        result["backups"] = [p["backup"] for p in operation.get("plans", [])]
        result["success"] = operation["status"] == "SUCCEEDED"
        result["pending"] = operation["status"] not in TERMINAL
        return result

    def roots(self):
        loader = self.loader
        if loader._uses_repository_sync_mode():
            paths = [loader._job_repository_path(), loader.config.model_repository]
        elif loader._uses_job_only_repository():
            if not loader.config.repository_maintenance_image:
                from hot_loader import HotLoaderError
                raise HotLoaderError("Job-only 版本下线需要配置包含 /app/version_transaction.py 的 REPOSITORY_MAINTENANCE_IMAGE")
            return [{"path": loader.config.model_target_path, "remote": True}]
        else:
            paths = [loader.config.model_repository]
        return [{"path": str(path.absolute()), "remote": False} for path in dict.fromkeys(paths)]

    def submit(self, version_refs):
        from hot_loader import HotLoaderError, HotLoaderConflictError, _ACTIVE_JOB_STATUSES
        grouped = {}
        for ref in version_refs:
            model, version = self.loader._parse_model_version_ref(ref)
            self.loader._validate_model_name(model)
            if str(int(version)) != version:
                raise HotLoaderError("版本必须为规范数字目录名，不允许前导零")
            grouped.setdefault(model, set()).add(version)
        if not grouped:
            raise HotLoaderError("请至少提供一个 model_name@version")
        results, errors = [], []
        for model, versions in sorted(grouped.items()):
            try:
                with self.loader._version_submission_lock(model), self.loader._state_lock:
                    state = self.loader._load_state()
                    self.loader._assert_current_instance(state)
                    self.loader._assert_no_version_operation(model, state=state)
                    if any(job.get("model_name") == model and job.get("status") in _ACTIVE_JOB_STATUSES
                           for job in state["jobs"].values()):
                        raise HotLoaderConflictError(f"模型 {model} 有活跃加载任务，不能移除版本")
                    roots = self.roots()
                    operation_id = uuid.uuid4().hex
                    # Local preflight rejects invalid requests before recording an operation.
                    plans = []
                    for root in roots:
                        if root["remote"]:
                            continue
                        plans.append(files.inspect(root["path"], model, operation_id, sorted(versions, key=int)))
                    self.check_plans(plans)
                    operation = {"id": operation_id, "model_name": model, "versions": sorted(versions, key=int),
                        "instance_id": self.loader.instance_id, "triton_url": self.loader.config.triton_url,
                        "triton_metrics_url": self.loader.config.triton_metrics_url,
                        "created_at": self.loader._utc_now(), "updated_at": self.loader._utc_now(),
                        "status": "VALIDATING", "phase": "VALIDATING", "roots": roots,
                        "plans": plans, "remote_results": {}, "detail": "正在验证版本下线事务"}
                    state.setdefault("version_operations", {})[operation_id] = operation
                    self.loader._save_state(state)
                results.append(self.advance(operation_id))
            except HotLoaderConflictError:
                if len(grouped) == 1:
                    raise
                errors.append({"model_name": model, "error": "模型存在活跃操作", "conflict": True})
            except (HotLoaderError, OSError, ValueError) as exc:
                if len(grouped) == 1:
                    raise HotLoaderError(str(exc)) from exc
                errors.append({"model_name": model, "error": str(exc)})
        return {"success": not errors and all(r["success"] for r in results),
                "pending": any(r["pending"] for r in results), "operations": results,
                "errors": errors, "state": self.loader.get_managed_state()}

    @staticmethod
    def check_plans(plans):
        if plans and any(p["remaining"] != plans[0]["remaining"] for p in plans):
            raise ValueError("PVC 与在线仓库版本集合不一致，尚未修改文件")

    def remote(self, operation, action, index, request):
        key = f"{action}-{index}"
        if key in operation["remote_results"]:
            return operation["remote_results"][key]
        loader = self.loader
        job_name = f"version-{operation['id']}-{key}"
        try:
            batch = loader._get_batch_v1_api()
        except Exception as exc:
            raise PendingIO(f"无法确认维护 Job 状态: {exc}") from exc
        try:
            job = batch.read_namespaced_job(name=job_name, namespace=loader.config.k8s_namespace)
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise PendingIO(f"维护 Job 查询结果未知: {exc}") from exc
            manifest = loader._build_repository_cleanup_job_manifest(job_name, operation["model_name"])
            manifest["spec"]["backoffLimit"] = 0
            manifest["spec"]["ttlSecondsAfterFinished"] = 86400
            container = manifest["spec"]["template"]["spec"]["containers"][0]
            container["command"] = ["python", "/app/version_transaction.py"]
            container["args"] = []
            container["env"] = [{"name": "VERSION_TRANSACTION_REQUEST", "value": json.dumps({"protocol": files.PROTOCOL, **request})}]
            manifest["metadata"]["annotations"]["hot-loader/version-operation"] = operation["id"]
            try:
                batch.create_namespaced_job(namespace=loader.config.k8s_namespace, body=manifest)
            except Exception as create_exc:
                raise PendingIO(f"维护 Job 创建结果待确认: {create_exc}") from create_exc
            raise PendingIO(f"等待维护 Job {job_name}")
        status = getattr(job, "status", None)
        if not (getattr(status, "succeeded", 0) or getattr(status, "failed", 0)):
            raise PendingIO(f"维护 Job {job_name} 仍在执行")
        pods = loader._list_job_pods(job_name)
        pod_name = getattr(getattr(pods[0], "metadata", None), "name", None) if pods else None
        logs = loader._read_pod_logs(pod_name) or ""
        parsed = None
        for line in logs.splitlines():
            if line.startswith(files.RESULT_PREFIX):
                parsed = json.loads(line[len(files.RESULT_PREFIX):])
        if getattr(status, "failed", 0):
            raise ValueError((parsed or {}).get("error") or f"维护 Job 失败（确认镜像包含事务脚本）: {logs}")
        if not parsed or parsed.get("protocol") != files.PROTOCOL:
            raise PendingIO("维护 Job 已结束但缺少协议结果，保留备份等待确认")
        if not parsed.get("success"):
            raise ValueError(parsed.get("error", "维护 Job 失败"))
        operation["remote_results"][key] = parsed["result"]
        self.update(operation["id"], remote_results=operation["remote_results"])
        return parsed["result"]

    def action(self, operation, action):
        for index, (root, plan) in enumerate(zip(operation["roots"], operation["plans"])):
            if root["remote"]:
                self.remote(operation, action, index, {"action": action, "plan": plan})
            else:
                files.execute(action, plan)

    def verify(self, operation):
        rows = self.loader.list_repository_models()
        model_rows = [row for row in rows if row.get("name") == operation["model_name"]]
        ready = {str(row.get("version")) for row in model_rows if str(row.get("state", "")).upper() == "READY"}
        transitioning = any(str(row.get("state", "")).upper() in {"LOADING", "UNLOADING"} for row in model_rows)
        return not transitioning and set(operation["remaining_versions"]) <= ready and not ready.intersection(operation["versions"])

    def advance(self, operation_id):
        from hot_loader import HotLoaderError
        operation = self.loader._load_state().get("version_operations", {}).get(operation_id)
        if not operation or not self.loader._owns_job(operation):
            raise HotLoaderError("版本操作不存在或不属于当前实例")
        if operation["status"] in TERMINAL:
            return self.public(operation)
        if operation["triton_url"] != self.loader.config.triton_url:
            raise HotLoaderError("实例地址与版本操作记录不一致")
        with self.loader._model_operation_lock(operation["model_name"]):
            operation = self.loader._load_state()["version_operations"][operation_id]
            if operation["status"] in TERMINAL:
                return self.public(operation)
            # A REQUESTING record may have been persisted just before a crash.
            # The HTTP request may have reached Triton: never send it twice.
            if operation["phase"] == "REQUESTING":
                operation = self.phase(operation, "VERIFYING", detail="重载结果未知，继续确认 Triton 状态")
            for _ in range(8):
                phase = operation["phase"]
                try:
                    if phase == "VALIDATING":
                        rows = self.loader.list_repository_models()
                        if any(row.get("name") == operation["model_name"] and row.get("state") in {"LOADING", "UNLOADING"} for row in rows):
                            raise ValueError("Triton 模型正在加载或卸载，尚未修改文件")
                        plans = list(operation["plans"])
                        for index, root in enumerate(operation["roots"]):
                            if root["remote"] and len(plans) <= index:
                                plans.append(self.remote(operation, "inspect", index, {"action": "inspect", "arguments": {
                                    "root": root["path"], "model": operation["model_name"], "operation_id": operation_id,
                                    "versions": operation["versions"]}}))
                        self.check_plans(plans)
                        operation = self.phase(operation, "PREPARING", plans=plans, remaining_versions=plans[0]["remaining"], detail="已验证，正在备份并移出版本")
                    elif phase == "PREPARING":
                        self.action(operation, "prepare")
                        operation = self.phase(operation, "REQUESTING", detail="正在请求当前实例重载")
                        # Bypass the generic wrapper to distinguish a definitive
                        # Triton rejection from a lost response or proxy timeout.
                        try:
                            response = httpx.request("POST", f"{operation['triton_url']}/v2/repository/models/{quote(operation['model_name'], safe='')}/load",
                                json={}, timeout=self.loader.config.request_timeout)
                        except httpx.HTTPError as exc:
                            operation = self.phase(operation, "VERIFYING", detail=f"重载响应未知，保留备份: {exc}")
                            return self.public(operation)
                        try:
                            rejected = response.is_error and isinstance(response.json().get("error"), str)
                        except (ValueError, AttributeError):
                            rejected = False
                        # 408/502/503/504 can originate at a proxy; not proof that
                        # Triton stopped loading. Only explicit Triton errors restore.
                        if rejected and response.status_code not in {408, 502, 503, 504}:
                            operation = self.phase(operation, "RESTORING", error=response.text, detail="Triton 明确拒绝重载，正在恢复文件")
                        else:
                            operation = self.phase(operation, "VERIFYING", detail="等待所有剩余版本 READY、移除版本退出服务")
                    elif phase == "VERIFYING":
                        if not self.verify(operation):
                            raise PendingIO("剩余版本尚未全部 READY，或移除版本仍在服务；保留备份")
                        operation = self.phase(operation, "COMMITTING", detail="切换已验证，更新元数据并清理备份")
                    elif phase == "COMMITTING":
                        with self.loader._state_lock:
                            state = self.loader._load_state()
                            for meta in state["aliases"].values():
                                if operation["model_name"] in meta.get("models", []):
                                    meta.setdefault("model_versions", {})[operation["model_name"]] = operation["remaining_versions"]
                                    meta["updated_at"] = self.loader._utc_now()
                            self.loader._save_state(state)
                        self.action(operation, "commit")
                        operation = self.phase(operation, "SUCCEEDED", detail="版本已移除，当前实例重载成功，备份已清理")
                    elif phase == "RESTORING":
                        self.action(operation, "restore")
                        operation = self.phase(operation, "FAILED_RESTORED", detail="操作失败，版本目录与原始配置已恢复")
                    else:
                        return self.public(operation)
                except PendingIO as exc:
                    operation = self.pending(operation, str(exc))
                    break
                except Exception as exc:
                    if phase == "VALIDATING":
                        operation = self.phase(operation, "FAILED_RESTORED", error=str(exc), detail="验证失败，未修改仓库文件")
                    elif phase == "PREPARING" and operation["phase"] == "PREPARING":
                        operation = self.phase(operation, "RESTORING", error=str(exc), detail="文件操作失败，正在恢复")
                        continue
                    else:
                        operation = self.pending(operation, str(exc), recovery=phase in {"RESTORING", "COMMITTING"})
                    break
            return self.public(operation)

    def pending(self, operation, detail, *, recovery=False):
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(operation["created_at"])).total_seconds()
        if recovery or elapsed >= self.loader.config.triton_reload_timeout_seconds:
            return self.update(operation["id"], status="RECOVERY_REQUIRED", detail=f"需恢复确认：{detail}")
        return self.update(operation["id"], detail=detail)

    def refresh(self):
        operations = self.loader._load_state().get("version_operations", {})
        results = []
        for operation_id, operation in operations.items():
            if self.loader._owns_job(operation) and operation["status"] not in TERMINAL:
                try:
                    results.append(self.advance(operation_id))
                except Exception as exc:
                    results.append(self.public(self.pending(operation, str(exc), recovery=True)))
        return results
