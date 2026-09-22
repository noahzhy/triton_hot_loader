from __future__ import annotations

import json
import io
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from hot_loader import HotLoaderConfig, HotLoaderError, HotLoaderConflictError, TritonHotLoader
from server import create_app, _watch_instance
from cli import build_parser, execute
from tests.test_hot_loader import write_model_bundle
from version_operations import VersionOperations
import version_transaction as files


class SimulatedCrash(BaseException):
    pass


class VersionOperationTests(unittest.TestCase):
    def test_concurrent_requests_reject_while_load_http_is_in_flight(self):
        entered, release = threading.Event(), threading.Event()
        original_http = self.http

        def slow_http(method, url, **kwargs):
            if url.endswith('/load'):
                entered.set()
                if not release.wait(5):
                    raise AssertionError('test did not release reload')
            return original_http(method, url, **kwargs)

        with patch('httpx.request', side_effect=slow_http), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.loader.unload_model_versions, ['demo@2'])
            try:
                self.assertTrue(entered.wait(2))
                for action in (
                    lambda: self.loader.unload_model_versions(['demo@2']),
                    lambda: self.loader.reload_models(['demo']),
                    lambda: self.loader.unload_models(['demo']),
                    lambda: self.loader.create_model_copy_job('demo', 'ccr.ccs.tencentyun.com/clobotics/demo:3'),
                ):
                    with self.assertRaises(HotLoaderConflictError):
                        action()
            finally:
                release.set()
            self.assertTrue(future.result()['success'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = HotLoaderConfig(triton_url="http://127.0.0.1:18000",
            model_repository=root / "models", state_file=root / "state.json",
            staging_root=root / "staging", model_target_path=str(root / "models"))
        self.loader = TritonHotLoader(self.config)
        self.model_dir = write_model_bundle(self.config.model_repository, "demo", ["1", "2"], include_version_policy=True)
        self.original = (self.model_dir / "config.pbtxt").read_text()
        self.loader._register_loaded_model("demo", "ccr.ccs.tencentyun.com/clobotics/demo:2")
        self.rows = [{"name": "demo", "version": v, "state": "READY"} for v in ("1", "2")]
        self.calls = []
        self.load_mode = "success"
        self.patcher = patch("httpx.request", side_effect=self.http)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def http(self, method, url, **kwargs):
        self.calls.append((method, url))
        if url.endswith("/load"):
            if self.load_mode == "crash":
                raise SimulatedCrash()
            if self.load_mode == "timeout":
                raise httpx.ReadTimeout("response lost")
            if self.load_mode == "reject":
                return httpx.Response(400, json={"error": "invalid model"})
            if self.load_mode == "proxy":
                return httpx.Response(504, json={"error": "gateway timeout"})
            if self.load_mode == "success":
                self.rows = [{"name": "demo", "version": p.name, "state": "READY"}
                             for p in self.model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
            return httpx.Response(200, json={})
        if url.endswith("/index"):
            return httpx.Response(200, json=self.rows)
        return httpx.Response(200, json={})

    def operation(self):
        return next(iter(self.loader._load_state()["version_operations"].values()))

    def test_remove_and_reload_commits_only_after_ready_without_unload(self):
        result = self.loader.unload_model_versions(["demo@2", "demo@2"])
        self.assertTrue(result["success"])
        self.assertFalse(result["pending"])
        self.assertFalse((self.model_dir / "2").exists())
        self.assertTrue((self.model_dir / "1").exists())
        self.assertEqual(self.loader._read_specific_version_policy(self.model_dir), ["1"])
        self.assertFalse(Path(result["operations"][0]["backups"][0]).exists())
        self.assertEqual(self.loader.get_managed_state()["managed_model_versions"]["demo"], ["1"])
        self.assertEqual(sum(url.endswith("/load") for _, url in self.calls), 1)
        self.assertFalse(any(url.endswith("/unload") for _, url in self.calls))

    def test_reject_last_version_paths_links_and_missing_version_without_mutation(self):
        for refs in [["demo@1", "demo@2"], ["../demo@2"], ["demo@3"], ["demo@02"], []]:
            with self.subTest(refs=refs), self.assertRaises(HotLoaderError):
                self.loader.unload_model_versions(refs)
        (self.model_dir / "2" / "escape").symlink_to(self.config.state_file)
        with self.assertRaisesRegex(HotLoaderError, "符号链接"):
            self.loader.unload_model_versions(["demo@2"])
        self.assertEqual((self.model_dir / "config.pbtxt").read_text(), self.original)
        self.assertFalse(self.calls)

    def test_definitive_triton_failure_restores_original_files_and_config(self):
        self.load_mode = "reject"
        result = self.loader.unload_model_versions(["demo@2"])
        self.assertFalse(result["success"])
        self.assertFalse(result["pending"])
        self.assertEqual(result["operations"][0]["status"], "FAILED_RESTORED")
        self.assertTrue((self.model_dir / "2" / "model.onnx").exists())
        self.assertEqual((self.model_dir / "config.pbtxt").read_text(), self.original)
        self.assertEqual(self.loader.get_managed_state()["managed_model_versions"]["demo"], ["1", "2"])

    def test_partial_file_failure_restores_without_calling_load(self):
        original_write = files.atomic_write
        def fail_new_config(path, text):
            if path == self.model_dir / "config.pbtxt" and text != self.original:
                raise OSError("disk full")
            return original_write(path, text)
        with patch.object(files, "atomic_write", side_effect=fail_new_config):
            result = self.loader.unload_model_versions(["demo@2"])
        self.assertEqual(result["operations"][0]["status"], "FAILED_RESTORED")
        self.assertTrue((self.model_dir / "2").exists())
        self.assertEqual((self.model_dir / "config.pbtxt").read_text(), self.original)
        self.assertFalse(any(url.endswith("/load") for _, url in self.calls))

    def test_timeout_preserves_backup_and_global_locks_then_restart_finishes(self):
        self.load_mode = "timeout"
        result = self.loader.unload_model_versions(["demo@2"])
        operation = result["operations"][0]
        self.assertTrue(result["pending"])
        self.assertTrue(Path(operation["backups"][0]).is_dir())
        for action in [lambda: self.loader.reload_models(["demo"]), lambda: self.loader.unload_models(["demo"]),
                       lambda: self.loader.create_model_copy_job("demo", "ccr.ccs.tencentyun.com/clobotics/demo:3"),
                       lambda: self.loader.unload_model_versions(["demo@1"]),
                       lambda: self.loader.save_instance("changed", "10.0.0.8", instance_id="default")]:
            with self.assertRaises(HotLoaderConflictError):
                action()
        VersionOperations(self.loader).update(operation["id"], created_at="2000-01-01T00:00:00+00:00")
        restarted = TritonHotLoader(self.config)
        pending = restarted.get_version_operation(operation["id"])
        self.assertEqual(pending["status"], "RECOVERY_REQUIRED")
        self.rows = [{"name": "demo", "version": "1", "state": "READY"}]
        self.assertEqual(restarted.get_version_operation(operation["id"])["status"], "SUCCEEDED")
        self.assertEqual(sum(url.endswith("/load") for _, url in self.calls), 1)

    def test_success_response_is_not_enough_requires_all_remaining_versions(self):
        write_model_bundle(self.config.model_repository, "demo", ["3"])
        self.load_mode = "nochange"
        result = self.loader.unload_model_versions(["demo@2"])
        self.assertTrue(result["pending"])
        op_id = result["operations"][0]["id"]
        self.rows = [{"name": "demo", "version": "1", "state": "READY"}]
        self.assertTrue(self.loader.get_version_operation(op_id)["pending"])
        self.rows.append({"name": "demo", "version": "3", "state": "READY"})
        self.assertTrue(self.loader.get_version_operation(op_id)["success"])

    def test_proxy_error_does_not_restore_during_possible_load(self):
        self.load_mode = "proxy"
        result = self.loader.unload_model_versions(["demo@2"])
        self.assertTrue(result["pending"])
        self.assertFalse((self.model_dir / "2").exists())
        self.assertTrue(Path(result["operations"][0]["backups"][0]).exists())

    def test_crash_at_load_does_not_resubmit_and_query_is_instance_scoped(self):
        self.load_mode = "crash"
        with self.assertRaises(SimulatedCrash):
            self.loader.unload_model_versions(["demo@2"])
        operation = self.operation()
        self.assertEqual(operation["phase"], "REQUESTING")
        b = self.loader.save_instance("B", "10.0.0.2")
        other = self.loader.for_instance(b["id"])
        with self.assertRaises(HotLoaderError):
            other.get_version_operation(operation["id"])
        with self.assertRaises(HotLoaderConflictError):
            other.unload_model_versions(["demo@1"])
        self.rows = [{"name": "demo", "version": "1", "state": "READY"}]
        _watch_instance(TritonHotLoader(self.config))
        self.assertEqual(self.operation()["status"], "SUCCEEDED")
        self.assertEqual(sum(url.endswith("/load") for _, url in self.calls), 1)

    def test_restart_during_prepare_restore_and_cleanup_is_idempotent(self):
        original_execute = files.execute
        for phase in ["prepare", "restore", "commit"]:
            with self.subTest(phase=phase):
                write_model_bundle(self.config.model_repository, "demo", ["1", "2"], include_version_policy=True)
                self.load_mode = "reject" if phase == "restore" else "success"
                crashed = False
                def crash_after_action(action, plan):
                    nonlocal crashed
                    result = original_execute(action, plan)
                    if action == phase and not crashed:
                        crashed = True
                        raise SimulatedCrash()
                    return result
                with patch.object(files, "execute", side_effect=crash_after_action), self.assertRaises(SimulatedCrash):
                    self.loader.unload_model_versions(["demo@2"])
                pending = [op for op in self.loader._load_state()["version_operations"].values()
                           if op["status"] not in {"SUCCEEDED", "FAILED_RESTORED"}][0]
                restarted = TritonHotLoader(self.config)
                result = restarted.get_version_operation(pending["id"])
                self.assertEqual(result["status"], "FAILED_RESTORED" if phase == "restore" else "SUCCEEDED")

    def test_group_versions_and_sync_pvc_with_runtime(self):
        source = Path(self.temp.name) / "pvc" / "models"
        write_model_bundle(source, "demo", ["1", "2", "3"])
        write_model_bundle(self.config.model_repository, "demo", ["3"])
        self.loader.config = self.config.with_updates(model_target_path=str(source))
        result = self.loader.unload_model_versions(["demo@2", "demo@3"])
        self.assertTrue(result["success"])
        for model in [source / "demo", self.model_dir]:
            self.assertTrue((model / "1").exists())
            self.assertFalse((model / "2").exists())
            self.assertFalse((model / "3").exists())
            self.assertEqual(self.loader._read_specific_version_policy(model), ["1"])
        self.assertEqual(sum(url.endswith("/load") for _, url in self.calls), 1)

    def test_api_accepts_versions_and_rejects_mixed_requests(self):
        with TestClient(create_app(self.loader, enable_background_worker=False)) as client:
            for route in ["/api/unload", "/api/models/unload-batch"]:
                mixed = client.post(route, json={"versions": ["demo@2"], "models": ["demo"]})
                self.assertEqual(mixed.status_code, 400)
            response = client.post("/api/models/unload-batch", json={"versions": ["demo@2"]})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["success"])
            op_id = response.json()["operations"][0]["id"]
            state = client.get(f"/api/version-operations/{op_id}").json()
            self.assertEqual(state["status"], "SUCCEEDED")
            self.assertNotIn("plans", state)

    def test_cli_version_removal_and_cross_model_partial_result(self):
        args = build_parser().parse_args(["unload", "--versions", "demo@2"])
        with patch("cli.build_config_from_args", return_value=self.config), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(execute(args), 0)
            self.assertTrue(json.loads(output.getvalue())["success"])
        write_model_bundle(self.config.model_repository, "demo", ["2"])
        result = self.loader.unload_model_versions(["demo@2", "missing@2"])
        self.assertFalse(result["success"])
        self.assertEqual(result["operations"][0]["status"], "SUCCEEDED")
        self.assertEqual(result["errors"][0]["model_name"], "missing")

    def test_policy_parser_preserves_other_fields_and_comments(self):
        config = 'name: "demo" version_policy { specific { versions: [2] } }\n# version_policy: {\nparameters { key: "version_policy" value { string_value: "}" } }\n'
        updated = files.version_policy(config, ["1"])
        self.assertIn('name: "demo"', updated)
        self.assertIn('string_value: "}"', updated)
        self.assertNotIn('versions: [2]', updated)
        self.assertIn('versions: [ 1 ]', updated)


class LocalMaintenanceJobs:
    """Execute exactly the emitted helper command in a temporary PVC directory."""
    def __init__(self, core):
        self.core = core
        self.jobs = {}
        self.manifests = []
        self.missing_helper = False

    def read_namespaced_job(self, name, namespace):
        if name not in self.jobs:
            error = RuntimeError("not found")
            error.status = 404
            raise error
        self.core.current = name
        return self.jobs[name]

    def create_namespaced_job(self, namespace, body):
        self.manifests.append(body)
        name = body["metadata"]["name"]
        container = body["spec"]["template"]["spec"]["containers"][0]
        env = {**os.environ, **{item["name"]: item["value"] for item in container["env"]}}
        if self.missing_helper:
            result = SimpleNamespace(returncode=2, stdout="python: cannot open version_transaction.py")
        else:
            result = subprocess.run([sys.executable, str(Path(files.__file__))], env=env, capture_output=True, text=True)
        self.core.logs[name] = result.stdout
        self.jobs[name] = SimpleNamespace(status=SimpleNamespace(succeeded=int(result.returncode == 0), failed=int(result.returncode != 0)))


class MaintenanceCore:
    def __init__(self):
        self.logs = {}
        self.current = None

    def list_namespaced_pod(self, **kwargs):
        return SimpleNamespace(items=[SimpleNamespace(metadata=SimpleNamespace(name=self.current))])

    def read_namespaced_pod_log(self, name, **kwargs):
        return self.logs[name]


class RemoteVersionOperationTests(unittest.TestCase):
    setUp = VersionOperationTests.setUp
    http = VersionOperationTests.http
    def test_remote_prepare_commit_and_missing_capability(self):
        core = MaintenanceCore()
        batch = LocalMaintenanceJobs(core)
        self.loader._batch_v1_api, self.loader._core_v1_api = batch, core
        self.loader.config = self.config.with_updates(repository_maintenance_image="controller:with-version-helper")
        with patch.object(self.loader, "_uses_job_only_repository", return_value=True), patch.object(self.loader, "_uses_repository_sync_mode", return_value=False):
            result = self.loader.unload_model_versions(["demo@2"])
            operation_id = result["operations"][0]["id"]
            self.assertTrue(result["pending"])
            for _ in range(12):
                op = self.loader.get_version_operation(operation_id)
                if not op["pending"]:
                    break
            self.assertEqual(op["status"], "SUCCEEDED")
            self.assertFalse((self.model_dir / "2").exists())
            self.assertEqual(len(batch.manifests), 3)  # inspect, prepare, commit
            self.assertTrue(all(m["spec"]["backoffLimit"] == 0 for m in batch.manifests))
            write_model_bundle(self.config.model_repository, "demo", ["2"])
            batch.missing_helper = True
            result = self.loader.unload_model_versions(["demo@2"])
            operation_id = result["operations"][0]["id"]
            self.assertEqual(self.loader.get_version_operation(operation_id)["status"], "FAILED_RESTORED")
            self.assertTrue((self.model_dir / "2").exists())

    def test_remote_explicit_failure_restores_via_helper_after_restart(self):
        core = MaintenanceCore()
        batch = LocalMaintenanceJobs(core)
        self.loader._batch_v1_api, self.loader._core_v1_api = batch, core
        self.loader.config = self.config.with_updates(repository_maintenance_image="controller:with-version-helper")
        self.load_mode = "reject"
        with patch.object(TritonHotLoader, "_uses_job_only_repository", return_value=True), patch.object(TritonHotLoader, "_uses_repository_sync_mode", return_value=False):
            result = self.loader.unload_model_versions(["demo@2"])
            operation_id = result["operations"][0]["id"]
            # Inspect completes and prepare is submitted; a new controller must
            # consume its result and use the restore helper after Triton's error.
            self.loader.get_version_operation(operation_id)
            restarted = TritonHotLoader(self.loader.config)
            restarted._batch_v1_api, restarted._core_v1_api = batch, core
            for _ in range(12):
                op = restarted.get_version_operation(operation_id)
                if not op["pending"]:
                    break
            self.assertEqual(op["status"], "FAILED_RESTORED")
            self.assertTrue((self.model_dir / "2").is_dir())
            self.assertEqual((self.model_dir / "config.pbtxt").read_text(), self.original)
            self.assertEqual(len(batch.manifests), 3)  # inspect, prepare, restore


if __name__ == "__main__":
    unittest.main()
