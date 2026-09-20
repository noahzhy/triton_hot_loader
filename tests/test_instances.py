from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from hot_loader import HotLoaderConfig, HotLoaderConflictError, HotLoaderError, TritonHotLoader
from server import TRITON_INSTANCE_HEADER, _background_watch_loop, _watch_instance, create_app
from tests.test_hot_loader import FakeBatchApi, FakeCoreApi, write_model_bundle


class MockTriton:
    """A real HTTP endpoint with independent runtime state (no GPU required)."""

    def __init__(self):
        self.loaded = set()
        self.calls = []
        self.callbacks = []
        self.fail_load = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.respond()

            def do_POST(self):
                self.respond()

            def respond(self):
                owner.calls.append((self.command, self.path))
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.endswith("/load") and owner.fail_load:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"temporarily offline")
                    return
                if self.path == "/callback":
                    owner.callbacks.append(json.loads(body))
                if self.path.endswith("/load"):
                    owner.loaded.add(self.path.split("/")[-2])
                elif self.path.endswith("/unload"):
                    owner.loaded.discard(self.path.split("/")[-2])
                if self.path.endswith("/index"):
                    payload = [{"name": name, "version": "1", "state": "READY"} for name in owner.loaded]
                else:
                    payload = {}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class InstanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = HotLoaderConfig(
            triton_url="http://127.0.0.1:8000", model_repository=root / "models",
            state_file=root / "state.json", staging_root=root / "staging",
            model_target_path=str(root / "models"),
        )
        self.loader = TritonHotLoader(self.config)
        self.loader._batch_v1_api = FakeBatchApi()
        self.loader._core_v1_api = FakeCoreApi()
        self.client = TestClient(create_app(self.loader, enable_background_worker=False))
        self.addCleanup(self.client.close)
        self.record = self.loader.save_instance("GPU B", "10.0.0.2")
        self.other = self.loader.for_instance(self.record["id"])

    def test_registry_defaults_crud_persistence_and_validation(self):
        self.assertEqual(self.record["triton_url"], "http://10.0.0.2:8000")
        self.assertEqual(self.record["metrics_url"], "http://10.0.0.2:8002/metrics")
        added = self.client.post("/api/instances", json={"name": "C", "triton_url": "10.0.0.3:9000"})
        self.assertEqual(added.status_code, 201)
        identifier = added.json()["id"]
        updated = self.client.put(f"/api/instances/{identifier}", json={
            "name": "renamed", "triton_url": "10.0.0.3:9001", "metrics_url": "10.0.0.3:9003"})
        self.assertEqual(updated.status_code, 200)
        restarted = TritonHotLoader(self.config)
        self.assertEqual(restarted.for_instance(identifier).config.triton_url, "http://10.0.0.3:9001")
        self.assertEqual(self.client.delete(f"/api/instances/{identifier}").status_code, 200)
        self.assertEqual(self.client.delete("/api/instances/default").status_code, 400)
        for url in ["", "http://", "ftp://a", "http://a:0", "http://a:65536", "http://a/path", "http://u:p@a", "a b", "http://<b>"]:
            with self.subTest(url=url):
                self.assertEqual(self.client.post("/api/instances", json={"name": "bad", "triton_url": url}).status_code // 100, 4)
        self.assertEqual(self.client.post("/api/instances", json={"name": "dup", "triton_url": "10.0.0.2:8000/"}).status_code, 409)
        self.assertEqual(TritonHotLoader.normalize_endpoint("::1"), "http://[::1]:8000")

    def test_unknown_and_mixed_selectors_rejected_before_network(self):
        for headers in [
            {TRITON_INSTANCE_HEADER: "missing"},
            {"x-hot-triton-url": "10.8.0.1"},
            {TRITON_INSTANCE_HEADER: "default", "x-hot-triton-url": self.config.triton_url},
            {"x-hot-triton-url": "10.0.0.2", "x-hot-triton-metrics-port": "9999"},
        ]:
            with self.subTest(headers=headers), patch.object(TritonHotLoader, "_triton_request") as request:
                self.assertEqual(self.client.get("/api/status", headers=headers).status_code, 400)
                request.assert_not_called()

    def test_active_copy_conflicts_globally_and_reuses_only_same_instance(self):
        image = "ccr.ccs.tencentyun.com/clobotics/demo:1"
        first = self.loader.create_model_copy_job("demo", image)
        self.assertTrue(self.loader.create_model_copy_job("demo", image)["reused"])
        with self.assertRaises(HotLoaderConflictError):
            self.other.create_model_copy_job("demo", image)
        # Different models remain independent.
        second = self.other.create_model_copy_job("different", image)
        self.assertEqual(second["instance_id"], self.record["id"])
        self.assertEqual(set(self.loader.get_managed_state()["jobs"]), {first["job_name"]})
        self.assertEqual(set(self.other.get_managed_state()["jobs"]), {second["job_name"]})
        with self.assertRaisesRegex(HotLoaderError, "不属于"):
            self.other.get_job_status(first["job_name"])
        headers = {TRITON_INSTANCE_HEADER: self.record["id"]}
        self.assertEqual(self.client.get(f"/api/jobs/{first['job_name']}", headers=headers).status_code, 400)

    def test_concurrent_instances_cannot_create_two_copy_jobs_for_same_model(self):
        barrier = threading.Barrier(2)
        results = []
        def submit(loader):
            barrier.wait()
            try:
                results.append(loader.create_model_copy_job("demo", "ccr.ccs.tencentyun.com/clobotics/demo:1"))
            except HotLoaderConflictError:
                results.append("conflict")
        threads = [threading.Thread(target=submit, args=(loader,)) for loader in [self.loader, self.other]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(results.count("conflict"), 1)
        self.assertEqual(len(self.loader._batch_v1_api.created_jobs), 1)

    def test_pending_work_blocks_address_changes_and_deletion_but_not_rename(self):
        self.loader._update_job_state("default-pending", status="JOB_CREATED")
        renamed = self.loader.save_instance("默认实例新名称", self.config.triton_url, instance_id="default")
        self.assertIsNone(renamed["metrics_url"])
        self.other._update_job_state("pending", status="JOB_CREATED")
        with self.assertRaises(HotLoaderConflictError):
            self.loader.delete_instance(self.other.instance_id)
        with self.assertRaises(HotLoaderConflictError):
            self.loader.save_instance("new", "10.0.0.9", instance_id=self.other.instance_id)
        self.loader.save_instance("renamed", self.record["triton_url"], self.record["metrics_url"], instance_id=self.other.instance_id)
        self.other._update_job_state("pending", status="MODEL_READY", callback={"events": ["terminal"], "url": "http://callback"})
        with self.assertRaises(HotLoaderConflictError):
            self.loader.delete_instance(self.other.instance_id)
        self.other.record_terminal_callback_result("pending", delivered=True, event_id="e")
        self.loader.delete_instance(self.other.instance_id)
        self.assertIn("pending", self.loader._load_state()["jobs"])
        with self.assertRaises(HotLoaderConflictError):
            self.other.create_model_copy_job("demo", "ccr.ccs.tencentyun.com/clobotics/demo:1")

    def test_legacy_jobs_migrate_without_losing_shared_model_metadata(self):
        self.loader._save_state({"aliases": {"model_demo": {"models": ["demo"], "image": "old"}},
                                 "jobs": {"old-job": {"status": "JOB_CREATED"}}, "updated_at": None})
        restarted = TritonHotLoader(self.config)
        raw = restarted._load_state()
        self.assertEqual(raw["jobs"]["old-job"]["instance_id"], "default")
        self.assertEqual(raw["jobs"]["old-job"]["triton_url"], self.config.triton_url)
        self.assertEqual(raw["aliases"]["model_demo"]["image"], "old")
        # Subsequent environment changes cannot silently retarget the registry.
        changed = TritonHotLoader(self.config.with_updates(triton_url="http://10.9.0.1:8000"))
        self.assertEqual(changed.for_instance("default").config.triton_url, self.config.triton_url)

    def test_one_blocked_background_instance_does_not_delay_another(self):
        stop, blocked, release, progressed = (threading.Event() for _ in range(4))
        def watch(loader):
            if loader.instance_id == "default":
                blocked.set()
                release.wait(timeout=3)
            else:
                progressed.set()
        with patch("server._watch_instance", side_effect=watch):
            thread = threading.Thread(target=_background_watch_loop, args=(self.loader, stop))
            thread.start()
            try:
                self.assertTrue(blocked.wait(timeout=2))
                self.assertTrue(progressed.wait(timeout=2))
            finally:
                stop.set()
                release.set()
                thread.join(timeout=3)
            self.assertFalse(thread.is_alive())


class TwoTritonHTTPTests(unittest.TestCase):
    def test_retry_of_failed_endpoint_stays_bound_while_other_instance_finishes(self):
        a, b = MockTriton(), MockTriton()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        a.fail_load = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HotLoaderConfig(triton_url=a.url, model_repository=root / "models",
                state_file=root / "state.json", staging_root=root / "staging",
                model_target_path=str(root / "models"))
            loader = TritonHotLoader(config)
            record = loader.save_instance("B", b.url)
            write_model_bundle(config.model_repository, "model_a", ["1"])
            write_model_bundle(config.model_repository, "model_b", ["1"])
            batch, core = FakeBatchApi(), FakeCoreApi()
            batch.job_to_read = SimpleNamespace(metadata=SimpleNamespace(annotations={}),
                                               status=SimpleNamespace(succeeded=1, failed=0, active=0))
            loader._batch_v1_api, loader._core_v1_api = batch, core
            job_a = loader.create_model_copy_job("model_a", "ccr.ccs.tencentyun.com/clobotics/model_a:1")["job_name"]
            other = loader.for_instance(record["id"])
            job_b = other.create_model_copy_job("model_b", "ccr.ccs.tencentyun.com/clobotics/model_b:1")["job_name"]
            _watch_instance(loader)
            _watch_instance(other)
            self.assertEqual(loader.get_managed_state()["jobs"][job_a]["status"], "TRITON_RELOAD_RUNNING")
            self.assertEqual(other.get_managed_state()["jobs"][job_b]["status"], "MODEL_READY")
            loader._update_job_state(job_a, triton_reload_next_attempt_at="2000-01-01T00:00:00+00:00")
            restarted = TritonHotLoader(config)
            restarted._batch_v1_api, restarted._core_v1_api = batch, core
            a.fail_load = False
            _watch_instance(restarted)
            self.assertEqual(restarted.get_managed_state()["jobs"][job_a]["status"], "MODEL_READY")
            self.assertEqual(a.loaded, {"model_a"})
            self.assertEqual(b.loaded, {"model_b"})

    def test_load_unload_reload_jobs_callbacks_and_restart_are_targeted(self):
        a, b = MockTriton(), MockTriton()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HotLoaderConfig(triton_url=a.url, triton_metrics_url=a.url + "/metrics",
                model_repository=root / "models", state_file=root / "state.json",
                staging_root=root / "staging", model_target_path=str(root / "models"))
            loader = TritonHotLoader(config)
            record = loader.save_instance("B", b.url, b.url + "/metrics")
            write_model_bundle(config.model_repository, "demo", ["1"])
            batch, core = FakeBatchApi(), FakeCoreApi()
            batch.job_to_read = SimpleNamespace(metadata=SimpleNamespace(annotations={}), status=SimpleNamespace(succeeded=1, failed=0, active=0))
            loader._batch_v1_api, loader._core_v1_api = batch, core
            headers = {TRITON_INSTANCE_HEADER: record["id"]}
            with TestClient(create_app(loader, enable_background_worker=False)) as client:
                submitted = client.post("/api/models/load", headers=headers, json={
                    "model_name": "demo", "image": "ccr.ccs.tencentyun.com/clobotics/demo:1",
                    "callback": {"url": b.url + "/callback"}})
                self.assertEqual(submitted.status_code, 200)
                job = submitted.json()["job_name"]
                self.assertEqual(client.get("/api/state").json()["jobs"], {})
            # Restart the controller, then advance through the real HTTP load / readiness path.
            restarted = TritonHotLoader(config)
            restarted._batch_v1_api, restarted._core_v1_api = batch, core
            _watch_instance(restarted.for_instance(record["id"]))
            self.assertEqual(b.loaded, {"demo"})
            self.assertEqual(a.loaded, set())
            self.assertEqual(b.callbacks[0]["instance_id"], record["id"])
            self.assertEqual(b.callbacks[0]["triton_url"], b.url)
            with TestClient(create_app(restarted, enable_background_worker=False)) as client:
                result = client.get(f"/api/jobs/{job}", headers=headers).json()
                self.assertEqual(result["status"], "MODEL_READY")
                self.assertEqual(client.get("/api/models", headers=headers).json()["triton_models"][0]["name"], "demo")
                submitted_a = client.post("/api/models/load", json={
                    "model_name": "demo", "image": "ccr.ccs.tencentyun.com/clobotics/demo:1"})
                self.assertEqual(submitted_a.status_code, 200)
                _watch_instance(restarted.for_instance("default"))
                self.assertEqual(a.loaded, {"demo"})
                for target in [None, headers]:
                    for action in ["reload", "unload", "reload"]:
                        response = client.post(f"/api/models/{action}", headers=target, json={"model_name": "demo"})
                        self.assertEqual(response.status_code, 200)
                self.assertEqual(a.loaded, {"demo"})
                self.assertEqual(b.loaded, {"demo"})
                client.post("/api/models/unload", headers=headers, json={"model_name": "demo"})
                self.assertEqual(a.loaded, {"demo"})
                self.assertEqual(b.loaded, set())
                self.assertTrue((config.model_repository / "demo" / "1").exists())
                # A terminal historical job remains readable after changing the instance address.
                restarted.save_instance("B moved", "10.99.0.1", instance_id=record["id"])
                old = client.get(f"/api/jobs/{job}", headers=headers).json()
                self.assertEqual(old["triton_url"], b.url)
                self.assertEqual(old["status"], "MODEL_READY")
            self.assertEqual(sum(path.endswith("/load") for _, path in a.calls), 3)
            self.assertEqual(sum(path.endswith("/load") for _, path in b.calls), 3)


if __name__ == "__main__":
    unittest.main()
