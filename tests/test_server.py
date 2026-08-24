from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from hot_loader import HotLoaderConfig, HotLoaderConflictError, TritonHotLoader
from server import (
    TRITON_METRICS_PORT_OVERRIDE_HEADER,
    TRITON_URL_OVERRIDE_HEADER,
    create_app,
)


class ServerRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        base_dir = Path(self.temp_dir.name)
        self.config = HotLoaderConfig(
            triton_url="http://127.0.0.1:8000",
            model_repository=base_dir / "repository" / "trt_models",
            state_file=base_dir / "state.json",
            staging_root=base_dir / "staging",
            model_source_path="/trt_models",
            model_target_path="/repository/trt_models",
            triton_repository_pvc="triton-repository-pvc",
            k8s_namespace="default",
            model_image_registry_prefix="ccr.ccs.tencentyun.com/clobotics/",
        )
        self.loader = TritonHotLoader(self.config)
        self.client = TestClient(create_app(self.loader, enable_background_worker=False))
        self.addCleanup(self.client.close)

    def test_status_uses_request_header_override_without_mutating_base_loader(self) -> None:
        metrics_payload = {
            "available": False,
            "url": None,
            "candidate_urls": [],
            "detail": "metrics unavailable",
            "updated_at": "2026-06-04T00:00:00+00:00",
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

        with patch.object(TritonHotLoader, "triton_ready", return_value={"ready": True, "detail": "OK"}), patch.object(
            TritonHotLoader,
            "list_repository_models",
            return_value=[],
        ), patch.object(TritonHotLoader, "get_triton_gpu_metrics", return_value=metrics_payload):
            response = self.client.get(
                "/api/status",
                headers={TRITON_URL_OVERRIDE_HEADER: "http://127.0.0.1:19000"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["triton"]["url"], "http://127.0.0.1:19000")

            default_response = self.client.get("/api/status")
            self.assertEqual(default_response.status_code, 200)
            self.assertEqual(default_response.json()["triton"]["url"], "http://127.0.0.1:8000")

        self.assertEqual(self.loader.config.triton_url, "http://127.0.0.1:8000")

    def test_status_uses_metrics_port_override_with_effective_triton_host(self) -> None:
        def fake_metrics(self):
            return {
                "available": True,
                "url": self.config.triton_metrics_url,
                "candidate_urls": [self.config.triton_metrics_url] if self.config.triton_metrics_url else [],
                "detail": "ok",
                "updated_at": "2026-06-04T00:00:00+00:00",
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

        with patch.object(TritonHotLoader, "triton_ready", return_value={"ready": True, "detail": "OK"}), patch.object(
            TritonHotLoader,
            "list_repository_models",
            return_value=[],
        ), patch.object(TritonHotLoader, "get_triton_gpu_metrics", fake_metrics):
            response = self.client.get(
                "/api/status",
                headers={
                    TRITON_URL_OVERRIDE_HEADER: "http://10.0.0.8:19000",
                    TRITON_METRICS_PORT_OVERRIDE_HEADER: "19002",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["triton"]["metrics"]["url"],
            "http://10.0.0.8:19002/metrics",
        )

    def test_api_models_load_route_waits_for_final_status(self) -> None:
        captured = {}

        def fake_create_model_copy_job_and_wait(self, model_name, image):
            captured["triton_url"] = self.config.triton_url
            captured["model_name"] = model_name
            captured["image"] = image
            effective_model_name = model_name or "demo_model"
            return {
                "success": True,
                "job_name": "model-copy-demo",
                "model_name": effective_model_name,
                "status": "MODEL_READY",
            }

        with patch.object(TritonHotLoader, "create_model_copy_job_and_wait", fake_create_model_copy_job_and_wait):
            response = self.client.post(
                "/api/models/load",
                json={
                    "image": "ccr.ccs.tencentyun.com/clobotics/demo:new",
                    "wait_for_ready": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(captured["model_name"])
        self.assertEqual(captured["image"], "ccr.ccs.tencentyun.com/clobotics/demo:new")
        self.assertEqual(response.json()["job_name"], "model-copy-demo")
        self.assertEqual(response.json()["status"], "MODEL_READY")

    def test_api_models_load_route_can_submit_without_waiting(self) -> None:
        captured = {}

        def fake_create_model_copy_job(self, model_name, image, callback=None):
            captured["model_name"] = model_name
            captured["image"] = image
            captured["callback"] = callback
            return {
                "success": True,
                "job_name": "model-copy-demo",
                "model_name": model_name or "demo_model",
                "status": "JOB_CREATED",
            }

        with patch.object(TritonHotLoader, "create_model_copy_job", fake_create_model_copy_job):
            response = self.client.post(
                "/api/models/load",
                json={
                    "image": "ccr.ccs.tencentyun.com/clobotics/demo:new",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(captured["model_name"])
        self.assertEqual(captured["image"], "ccr.ccs.tencentyun.com/clobotics/demo:new")
        self.assertIsNone(captured["callback"])
        self.assertEqual(response.json()["status"], "JOB_CREATED")

    def test_api_models_load_route_returns_conflict_for_different_active_image(self) -> None:
        with patch.object(
            TritonHotLoader,
            "create_model_copy_job",
            side_effect=HotLoaderConflictError("demo_model already has an active operation"),
        ):
            response = self.client.post(
                "/api/models/load",
                json={"image": "ccr.ccs.tencentyun.com/clobotics/demo:20260818"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertFalse(response.json()["success"])

    def test_api_models_load_route_registers_callback(self) -> None:
        captured = {}

        def fake_create_model_copy_job(self, model_name, image, callback=None):
            captured["callback"] = callback
            return {
                "success": True,
                "job_name": "model-copy-demo",
                "model_name": model_name or "demo_model",
                "status": "JOB_CREATED",
                "callback_registered": bool(callback),
            }

        with patch.object(TritonHotLoader, "create_model_copy_job", fake_create_model_copy_job):
            response = self.client.post(
                "/api/models/load",
                json={
                    "image": "ccr.ccs.tencentyun.com/clobotics/demo:new",
                    "callback": {
                        "url": "https://callback.example.com/hook",
                        "events": ["terminal"],
                        "token": "secret-token",
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            captured["callback"],
            {
                "url": "https://callback.example.com/hook",
                "events": ["terminal"],
                "token": "secret-token",
            },
        )
        self.assertTrue(response.json()["callback_registered"])

    def test_api_models_load_batch_route_passes_models_list(self) -> None:
        captured = {}

        def fake_load_models_from_images_sync(self, models):
            captured["models"] = models
            return {"success": True, "completed": [{"job_name": "job-a", "status": "MODEL_READY"}], "errors": []}

        with patch.object(TritonHotLoader, "load_models_from_images_sync", fake_load_models_from_images_sync):
            response = self.client.post(
                "/api/models/load-batch",
                json={
                    "wait_for_ready": True,
                    "models": [
                        {
                            "image": "ccr.ccs.tencentyun.com/clobotics/demo:a",
                        }
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["models"][0], {"image": "ccr.ccs.tencentyun.com/clobotics/demo:a"})
        self.assertEqual(response.json()["completed"][0]["job_name"], "job-a")

    def test_api_models_load_batch_route_can_submit_without_waiting(self) -> None:
        captured = {}

        def fake_load_models_from_images(self, models):
            captured["models"] = models
            return {"success": True, "submitted": [{"job_name": "job-a", "status": "JOB_CREATED"}], "errors": []}

        with patch.object(TritonHotLoader, "load_models_from_images", fake_load_models_from_images):
            response = self.client.post(
                "/api/models/load-batch",
                json={
                    "models": [
                        {
                            "image": "ccr.ccs.tencentyun.com/clobotics/demo:a",
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["models"][0], {"image": "ccr.ccs.tencentyun.com/clobotics/demo:a"})
        self.assertEqual(response.json()["submitted"][0]["job_name"], "job-a")

    def test_api_jobs_status_route_delegates_to_loader(self) -> None:
        with patch.object(
            TritonHotLoader,
            "get_job_status",
            return_value={
                "job_name": "model-copy-demo",
                "model_name": "demo_model",
                "status": "COPY_RUNNING",
                "pod_name": "pod-1",
                "logs": "copying",
            },
        ):
            response = self.client.get("/api/jobs/model-copy-demo")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "COPY_RUNNING")

    def test_api_unload_route_passes_model_list_to_loader(self) -> None:
        captured = {}

        def fake_unload_models(self, model_names):
            captured["model_names"] = list(model_names)
            return {
                "success": True,
                "unloaded_models": list(model_names),
                "affected_aliases": [],
                "state": {"managed_model_count": 0},
            }

        with patch.object(TritonHotLoader, "unload_models", fake_unload_models), patch.object(
            TritonHotLoader,
            "get_managed_state",
            return_value={"managed_model_count": 0},
        ):
            response = self.client.post(
                "/api/unload",
                json={"models": ["demo_model_a", "demo_model_b"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["model_names"], ["demo_model_a", "demo_model_b"])
        self.assertTrue(response.json()["success"])
        self.assertEqual(
            response.json()["model_result"]["unloaded_models"],
            ["demo_model_a", "demo_model_b"],
        )

    def test_api_unload_route_rejects_version_level_request(self) -> None:
        response = self.client.post(
            "/api/unload",
            json={"versions": ["demo_model@3"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("取消版本管理", response.json()["detail"])

    def test_runtime_gpu_status_route_formats_summary_payload(self) -> None:
        with patch.object(
            TritonHotLoader,
            "get_triton_gpu_metrics",
            return_value={
                "available": True,
                "url": "http://127.0.0.1:8002/metrics",
                "detail": "ok",
                "updated_at": "2026-06-05T00:00:00+00:00",
                "summary": {
                    "device_count": 1,
                    "used_bytes": 2 * 1024 * 1024,
                    "total_bytes": 8 * 1024 * 1024,
                    "used_ratio": 0.25,
                    "used_percent": 25.0,
                    "average_utilization_ratio": 0.75,
                    "average_utilization_percent": 75.0,
                    "total_power_usage_watts": 95.4,
                },
                "gpus": [
                    {
                        "index": 0,
                        "gpu_uuid": "GPU-0",
                        "gpu_bus_id": "0000:01:00.0",
                        "used_bytes": 2 * 1024 * 1024,
                        "total_bytes": 8 * 1024 * 1024,
                        "used_percent": 25.0,
                        "utilization_percent": 75.0,
                        "power_usage_watts": 95.4,
                    }
                ],
            },
        ):
            response = self.client.get("/runtime/gpu-status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["source_url"], "http://127.0.0.1:8002/metrics")
        self.assertEqual(payload["gpus"][0]["gpu_index"], 0)
        self.assertEqual(payload["gpus"][0]["memory_total_mb"], 8)
        self.assertEqual(payload["gpus"][0]["memory_used_mb"], 2)
        self.assertEqual(payload["gpus"][0]["memory_free_mb"], 6)
        self.assertEqual(payload["gpus"][0]["gpu_utilization_percent"], 75.0)


if __name__ == "__main__":
    unittest.main()
