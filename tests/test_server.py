from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from hot_loader import HotLoaderConfig, TritonHotLoader
from server import (
    TRITON_METRICS_PORT_OVERRIDE_HEADER,
    TRITON_URL_OVERRIDE_HEADER,
    create_app,
)


class ServerTritonUrlOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        base_dir = Path(self.temp_dir.name)
        self.config = HotLoaderConfig(
            triton_url="http://127.0.0.1:8000",
            model_repository=base_dir / "model_repository",
            state_file=base_dir / "state.json",
            staging_root=base_dir / "staging",
        )
        self.loader = TritonHotLoader(self.config)
        self.client = TestClient(create_app(self.loader))
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

    def test_apply_config_route_uses_request_header_override(self) -> None:
        captured: dict[str, object] = {}

        def fake_apply_config(self, config_map, *, prune_missing=True, force=False):
            captured["triton_url"] = self.config.triton_url
            captured["config"] = config_map
            captured["prune_missing"] = prune_missing
            captured["force"] = force
            return {"success": True, "applied": [], "skipped": [], "removed": [], "errors": []}

        with patch.object(TritonHotLoader, "apply_config", fake_apply_config):
            response = self.client.post(
                "/api/apply-config",
                headers={TRITON_URL_OVERRIDE_HEADER: "http://127.0.0.1:29000"},
                json={
                    "config": {"demo": "registry.example.com/demo:model"},
                    "prune_missing": False,
                    "force": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["triton_url"], "http://127.0.0.1:29000")
        self.assertEqual(captured["config"], {"demo": "registry.example.com/demo:model"})
        self.assertFalse(captured["prune_missing"])
        self.assertTrue(captured["force"])

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


if __name__ == "__main__":
    unittest.main()
