from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hot_loader import HotLoaderConfig, TritonHotLoader, _default_runtime_paths


def write_model_bundle(
    root_dir: Path,
    model_name: str,
    versions: list[str],
    *,
    include_version_policy: bool = False,
) -> Path:
    model_dir = root_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    config_lines = [
        f'name: "{model_name}"',
        'platform: "onnxruntime_onnx"',
        'max_batch_size: 0',
    ]
    if include_version_policy:
        config_lines.extend(
            [
                "version_policy: {",
                "  latest {",
                "    num_versions: 2",
                "  }",
                "}",
            ]
        )
    (model_dir / "config.pbtxt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    for version in versions:
        version_dir = model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "model.onnx").write_text(f"fake model payload {version}\n", encoding="utf-8")

    return model_dir


class TritonHotLoaderVersionLoadingTests(unittest.TestCase):
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

    def test_apply_alias_records_versions_and_uses_load_api_reload(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["1"])
        self.loader._save_state(
            {
                "aliases": {
                    "demo_alias": {
                        "image": "registry.example.com/demo:old",
                        "models": ["demo_model"],
                        "model_versions": {"demo_model": ["1"]},
                        "active_versions": {"demo_model": "1"},
                        "updated_at": "2026-06-03T00:00:00+00:00",
                    }
                },
                "updated_at": "2026-06-03T00:00:00+00:00",
            }
        )

        stage_dir = self.config.staging_root / "apply-case"
        bundle_dir = stage_dir / "bundle"
        write_model_bundle(bundle_dir, "demo_model", ["2"])

        events: list[tuple[str, str]] = []
        self.loader._stage_image_bundle = lambda image_ref: (stage_dir, bundle_dir, ["demo_model"])
        self.loader._load_model = lambda model_name: events.append(("load", model_name))
        self.loader._unload_model = lambda model_name, tolerate_missing=True: events.append(("unload", model_name))

        result = self.loader._apply_alias("demo_alias", "registry.example.com/demo:new")
        state = self.loader.get_managed_state()
        alias_meta = state["aliases"]["demo_alias"]
        config_text = (self.config.model_repository / "demo_model" / "config.pbtxt").read_text(
            encoding="utf-8"
        )

        self.assertEqual(events, [("load", "demo_model")])
        self.assertEqual(result["model_versions"], {"demo_model": ["2"]})
        self.assertEqual(result["active_versions"], {"demo_model": "2"})
        self.assertEqual(result["version_policy_models"], ["demo_model"])
        self.assertEqual(alias_meta["model_versions"], {"demo_model": ["2"]})
        self.assertEqual(alias_meta["active_versions"], {"demo_model": "2"})
        self.assertIn("version_policy:", config_text)
        self.assertIn("versions: [ 2 ]", config_text)
        self.assertNotIn("versions: [ 1 ]", config_text)

    def test_stage_image_model_uses_project_image_root_plus_model_name(self) -> None:
        commands: list[list[str]] = []

        def fake_pull_image(image_ref: str) -> None:
            commands.append(["docker", "pull", image_ref])

        def fake_run_command(args, *, allow_failure: bool = False):
            commands.append(list(args))
            if args[:2] == ["docker", "create"]:
                return subprocess.CompletedProcess(args, 0, stdout="container-123\n", stderr="")
            if args[:2] == ["docker", "cp"]:
                target_dir = Path(args[-1])
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "config.pbtxt").write_text('name: "demo_model"\n', encoding="utf-8")
                version_dir = target_dir / "1"
                version_dir.mkdir(parents=True, exist_ok=True)
                (version_dir / "model.onnx").write_text("fake\n", encoding="utf-8")
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        self.loader._pull_image = fake_pull_image  # type: ignore[method-assign]
        self.loader._run_command = fake_run_command  # type: ignore[method-assign]

        stage_dir, bundle_dir = self.loader._stage_image_model(
            "registry.example.com/demo:v1",
            "demo_model",
        )

        self.assertEqual(
            commands[2],
            [
                "docker",
                "cp",
                "container-123:/trt_models/demo_model/.",
                str(bundle_dir / "demo_model"),
            ],
        )
        self.assertTrue((bundle_dir / "demo_model" / "config.pbtxt").exists())
        self.assertTrue((bundle_dir / "demo_model" / "1" / "model.onnx").exists())
        self.assertTrue(stage_dir.exists())

    def test_validate_config_map_ignores_json_keys_and_skips_mlman_config(self) -> None:
        normalized = self.loader._validate_config_map(
            {
                "whatever": "registry.example.com/team/unit-model:v1",
                "mlman_config": "ccr.ccs.tencentyun.com/clobotics/mlmanconfig-init:hpc_us-20260529_1",
                "another_placeholder": "registry.example.com/team/sku-model:v2",
            }
        )

        self.assertEqual(
            normalized,
            {
                "registry.example.com/team/unit-model:v1": "registry.example.com/team/unit-model:v1",
                "registry.example.com/team/sku-model:v2": "registry.example.com/team/sku-model:v2",
            },
        )

    def test_apply_config_matches_existing_entry_by_image_not_json_key(self) -> None:
        self.loader._save_state(
            {
                "aliases": {
                    "legacy_alias": {
                        "image": "registry.example.com/team/unit-model:v1",
                        "models": ["unit_model"],
                        "model_versions": {"unit_model": ["1"]},
                        "active_versions": {"unit_model": "1"},
                        "updated_at": "2026-06-03T00:00:00+00:00",
                    }
                },
                "updated_at": "2026-06-03T00:00:00+00:00",
            }
        )

        calls: list[tuple[str, str]] = []

        def fail_if_called(bundle_id: str, image_ref: str) -> dict[str, str]:
            calls.append((bundle_id, image_ref))
            raise AssertionError("_apply_alias should not be called for unchanged image")

        self.loader._apply_alias = fail_if_called  # type: ignore[method-assign]

        result = self.loader.apply_config(
            {
                "renamed_json_key": "registry.example.com/team/unit-model:v1",
            },
            prune_missing=False,
            force=False,
        )

        self.assertEqual(calls, [])
        self.assertTrue(result["success"])
        self.assertEqual(result["requested_images"], ["registry.example.com/team/unit-model:v1"])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(result["skipped"][0]["bundle_id"], "legacy_alias")
        self.assertEqual(result["skipped"][0]["image"], "registry.example.com/team/unit-model:v1")

    def test_reload_models_uses_load_only(self) -> None:
        events: list[tuple[str, str]] = []
        self.loader._load_model = lambda model_name: events.append(("load", model_name))
        self.loader._unload_model = lambda model_name, tolerate_missing=True: events.append(("unload", model_name))

        result = self.loader.reload_models(["demo_model", "demo_model"])

        self.assertEqual(events, [("load", "demo_model")])
        self.assertEqual(result["reloaded_models"], ["demo_model"])

    def test_unload_model_versions_removes_selected_version_and_reloads_remaining(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["1", "2", "3"])
        self.loader._save_state(
            {
                "aliases": {
                    "demo_alias": {
                        "image": "registry.example.com/demo:multi",
                        "models": ["demo_model"],
                        "model_versions": {"demo_model": ["1", "2", "3"]},
                        "active_versions": {"demo_model": "3"},
                        "updated_at": "2026-06-03T00:00:00+00:00",
                    }
                },
                "updated_at": "2026-06-03T00:00:00+00:00",
            }
        )

        events: list[tuple[str, str]] = []
        self.loader._load_model = lambda model_name: events.append(("load", model_name))
        self.loader._unload_model = lambda model_name, tolerate_missing=True: events.append(("unload", model_name))

        result = self.loader.unload_model_versions(["demo_model@3"])
        state = self.loader.get_managed_state()
        alias_meta = state["aliases"]["demo_alias"]
        config_text = (self.config.model_repository / "demo_model" / "config.pbtxt").read_text(
            encoding="utf-8"
        )

        self.assertEqual(events, [("load", "demo_model")])
        self.assertEqual(alias_meta["model_versions"], {"demo_model": ["1", "2"]})
        self.assertEqual(alias_meta["active_versions"], {"demo_model": "2"})
        self.assertTrue((self.config.model_repository / "demo_model" / "1").exists())
        self.assertTrue((self.config.model_repository / "demo_model" / "2").exists())
        self.assertFalse((self.config.model_repository / "demo_model" / "3").exists())
        self.assertIn("versions: [ 2 ]", config_text)
        self.assertEqual(result["removed_versions"][0]["removed_versions"], ["3"])
        self.assertEqual(result["removed_versions"][0]["remaining_versions"], ["1", "2"])
        self.assertEqual(result["switched_active_versions"], [{"model": "demo_model", "from": "3", "to": "2"}])

    def test_unload_model_versions_removes_last_version_and_alias(self) -> None:
        write_model_bundle(self.config.model_repository, "solo_model", ["7"])
        self.loader._save_state(
            {
                "aliases": {
                    "solo_alias": {
                        "image": "registry.example.com/solo:7",
                        "models": ["solo_model"],
                        "model_versions": {"solo_model": ["7"]},
                        "active_versions": {"solo_model": "7"},
                        "updated_at": "2026-06-03T00:00:00+00:00",
                    }
                },
                "updated_at": "2026-06-03T00:00:00+00:00",
            }
        )

        events: list[tuple[str, str]] = []
        self.loader._load_model = lambda model_name: events.append(("load", model_name))
        self.loader._unload_model = lambda model_name, tolerate_missing=True: events.append(("unload", model_name))

        result = self.loader.unload_model_versions(["solo_model@7"])
        state = self.loader.get_managed_state()

        self.assertEqual(events, [("unload", "solo_model")])
        self.assertFalse((self.config.model_repository / "solo_model").exists())
        self.assertNotIn("solo_alias", state["aliases"])
        self.assertEqual(result["removed_models"], ["solo_model"])
        self.assertEqual(result["deleted_aliases"], ["solo_alias"])
        self.assertTrue(result["removed_versions"][0]["unloaded_model"])

    def test_get_managed_state_backfills_versions_for_legacy_state(self) -> None:
        write_model_bundle(self.config.model_repository, "legacy_model", ["20260530"])
        self.loader._save_state(
            {
                "aliases": {
                    "legacy_alias": {
                        "image": "registry.example.com/legacy:20260530",
                        "models": ["legacy_model"],
                        "updated_at": "2026-06-03T00:00:00+00:00",
                    }
                },
                "updated_at": "2026-06-03T00:00:00+00:00",
            }
        )

        state = self.loader.get_managed_state()
        alias_meta = state["aliases"]["legacy_alias"]

        self.assertEqual(alias_meta["model_versions"], {"legacy_model": ["20260530"]})
        self.assertEqual(alias_meta["active_versions"], {"legacy_model": "20260530"})
        self.assertEqual(state["managed_model_versions"], {"legacy_model": ["20260530"]})
        self.assertEqual(state["managed_active_versions"], {"legacy_model": "20260530"})

    def test_load_model_from_image_replaces_existing_model_and_updates_state(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["1"])
        self.loader._save_state(
            {
                "aliases": {
                    "legacy_alias": {
                        "image": "registry.example.com/demo:old",
                        "models": ["demo_model"],
                        "model_versions": {"demo_model": ["1"]},
                        "active_versions": {"demo_model": "1"},
                        "updated_at": "2026-06-03T00:00:00+00:00",
                    }
                },
                "updated_at": "2026-06-03T00:00:00+00:00",
            }
        )

        stage_dir = self.config.staging_root / "load-from-image"
        bundle_dir = stage_dir / "bundle"
        write_model_bundle(bundle_dir, "demo_model", ["2"])

        events: list[tuple[str, str]] = []
        self.loader._stage_image_model = lambda image_ref, model_name: (stage_dir, bundle_dir)  # type: ignore[method-assign]
        self.loader._unload_model = lambda model_name, tolerate_missing=True: events.append(("unload", model_name))
        self.loader._load_model = lambda model_name: events.append(("load", model_name))

        result = self.loader.load_model_from_image(
            "demo_model",
            "registry.example.com/demo:new",
        )
        state = self.loader.get_managed_state()
        new_alias = result["alias"]
        new_meta = state["aliases"][new_alias]

        self.assertEqual(events, [("unload", "demo_model"), ("load", "demo_model")])
        self.assertNotIn("legacy_alias", state["aliases"])
        self.assertEqual(new_meta["image"], "registry.example.com/demo:new")
        self.assertEqual(new_meta["models"], ["demo_model"])
        self.assertEqual(new_meta["model_versions"], {"demo_model": ["2"]})
        self.assertEqual(new_meta["active_versions"], {"demo_model": "2"})
        self.assertFalse((self.config.model_repository / "demo_model" / "1").exists())
        self.assertTrue((self.config.model_repository / "demo_model" / "2" / "model.onnx").exists())

    def test_write_active_version_policy_replaces_existing_policy(self) -> None:
        model_dir = write_model_bundle(
            self.config.model_repository,
            "policy_model",
            ["5", "7"],
            include_version_policy=True,
        )

        updated = self.loader._write_active_version_policy(model_dir, "7")
        config_text = (model_dir / "config.pbtxt").read_text(encoding="utf-8")

        self.assertTrue(updated)
        self.assertEqual(config_text.count("version_policy:"), 1)
        self.assertIn("versions: [ 7 ]", config_text)
        self.assertNotIn("num_versions: 2", config_text)

    def test_get_triton_gpu_metrics_auto_detects_port_plus_two(self) -> None:
        metrics_text = """
# HELP nv_gpu_utilization GPU utilization rate [0.0 - 1.0)
# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_uuid="GPU-aaa"} 0.25
nv_gpu_utilization{gpu_uuid="GPU-bbb"} 0.5
# HELP nv_gpu_memory_total_bytes GPU total memory, in bytes
# TYPE nv_gpu_memory_total_bytes gauge
nv_gpu_memory_total_bytes{gpu_uuid="GPU-aaa"} 100
nv_gpu_memory_total_bytes{gpu_uuid="GPU-bbb"} 200
# HELP nv_gpu_memory_used_bytes GPU used memory, in bytes
# TYPE nv_gpu_memory_used_bytes gauge
nv_gpu_memory_used_bytes{gpu_uuid="GPU-aaa"} 40
nv_gpu_memory_used_bytes{gpu_uuid="GPU-bbb"} 60
# HELP nv_gpu_power_usage GPU power usage in watts
# TYPE nv_gpu_power_usage gauge
nv_gpu_power_usage{gpu_uuid="GPU-aaa"} 50
nv_gpu_power_usage{gpu_uuid="GPU-bbb"} 75
""".strip()

        class FakeResponse:
            def __init__(self, status_code: int, text: str) -> None:
                self.status_code = status_code
                self.text = text

            @property
            def is_error(self) -> bool:
                return self.status_code >= 400

        with patch(
            "hot_loader.httpx.get",
            side_effect=[
                FakeResponse(404, "not found"),
                FakeResponse(200, metrics_text),
            ],
        ):
            metrics = self.loader.get_triton_gpu_metrics()

        self.assertTrue(metrics["available"])
        self.assertEqual(metrics["url"], "http://127.0.0.1:8002/metrics")
        self.assertEqual(metrics["summary"]["device_count"], 2)
        self.assertEqual(metrics["summary"]["used_bytes"], 100)
        self.assertEqual(metrics["summary"]["total_bytes"], 300)
        self.assertAlmostEqual(metrics["summary"]["used_ratio"], 1 / 3)
        self.assertAlmostEqual(metrics["summary"]["average_utilization_ratio"], 0.375)
        self.assertEqual(metrics["summary"]["total_power_usage_watts"], 125)
        self.assertEqual(
            metrics["gpus"],
            [
                {
                    "index": 0,
                    "label": "GPU 0",
                    "gpu_uuid": "GPU-aaa",
                    "gpu_bus_id": None,
                    "used_bytes": 40,
                    "total_bytes": 100,
                    "used_ratio": 0.4,
                    "used_percent": 40.0,
                    "utilization_ratio": 0.25,
                    "utilization_percent": 25.0,
                    "power_usage_watts": 50.0,
                },
                {
                    "index": 1,
                    "label": "GPU 1",
                    "gpu_uuid": "GPU-bbb",
                    "gpu_bus_id": None,
                    "used_bytes": 60,
                    "total_bytes": 200,
                    "used_ratio": 0.3,
                    "used_percent": 30.0,
                    "utilization_ratio": 0.5,
                    "utilization_percent": 50.0,
                    "power_usage_watts": 75.0,
                },
            ],
        )


class HotLoaderDefaultRuntimePathTests(unittest.TestCase):
    def test_default_runtime_paths_use_env_controlled_repository_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "runtime"
            repository_path = Path(temp_dir) / "repository"

            with patch.dict(os.environ, {"HOT_TRITON_MODEL_REPOSITORY": str(repository_path)}, clear=False):
                model_repository, state_file, staging_root = _default_runtime_paths(base_dir)

        self.assertEqual(model_repository, repository_path)
        self.assertEqual(state_file, repository_path / ".hot_loader" / "state.json")
        self.assertEqual(staging_root, repository_path / ".staging")

    def test_default_runtime_paths_fall_back_to_local_runtime_when_mount_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "runtime"
            model_repository, state_file, staging_root = _default_runtime_paths(base_dir)

        self.assertEqual(model_repository, base_dir / "model_repository")
        self.assertEqual(state_file, base_dir / "state.json")
        self.assertEqual(staging_root, base_dir / "staging")

    def test_default_config_reads_triton_url_from_environment(self) -> None:
        with patch("hot_loader._load_dotenv_values", return_value={}), patch.dict(
            "os.environ",
            {
                "HOT_TRITON_TRITON_URL": "http://10.0.0.8:19000",
                "HOT_TRITON_TRITON_METRICS_URL": "http://10.0.0.8:19002/metrics",
            },
            clear=True,
        ):
            config = HotLoaderConfig.default()

        self.assertEqual(config.triton_url, "http://10.0.0.8:19000")
        self.assertEqual(config.triton_metrics_url, "http://10.0.0.8:19002/metrics")

    def test_default_config_prefers_trt_ip_and_ports_from_environment(self) -> None:
        with patch("hot_loader._load_dotenv_values", return_value={}), patch.dict(
            "os.environ",
            {
                "TRT_IP": "127.0.0.1",
                "HTTP_PORT": "8000",
                "METRICS_PORT": "8002",
                "HOT_TRITON_TRITON_URL": "http://10.0.0.8:19000",
                "HOT_TRITON_TRITON_METRICS_URL": "http://10.0.0.8:19002/metrics",
            },
            clear=True,
        ):
            config = HotLoaderConfig.default()

        self.assertEqual(config.triton_url, "http://127.0.0.1:8000")
        self.assertEqual(config.triton_metrics_url, "http://127.0.0.1:8002/metrics")

    def test_default_config_supports_trt_ip_with_embedded_port_by_taking_host_only(self) -> None:
        with patch("hot_loader._load_dotenv_values", return_value={}), patch.dict(
            "os.environ",
            {
                "TRT_IP": "10.2.20.6:30649",
                "HTTP_PORT": "31648",
                "METRICS_PORT": "31589",
            },
            clear=True,
        ):
            config = HotLoaderConfig.default()

        self.assertEqual(config.triton_url, "http://10.2.20.6:31648")
        self.assertEqual(config.triton_metrics_url, "http://10.2.20.6:31589/metrics")


if __name__ == "__main__":
    unittest.main()
