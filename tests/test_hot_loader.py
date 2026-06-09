from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hot_loader import (
    HotLoaderConfig,
    HotLoaderError,
    TritonHotLoader,
    _default_runtime_paths,
    _derive_job_volume_mount_path,
)


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


class FakeBatchApi:
    def __init__(self) -> None:
        self.created_jobs = []
        self.job_to_read = None
        self.list_response = SimpleNamespace(items=[])

    def list_namespaced_job(self, **kwargs):
        return self.list_response

    def create_namespaced_job(self, namespace, body):
        self.created_jobs.append((namespace, body))
        return SimpleNamespace(metadata=SimpleNamespace(uid="job-uid-1"))

    def read_namespaced_job(self, name, namespace):
        if self.job_to_read is None:
            raise AssertionError("job_to_read was not configured")
        return self.job_to_read


class FakeCoreApi:
    def __init__(self) -> None:
        self.pods = []
        self.logs = {}
        self.events = []

    def list_namespaced_pod(self, **kwargs):
        return SimpleNamespace(items=self.pods)

    def read_namespaced_pod_log(self, name, namespace, tail_lines):
        return self.logs.get(name, "")

    def list_namespaced_event(self, **kwargs):
        return SimpleNamespace(items=self.events)


class TritonHotLoaderKubernetesJobTests(unittest.TestCase):
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
            job_tolerations=[
                {"key": "gpu", "operator": "Exists", "effect": "NoSchedule"},
                {"key": "cpu", "operator": "Equal", "value": "cveng", "effect": "NoSchedule"},
            ],
        )
        self.loader = TritonHotLoader(self.config)
        self.batch_api = FakeBatchApi()
        self.core_api = FakeCoreApi()
        self.loader._get_batch_v1_api = lambda: self.batch_api  # type: ignore[method-assign]
        self.loader._get_core_v1_api = lambda: self.core_api  # type: ignore[method-assign]

    def test_create_model_copy_job_builds_kubernetes_manifest_and_records_state(self) -> None:
        result = self.loader.create_model_copy_job(
            "unit_empty_space_uspg_yolov8",
            "ccr.ccs.tencentyun.com/clobotics/unit-model-init:20260605",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "JOB_CREATED")
        namespace, manifest = self.batch_api.created_jobs[0]
        self.assertEqual(namespace, "default")
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["image"], "ccr.ccs.tencentyun.com/clobotics/unit-model-init:20260605")
        self.assertEqual(container["env"][1]["value"], "/trt_models")
        self.assertEqual(container["env"][2]["value"], "/repository/trt_models")
        self.assertIn('SOURCE_DIR="${MODEL_SOURCE_PATH%/}/${MODEL_NAME}"', container["args"][0])
        self.assertIn('cp -R "${COPY_SOURCE}/." "${TARGET_DIR}/"', container["args"][0])
        self.assertEqual(container["volumeMounts"][0]["mountPath"], "/repository")
        self.assertEqual(
            manifest["spec"]["template"]["spec"]["tolerations"],
            [
                {"key": "gpu", "operator": "Exists", "effect": "NoSchedule"},
                {"key": "cpu", "operator": "Equal", "value": "cveng", "effect": "NoSchedule"},
            ],
        )
        self.assertEqual(
            manifest["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"],
            "triton-repository-pvc",
        )

        state = self.loader.get_managed_state()
        self.assertIn(result["job_name"], state["jobs"])
        self.assertEqual(state["jobs"][result["job_name"]]["model_name"], "unit_empty_space_uspg_yolov8")

    def test_create_model_copy_job_derives_model_name_from_image_tag(self) -> None:
        result = self.loader.create_model_copy_job(
            "",
            "ccr.ccs.tencentyun.com/clobotics/unit-model-init:unit_empty_space_uspg_yolov8-20260430",
        )

        self.assertEqual(result["model_name"], "unit_empty_space_uspg_yolov8")
        _, manifest = self.batch_api.created_jobs[0]
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["env"][0]["value"], "unit_empty_space_uspg_yolov8")

    def test_derive_model_name_from_image_tag_normalizes_dash(self) -> None:
        derived = self.loader._derive_model_name_from_image_ref(
            "ccr.ccs.tencentyun.com/clobotics/unit-model-init:model-a"
        )

        self.assertEqual(derived, "model_a")

    def test_create_model_copy_job_rejects_unapproved_registry(self) -> None:
        with self.assertRaisesRegex(HotLoaderError, "registry 前缀"):
            self.loader.create_model_copy_job("demo_model", "registry.example.com/demo:1")

    def test_load_models_from_images_sync_derives_model_name_when_missing(self) -> None:
        captured = {}

        def fake_create_model_copy_job_and_wait(model_name, image_ref, **kwargs):
            captured["model_name"] = model_name
            captured["image"] = image_ref
            return {"success": True, "job_name": "job-a", "model_name": model_name, "status": "MODEL_READY"}

        self.loader.create_model_copy_job_and_wait = fake_create_model_copy_job_and_wait  # type: ignore[method-assign]

        result = self.loader.load_models_from_images_sync(
            [
                {
                    "image": "ccr.ccs.tencentyun.com/clobotics/unit-model-init:model-a",
                }
            ]
        )

        self.assertTrue(result["success"])
        self.assertEqual(captured["model_name"], "model_a")
        self.assertEqual(result["completed"][0]["model_name"], "model_a")

    def test_wait_for_job_terminal_state_polls_until_model_ready(self) -> None:
        responses = iter(
            [
                {"job_name": "model-copy-demo", "status": "JOB_CREATED", "detail": "created"},
                {"job_name": "model-copy-demo", "status": "COPY_RUNNING", "detail": "copying"},
                {"job_name": "model-copy-demo", "status": "MODEL_READY", "detail": "ready"},
            ]
        )
        self.loader.get_job_status = lambda job_name: next(responses)  # type: ignore[method-assign]

        with patch("hot_loader.time.sleep", return_value=None) as sleep_mock:
            result = self.loader.wait_for_job_terminal_state(
                "model-copy-demo",
                timeout_seconds=5,
                poll_interval_seconds=0,
            )

        self.assertEqual(result["status"], "MODEL_READY")
        self.assertEqual(sleep_mock.call_count, 2)

    def test_get_job_status_marks_image_pull_failure(self) -> None:
        self.batch_api.job_to_read = SimpleNamespace(
            metadata=SimpleNamespace(
                annotations={
                    "hot-loader/model-name": "demo_model",
                    "hot-loader/image-ref": "ccr.ccs.tencentyun.com/clobotics/demo:1",
                }
            ),
            status=SimpleNamespace(active=0, failed=0, succeeded=0),
        )
        self.core_api.pods = [
            SimpleNamespace(
                metadata=SimpleNamespace(name="demo-pod"),
                status=SimpleNamespace(
                    phase="Pending",
                    container_statuses=[
                        SimpleNamespace(
                            state=SimpleNamespace(
                                waiting=SimpleNamespace(
                                    reason="ImagePullBackOff",
                                    message="Back-off pulling image",
                                )
                            )
                        )
                    ],
                ),
            )
        ]

        result = self.loader.get_job_status("model-copy-demo")

        self.assertEqual(result["status"], "COPY_FAILED")
        self.assertIn("镜像拉取失败", result["detail"])
        self.assertEqual(result["pod_name"], "demo-pod")

    def test_get_job_status_marks_unschedulable_pod_as_scheduling(self) -> None:
        self.batch_api.job_to_read = SimpleNamespace(
            metadata=SimpleNamespace(
                annotations={
                    "hot-loader/model-name": "demo_model",
                    "hot-loader/image-ref": "ccr.ccs.tencentyun.com/clobotics/demo:1",
                }
            ),
            status=SimpleNamespace(active=0, failed=0, succeeded=0),
        )
        self.core_api.pods = [
            SimpleNamespace(
                metadata=SimpleNamespace(name="demo-pod"),
                status=SimpleNamespace(
                    phase="Pending",
                    container_statuses=[],
                    conditions=[
                        SimpleNamespace(
                            type="PodScheduled",
                            status="False",
                            reason="Unschedulable",
                            message="0/4 nodes are available: 4 Insufficient cpu.",
                        )
                    ],
                ),
            )
        ]
        self.core_api.events = [
            SimpleNamespace(
                type="Warning",
                reason="FailedScheduling",
                message="0/4 nodes are available: 4 Insufficient cpu.",
                count=8,
            )
        ]

        result = self.loader.get_job_status("model-copy-demo")

        self.assertEqual(result["status"], "SCHEDULING")
        self.assertIn("Insufficient cpu", result["detail"])
        self.assertEqual(result["events"][0]["reason"], "FailedScheduling")

    def test_get_job_status_finalizes_triton_load_after_copy_success(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["2"])
        self.batch_api.job_to_read = SimpleNamespace(
            metadata=SimpleNamespace(
                annotations={
                    "hot-loader/model-name": "demo_model",
                    "hot-loader/image-ref": "ccr.ccs.tencentyun.com/clobotics/demo:2",
                }
            ),
            status=SimpleNamespace(active=0, failed=0, succeeded=1),
        )
        self.core_api.pods = [SimpleNamespace(metadata=SimpleNamespace(name="demo-pod"), status=SimpleNamespace(phase="Succeeded"))]
        self.core_api.logs = {"demo-pod": "model copy done"}

        events = []
        self.loader._load_model = lambda model_name: events.append(model_name)  # type: ignore[method-assign]
        self.loader.list_repository_models = lambda safe=True: [  # type: ignore[method-assign]
            {"name": "demo_model", "version": "2", "state": "READY", "reason": ""}
        ]

        result = self.loader.get_job_status("model-copy-demo")
        state = self.loader.get_managed_state()

        self.assertEqual(events, ["demo_model"])
        self.assertEqual(result["status"], "MODEL_READY")
        self.assertIn("model copy done", result["logs"])
        self.assertTrue(any(item["image"] == "ccr.ccs.tencentyun.com/clobotics/demo:2" for item in state["managed_images"]))

    def test_reload_models_uses_load_only(self) -> None:
        events = []
        self.loader._load_model = lambda model_name: events.append(("load", model_name))  # type: ignore[method-assign]
        self.loader._unload_model = lambda model_name, tolerate_missing=True: events.append(("unload", model_name))  # type: ignore[method-assign]

        result = self.loader.reload_models(["demo_model", "demo_model"])

        self.assertEqual(events, [("load", "demo_model")])
        self.assertEqual(result["reloaded_models"], ["demo_model"])

    def test_unload_model_versions_removes_selected_version_and_reloads_remaining(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["1", "2", "3"])
        self.loader._save_state(
            {
                "aliases": {
                    "demo_alias": {
                        "image": "ccr.ccs.tencentyun.com/clobotics/demo:multi",
                        "models": ["demo_model"],
                        "model_versions": {"demo_model": ["1", "2", "3"]},
                        "active_versions": {"demo_model": "3"},
                        "updated_at": "2026-06-03T00:00:00+00:00",
                    }
                },
                "jobs": {},
                "updated_at": "2026-06-03T00:00:00+00:00",
            }
        )

        events = []
        self.loader._load_model = lambda model_name: events.append(("load", model_name))  # type: ignore[method-assign]
        self.loader._unload_model = lambda model_name, tolerate_missing=True: events.append(("unload", model_name))  # type: ignore[method-assign]

        result = self.loader.unload_model_versions(["demo_model@3"])
        state = self.loader.get_managed_state()

        self.assertEqual(events, [("load", "demo_model")])
        self.assertEqual(state["managed_active_versions"], {"demo_model": "2"})
        self.assertEqual(result["removed_versions"][0]["remaining_versions"], ["1", "2"])


class HotLoaderDefaultRuntimePathTests(unittest.TestCase):
    def test_derive_job_volume_mount_path_uses_parent_for_nested_target_path(self) -> None:
        self.assertEqual(_derive_job_volume_mount_path("/repository/trt_models"), "/repository")

    def test_derive_job_volume_mount_path_keeps_top_level_target_path(self) -> None:
        self.assertEqual(_derive_job_volume_mount_path("/repository"), "/repository")

    def test_default_runtime_paths_prefer_hot_triton_repository_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "runtime"
            repository_path = Path(temp_dir) / "repository"

            with patch.dict(os.environ, {"HOT_TRITON_MODEL_REPOSITORY": str(repository_path)}, clear=False):
                model_repository, state_file, staging_root = _default_runtime_paths(base_dir)

        self.assertEqual(model_repository, repository_path)
        self.assertEqual(state_file, repository_path / ".hot_loader" / "state.json")
        self.assertEqual(staging_root, repository_path / ".staging")

    def test_default_runtime_paths_fall_back_to_model_target_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "runtime"
            repository_path = Path(temp_dir) / "repository"

            with patch.dict(
                os.environ,
                {
                    "MODEL_TARGET_PATH": str(repository_path),
                    "HOT_TRITON_MODEL_REPOSITORY": "",
                },
                clear=False,
            ):
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

    def test_default_config_reads_job_tolerations_from_environment(self) -> None:
        with patch("hot_loader._load_dotenv_values", return_value={}), patch.dict(
            "os.environ",
            {
                "JOB_TOLERATIONS_JSON": (
                    '[{"key":"gpu","operator":"Exists","effect":"NoSchedule"},'
                    '{"key":"cpu","operator":"Equal","value":"cveng","effect":"NoSchedule"}]'
                )
            },
            clear=True,
        ):
            config = HotLoaderConfig.default()

        self.assertEqual(
            config.job_tolerations,
            [
                {"key": "gpu", "operator": "Exists", "effect": "NoSchedule"},
                {"key": "cpu", "operator": "Equal", "value": "cveng", "effect": "NoSchedule"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
