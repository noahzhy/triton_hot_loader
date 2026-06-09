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
    _derive_job_staging_root,
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
        self.assertEqual(container["env"][3]["value"], _derive_job_staging_root(self.config.model_target_path))
        self.assertIn('SOURCE_DIR="${MODEL_SOURCE_PATH%/}/${MODEL_NAME}"', container["args"][0])
        self.assertIn('cp -R "${COPY_SOURCE}/." "${STAGING_DIR}/"', container["args"][0])
        self.assertIn('STAGING_DIR="${STAGING_ROOT%/}/${MODEL_NAME}/${JOB_NAME}"', container["args"][0])
        self.assertIn('if mv "${STAGING_DIR}" "${TARGET_DIR}"; then', container["args"][0])
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

    def test_create_model_copy_job_persists_callback_and_hides_token_in_public_state(self) -> None:
        result = self.loader.create_model_copy_job(
            "unit_empty_space_uspg_yolov8",
            "ccr.ccs.tencentyun.com/clobotics/unit-model-init:20260605",
            callback={
                "url": "https://callback.example.com/hook",
                "events": ["terminal"],
                "token": "secret-token",
            },
        )

        raw_state = self.loader._load_state()
        public_state = self.loader.get_managed_state()

        self.assertTrue(result["callback_registered"])
        self.assertEqual(raw_state["jobs"][result["job_name"]]["callback"]["token"], "secret-token")
        self.assertEqual(public_state["jobs"][result["job_name"]]["callback"]["url"], "https://callback.example.com/hook")
        self.assertNotIn("token", public_state["jobs"][result["job_name"]]["callback"])

    def test_terminal_callback_queue_tracks_retry_state(self) -> None:
        result = self.loader.create_model_copy_job(
            "unit_empty_space_uspg_yolov8",
            "ccr.ccs.tencentyun.com/clobotics/unit-model-init:20260605",
            callback={
                "url": "https://callback.example.com/hook",
                "token": "secret-token",
            },
        )

        self.loader._update_job_state(result["job_name"], status="MODEL_READY", detail="ready")
        pending = self.loader.list_pending_terminal_callbacks()

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["job_name"], result["job_name"])
        self.assertEqual(pending[0]["callback"]["token"], "secret-token")

        self.loader.record_terminal_callback_result(
            result["job_name"],
            delivered=False,
            event_id="evt-1",
            error="temporary failure",
            retry_delay_seconds=5,
        )
        raw_state = self.loader._load_state()
        callback = raw_state["jobs"][result["job_name"]]["callback"]

        self.assertEqual(callback["attempts"], 1)
        self.assertEqual(callback["last_event_id"], "evt-1")
        self.assertEqual(callback["last_error"], "temporary failure")
        self.assertTrue(callback["next_attempt_at"])

    def test_derive_model_name_from_image_tag_normalizes_dash(self) -> None:
        derived = self.loader._derive_model_name_from_image_ref(
            "ccr.ccs.tencentyun.com/clobotics/unit-model-init:model-a"
        )

        self.assertEqual(derived, "model_a")

    def test_create_model_copy_job_rejects_unapproved_registry(self) -> None:
        with self.assertRaisesRegex(HotLoaderError, "registry 前缀"):
            self.loader.create_model_copy_job("demo_model", "registry.example.com/demo:1")

    def test_assert_job_capacity_skips_limit_when_disabled(self) -> None:
        self.loader.config = self.loader.config.with_updates(max_concurrent_jobs=0)
        self.loader._active_job_count = lambda: (_ for _ in ()).throw(AssertionError("should not check active jobs"))  # type: ignore[method-assign]

        self.loader._assert_job_capacity()

    def test_assert_job_capacity_rejects_when_positive_limit_is_reached(self) -> None:
        self.loader.config = self.loader.config.with_updates(max_concurrent_jobs=2)
        self.loader._active_job_count = lambda: 2  # type: ignore[method-assign]

        with self.assertRaisesRegex(HotLoaderError, "当前运行中的 Job 数量已达到上限 2"):
            self.loader._assert_job_capacity()

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

    def test_refresh_active_job_statuses_only_polls_recent_active_jobs(self) -> None:
        state = self.loader._empty_state()
        state["jobs"] = {
            "job-old": {"status": "IMAGE_PULLING", "updated_at": "2026-06-09T00:00:00+00:00"},
            "job-new": {"status": "JOB_CREATED", "updated_at": "2026-06-09T00:00:03+00:00"},
            "job-ready": {"status": "MODEL_READY", "updated_at": "2026-06-09T00:00:04+00:00"},
            "job-mid": {"status": "COPY_RUNNING", "updated_at": "2026-06-09T00:00:02+00:00"},
        }
        self.loader._save_state(state)

        polled = []

        def fake_get_job_status(job_name, *, include_logs=True):
            polled.append((job_name, include_logs))
            return {"job_name": job_name, "status": "COPY_RUNNING"}

        self.loader.get_job_status = fake_get_job_status  # type: ignore[method-assign]

        self.loader.refresh_active_job_statuses(limit=2, include_logs=False)
        refreshed_state = self.loader._load_state()

        self.assertEqual(
            polled,
            [
                ("job-new", False),
                ("job-mid", False),
            ],
        )
        self.assertEqual(refreshed_state["jobs"]["job-new"]["updated_at"], "2026-06-09T00:00:03+00:00")
        self.assertEqual(refreshed_state["jobs"]["job-mid"]["updated_at"], "2026-06-09T00:00:02+00:00")
        self.assertIn("status_checked_at", refreshed_state["jobs"]["job-new"])

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

    def test_get_job_status_can_skip_logs_for_pending_jobs(self) -> None:
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
                    conditions=[],
                ),
            )
        ]
        self.core_api.read_namespaced_pod_log = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not read logs"))  # type: ignore[method-assign]

        result = self.loader.get_job_status("model-copy-demo", include_logs=False)

        self.assertEqual(result["status"], "IMAGE_PULLING")
        self.assertIsNone(result["logs"])

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
        self.assertEqual(result["alias"], "model_demo_model")

    def test_get_job_status_rolls_back_managed_state_when_triton_load_fails(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["1"])
        self.loader._register_loaded_model("demo_model", "ccr.ccs.tencentyun.com/clobotics/demo:old")
        write_model_bundle(self.config.model_repository, "demo_model", ["2"])
        self.batch_api.job_to_read = SimpleNamespace(
            metadata=SimpleNamespace(
                annotations={
                    "hot-loader/model-name": "demo_model",
                    "hot-loader/image-ref": "ccr.ccs.tencentyun.com/clobotics/demo:new",
                }
            ),
            status=SimpleNamespace(active=0, failed=0, succeeded=1),
        )
        self.core_api.pods = [SimpleNamespace(metadata=SimpleNamespace(name="demo-pod"), status=SimpleNamespace(phase="Succeeded"))]

        def fail_load(model_name):
            raise HotLoaderError("explicit load failed")

        self.loader._load_model = fail_load  # type: ignore[method-assign]

        result = self.loader.get_job_status("model-copy-demo")
        state = self.loader.get_managed_state()

        self.assertEqual(result["status"], "TRITON_RELOAD_FAILED")
        self.assertEqual(state["managed_images"][0]["image"], "ccr.ccs.tencentyun.com/clobotics/demo:old")
        self.assertEqual(state["managed_images"][0]["models"], ["demo_model"])

    def test_get_job_status_keeps_reloading_until_triton_reports_ready(self) -> None:
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

        load_calls = []
        self.loader._load_model = lambda model_name: load_calls.append(model_name)  # type: ignore[method-assign]
        repository_states = iter(
            [
                [{"name": "demo_model", "version": "2", "state": "UNAVAILABLE", "reason": ""}],
                [{"name": "demo_model", "version": "2", "state": "UNAVAILABLE", "reason": ""}],
                [{"name": "demo_model", "version": "2", "state": "READY", "reason": ""}],
            ]
        )
        self.loader.list_repository_models = lambda safe=True: next(repository_states)  # type: ignore[method-assign]

        first = self.loader.get_job_status("model-copy-demo")
        second = self.loader.get_job_status("model-copy-demo")

        self.assertEqual(first["status"], "TRITON_RELOAD_RUNNING")
        self.assertIn("等待模型变为 READY", first["detail"])
        self.assertEqual(second["status"], "MODEL_READY")
        self.assertEqual(second["triton_reload_attempts"], 2)
        self.assertEqual(load_calls, ["demo_model", "demo_model"])

    def test_get_job_status_continues_reload_after_job_is_ttl_deleted(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["2"])
        self.loader._update_job_state(
            "model-copy-demo",
            status="TRITON_RELOAD_RUNNING",
            model_name="demo_model",
            image="ccr.ccs.tencentyun.com/clobotics/demo:2",
            detail="Triton 已收到 load 请求，正在等待模型变为 READY",
            triton_reload_attempts=1,
        )
        self.batch_api.read_namespaced_job = lambda name, namespace: (_ for _ in ()).throw(RuntimeError("Not Found"))  # type: ignore[method-assign]

        load_calls = []
        self.loader._load_model = lambda model_name: load_calls.append(model_name)  # type: ignore[method-assign]
        repository_states = iter(
            [
                [{"name": "demo_model", "version": "2", "state": "UNAVAILABLE", "reason": ""}],
                [{"name": "demo_model", "version": "2", "state": "READY", "reason": ""}],
            ]
        )
        self.loader.list_repository_models = lambda safe=True: next(repository_states)  # type: ignore[method-assign]

        result = self.loader.get_job_status("model-copy-demo")

        self.assertEqual(result["status"], "MODEL_READY")
        self.assertEqual(result["triton_reload_attempts"], 2)
        self.assertEqual(load_calls, ["demo_model"])

    def test_get_job_status_recovers_when_job_disappears_before_copy_success_is_seen(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["2"])
        self.loader._update_job_state(
            "model-copy-demo",
            status="COPY_RUNNING",
            model_name="demo_model",
            image="ccr.ccs.tencentyun.com/clobotics/demo:2",
            detail="模型复制容器正在运行",
        )
        self.batch_api.read_namespaced_job = lambda name, namespace: (_ for _ in ()).throw(RuntimeError("Not Found"))  # type: ignore[method-assign]

        load_calls = []
        self.loader._load_model = lambda model_name: load_calls.append(model_name)  # type: ignore[method-assign]
        self.loader.list_repository_models = lambda safe=True: [  # type: ignore[method-assign]
            {"name": "demo_model", "version": "2", "state": "READY", "reason": ""}
        ]

        result = self.loader.get_job_status("model-copy-demo")

        self.assertEqual(result["status"], "MODEL_READY")
        self.assertEqual(load_calls, ["demo_model"])

    def test_reload_models_uses_load_only(self) -> None:
        events = []
        self.loader._load_model = lambda model_name: events.append(("load", model_name))  # type: ignore[method-assign]
        self.loader._unload_model = lambda model_name, tolerate_missing=True: events.append(("unload", model_name))  # type: ignore[method-assign]

        result = self.loader.reload_models(["demo_model", "demo_model"])

        self.assertEqual(events, [("load", "demo_model")])
        self.assertEqual(result["reloaded_models"], ["demo_model"])

    def test_register_loaded_model_replaces_existing_same_name_entry(self) -> None:
        write_model_bundle(self.config.model_repository, "demo_model", ["1"])

        first = self.loader._register_loaded_model("demo_model", "ccr.ccs.tencentyun.com/clobotics/demo:old")
        second = self.loader._register_loaded_model("demo_model", "ccr.ccs.tencentyun.com/clobotics/demo:new")
        state = self.loader.get_managed_state()

        self.assertEqual(first["alias"], "model_demo_model")
        self.assertEqual(second["alias"], "model_demo_model")
        self.assertEqual(state["managed_model_count"], 1)
        self.assertEqual(state["managed_images"], [
            {
                "id": "model_demo_model",
                "image": "ccr.ccs.tencentyun.com/clobotics/demo:new",
                "models": ["demo_model"],
                "updated_at": state["managed_images"][0]["updated_at"],
            }
        ])
        self.assertNotIn("managed_model_versions", state)
        self.assertNotIn("managed_active_versions", state)

    def test_unload_model_versions_is_rejected_after_version_management_removed(self) -> None:
        with self.assertRaisesRegex(HotLoaderError, "取消版本管理"):
            self.loader.unload_model_versions(["demo_model@3"])


class HotLoaderDefaultRuntimePathTests(unittest.TestCase):
    def test_derive_job_volume_mount_path_uses_parent_for_nested_target_path(self) -> None:
        self.assertEqual(_derive_job_volume_mount_path("/repository/trt_models"), "/repository")

    def test_derive_job_volume_mount_path_keeps_top_level_target_path(self) -> None:
        self.assertEqual(_derive_job_volume_mount_path("/repository"), "/repository")

    def test_derive_job_staging_root_uses_hidden_sibling_under_mount_path(self) -> None:
        self.assertEqual(_derive_job_staging_root("/repository/trt_models"), "/repository/.staging")

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
