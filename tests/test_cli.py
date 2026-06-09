from __future__ import annotations

import unittest
from unittest.mock import patch

from cli import build_config_from_args, build_parser


class CliConfigTests(unittest.TestCase):
    def test_build_config_from_args_preserves_default_only_env_fields(self) -> None:
        with patch("hot_loader._load_dotenv_values", return_value={}), patch.dict(
            "os.environ",
            {
                "TRITON_REPOSITORY_PVC": "triton-models-storage",
                "MODEL_TARGET_PATH": "/repository/trt_models",
                "MODEL_COPY_IMAGE_PULL_POLICY": "Always",
                "JOB_TOLERATIONS_JSON": (
                    '[{"key":"gpu","operator":"Exists","effect":"NoSchedule"},'
                    '{"key":"cpu","operator":"Equal","value":"cveng","effect":"NoSchedule"}]'
                ),
            },
            clear=True,
        ):
            args = build_parser().parse_args(["serve"])
            config = build_config_from_args(args)

        self.assertEqual(config.triton_repository_pvc, "triton-models-storage")
        self.assertEqual(config.model_target_path, "/repository/trt_models")
        self.assertEqual(config.job_image_pull_policy, "Always")
        self.assertEqual(
            config.job_tolerations,
            [
                {"key": "gpu", "operator": "Exists", "effect": "NoSchedule"},
                {"key": "cpu", "operator": "Equal", "value": "cveng", "effect": "NoSchedule"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
