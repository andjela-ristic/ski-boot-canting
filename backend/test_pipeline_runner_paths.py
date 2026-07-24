from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object))
sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {}))

capture_readiness_module = types.ModuleType("capture_readiness")
capture_readiness_module.FrameValidator = object
capture_readiness_module.load_config = lambda: {}
sys.modules.setdefault("capture_readiness", capture_readiness_module)

capture_readiness_validator_module = types.ModuleType("capture_readiness.validator")
capture_readiness_validator_module.FrameValidationError = Exception
sys.modules.setdefault("capture_readiness.validator", capture_readiness_validator_module)

from api.pipeline_runner import PROJECT_ROOT, PipelineRunner


class PipelineRunnerPathResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = PipelineRunner()
        self.expected_image = PROJECT_ROOT / "data" / "working_png" / "IMG_0505.png"
        self.windows_image_path = str(
            Path("C:/Users/panonit/Documents/ml-ski-boot-canting/backend/data/working_png/IMG_0505.png")
        )

    def test_windows_absolute_path_resolves_to_repo_file(self) -> None:
        resolved = self.runner._resolve_existing_path(self.windows_image_path)

        self.assertEqual(resolved, self.expected_image.resolve())

    def test_embedded_windows_absolute_path_resolves_to_repo_file(self) -> None:
        dockerized_path = f"/app/{self.windows_image_path}"

        resolved = self.runner._resolve_existing_path(dockerized_path)

        self.assertEqual(resolved, self.expected_image.resolve())


if __name__ == "__main__":
    unittest.main()
