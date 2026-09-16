"""Optional integration checks for stock and patched YOLO behavior."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = PROJECT_ROOT / "scripts/tools/check_yolo_contract.py"
YOLO_WEIGHTS = PROJECT_ROOT / "models/yolo/yolo11x.pt"
YOLO_AVAILABLE = importlib.util.find_spec("ultralytics") is not None


@unittest.skipUnless(
    YOLO_AVAILABLE and YOLO_WEIGHTS.is_file(),
    "Ultralytics and local YOLO weights are required",
)
class YoloContractIntegrationTest(unittest.TestCase):
    def test_upstream_behavior(self) -> None:
        self._run_mode("upstream")

    def test_repository_patched_behavior(self) -> None:
        self._run_mode("patched")

    def _run_mode(self, mode: str) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(CHECK_SCRIPT),
                "--mode",
                mode,
                "--device",
                "cpu",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
