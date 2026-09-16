"""Source-level contracts for the intentionally simple active entrypoint."""

from __future__ import annotations

import unittest
from pathlib import Path


MAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/main.py"


class MainScriptContractTest(unittest.TestCase):
    def test_voxel_dashboard_views_are_independent(self) -> None:
        source = MAIN_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"3D voxel view"', source)
        self.assertIn('"Voxel top-down map"', source)
        self.assertIn("voxel_mapper.render_camera_view(", source)
        self.assertIn("voxel_mapper.render_topdown_map(", source)
        self.assertNotIn("voxel_mapper.render_maps(", source)

    def test_dashboard_opens_before_segformer_loads(self) -> None:
        source = MAIN_SCRIPT.read_text(encoding="utf-8")

        self.assertLess(
            source.index("dashboard.open()"),
            source.index("build_segformer_segmenter("),
        )


if __name__ == "__main__":
    unittest.main()
