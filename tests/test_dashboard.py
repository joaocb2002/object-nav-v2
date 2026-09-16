"""Tests for the configurable OpenCV dashboard."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from object_nav.utils import DashboardConfig, OpenCVDashboard, compose_dashboard_bgr


class DashboardTest(unittest.TestCase):
    def test_compose_uses_enabled_order_and_grid_shape(self) -> None:
        config = DashboardConfig(
            enabled_panels=("Blue", "Green"),
            columns=2,
            tile_width=20,
            tile_height=40,
        )
        blue = np.full((40, 20, 3), (100, 0, 0), dtype=np.uint8)
        green = np.full((40, 20, 3), (0, 100, 0), dtype=np.uint8)

        canvas = compose_dashboard_bgr(
            {"Green": green, "Blue": blue},
            config=config,
        )

        self.assertEqual(canvas.shape, (40, 40, 3))
        np.testing.assert_array_equal(canvas[35, 5], (100, 0, 0))
        np.testing.assert_array_equal(canvas[35, 25], (0, 100, 0))

    @patch("object_nav.utils.visualization.cv2.imshow")
    @patch("object_nav.utils.visualization.cv2.namedWindow")
    def test_dashboard_only_evaluates_enabled_sources(
        self,
        named_window: object,
        imshow: object,
    ) -> None:
        del named_window
        calls = []
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        dashboard = OpenCVDashboard(
            DashboardConfig(
                enabled_panels=("RGB",),
                tile_width=10,
                tile_height=10,
            )
        )

        dashboard.show(
            {
                "RGB": lambda: calls.append("RGB") or image,
                "Disabled": lambda: calls.append("Disabled") or image,
            }
        )

        self.assertEqual(calls, ["RGB"])
        self.assertTrue(imshow.called)


if __name__ == "__main__":
    unittest.main()
