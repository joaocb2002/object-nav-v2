"""Tests for the configurable OpenCV dashboard."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from object_nav.utils import (
    DashboardConfig,
    OpenCVDashboard,
    blend_bgr_overlay,
    compose_dashboard_bgr,
)


class DashboardTest(unittest.TestCase):
    def test_overlay_opacity_endpoints(self) -> None:
        image = np.full((2, 2, 3), 20, dtype=np.uint8)
        overlay = np.full((2, 2, 3), 220, dtype=np.uint8)

        np.testing.assert_array_equal(
            blend_bgr_overlay(image, overlay, opacity=0.0), image
        )
        np.testing.assert_array_equal(
            blend_bgr_overlay(image, overlay, opacity=1.0), overlay
        )

    def test_overlay_uses_requested_opacity(self) -> None:
        image = np.full((2, 2, 3), 20, dtype=np.uint8)
        overlay = np.full((2, 2, 3), 220, dtype=np.uint8)

        blended = blend_bgr_overlay(image, overlay, opacity=0.25)

        np.testing.assert_array_equal(
            blended,
            np.full((2, 2, 3), 70, dtype=np.uint8),
        )

    def test_overlay_rejects_invalid_opacity(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            blend_bgr_overlay(image, image, opacity=1.1)

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
    @patch("object_nav.utils.visualization.cv2.waitKey")
    @patch("object_nav.utils.visualization.cv2.setWindowProperty")
    @patch("object_nav.utils.visualization.cv2.namedWindow")
    def test_dashboard_only_evaluates_enabled_sources(
        self,
        named_window: object,
        set_window_property: object,
        wait_key: object,
        imshow: object,
    ) -> None:
        del named_window, wait_key
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
        self.assertTrue(set_window_property.called)
        self.assertTrue(imshow.called)

    @patch("object_nav.utils.visualization.cv2.waitKey")
    @patch("object_nav.utils.visualization.cv2.imshow")
    @patch("object_nav.utils.visualization.cv2.setWindowProperty")
    @patch("object_nav.utils.visualization.cv2.namedWindow")
    def test_dashboard_can_initialize_before_first_panel(
        self,
        named_window: object,
        set_window_property: object,
        imshow: object,
        wait_key: object,
    ) -> None:
        dashboard = OpenCVDashboard(
            DashboardConfig(enabled_panels=("RGB",))
        )

        dashboard.open()
        dashboard.open()

        self.assertEqual(named_window.call_count, 1)
        self.assertEqual(set_window_property.call_count, 1)
        self.assertEqual(imshow.call_count, 1)
        self.assertEqual(wait_key.call_count, 1)


if __name__ == "__main__":
    unittest.main()
