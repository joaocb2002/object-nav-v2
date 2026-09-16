"""Tests for semantic connected-component labels."""

from __future__ import annotations

import unittest

import numpy as np

from object_nav.perception import annotate_semantic_islands_bgr


class SegmentationLabelTest(unittest.TestCase):
    def test_label_is_drawn_only_on_largest_component(self) -> None:
        labels = np.zeros((120, 240), dtype=np.uint8)
        labels[10:110, 10:170] = 3
        labels[10:35, 200:230] = 3
        colorized = np.zeros((120, 240, 3), dtype=np.uint8)
        colorized[labels == 3] = (30, 180, 220)

        annotated = annotate_semantic_islands_bgr(
            labels,
            colorized,
            class_names={3: "chair"},
        )

        self.assertTrue(np.any(annotated[:, :180] != colorized[:, :180]))
        np.testing.assert_array_equal(annotated[:, 190:], colorized[:, 190:])

    def test_component_that_cannot_fit_label_still_gets_label(self) -> None:
        labels = np.zeros((40, 40), dtype=np.uint8)
        labels[10:20, 10:20] = 3
        colorized = np.zeros((40, 40, 3), dtype=np.uint8)
        colorized[labels == 3] = (30, 180, 220)

        annotated = annotate_semantic_islands_bgr(
            labels,
            colorized,
            class_names={3: "chair"},
        )

        self.assertTrue(np.any(annotated != colorized))

    def test_single_pixel_class_gets_label(self) -> None:
        labels = np.zeros((80, 160), dtype=np.uint8)
        labels[40, 80] = 3
        colorized = np.zeros((80, 160, 3), dtype=np.uint8)
        colorized[40, 80] = (30, 180, 220)

        annotated = annotate_semantic_islands_bgr(
            labels,
            colorized,
            class_names={3: "chair"},
        )

        self.assertTrue(np.any(annotated != colorized))

    def test_label_stays_inside_image_for_border_touching_class(self) -> None:
        labels = np.full((120, 240), 3, dtype=np.uint8)
        colorized = np.full((120, 240, 3), (30, 180, 220), dtype=np.uint8)

        annotated = annotate_semantic_islands_bgr(
            labels,
            colorized,
            class_names={3: "chair"},
        )

        self.assertTrue(np.any(annotated != colorized))
        np.testing.assert_array_equal(annotated[:3], colorized[:3])
        np.testing.assert_array_equal(annotated[-3:], colorized[-3:])
        np.testing.assert_array_equal(annotated[:, :3], colorized[:, :3])
        np.testing.assert_array_equal(annotated[:, -3:], colorized[:, -3:])


if __name__ == "__main__":
    unittest.main()
