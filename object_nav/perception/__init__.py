"""Perception utilities for ObjectNav experiments."""

from importlib import import_module
from typing import Any

from object_nav.perception.config import DEFAULT_YOLO_WEIGHTS_PATH, YoloConfig
from object_nav.perception.detections import Detection, DetectionResult
from object_nav.perception.observations import (
    close_perception_windows,
    depth_to_bgr,
    print_detections,
    print_observations,
    plot_depth_rgb_detections,
    show_depth_rgb_detections,
)

__all__ = [
    "DEFAULT_YOLO_WEIGHTS_PATH",
    "Detection",
    "DetectionResult",
    "YOLODetector",
    "YoloConfig",
    "build_yolo_detector",
    "close_perception_windows",
    "depth_to_bgr",
    "print_detections",
    "print_observations",
    "plot_depth_rgb_detections",
    "show_depth_rgb_detections",
]


def __getattr__(name: str) -> Any:
    if name in {"YOLODetector", "build_yolo_detector"}:
        yolo = import_module("object_nav.perception.yolo")
        return getattr(yolo, name)

    raise AttributeError(f"module 'object_nav.perception' has no attribute {name!r}")
