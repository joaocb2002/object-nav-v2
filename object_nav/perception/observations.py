from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from object_nav.perception.detections import DetectionResult


def print_observations(
    observations: Mapping[str, Any],
    *,
    skip_images: bool = True,
) -> None:
    """Print Habitat observation values."""
    print("Observations:")
    for key, value in observations.items():
        if skip_images and key in {"rgb", "depth"}:
            continue
        print(f"  {key}: {_format_value(value)}")


def print_detections(result: DetectionResult) -> None:
    """Print YOLO detections, including softmax probability vectors."""
    boxes = getattr(result.yolo_result, "boxes", None)
    names = getattr(result.yolo_result, "names", {})

    print("Detections:")
    print(f"  raw_result_type={type(result.yolo_result).__name__}")
    print(f"  raw_boxes={0 if boxes is None else len(boxes)}")
    print(f"  class_names={len(names)}")

    if not result.detections:
        print("  none")
        return

    for det in result.detections:
        xyxy = ", ".join(f"{value:.2f}" for value in det.xyxy)
        probs = (
            ", ".join(f"{prob:.3f}" for prob in det.probs)
            if det.probs is not None
            else "N/A"
        )
        print(
            f"  class={det.cls_name} id={det.cls_id} conf={det.conf:.3f} "
            f"scale={det.scale:.3f} box=({xyxy})"
        )
        print(f"  probs=({probs})")


def plot_depth_rgb_detections(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    result: DetectionResult,
) -> np.ndarray:
    """Build a side-by-side depth and YOLO detection image."""
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detections_bgr = result.yolo_result.plot(
        conf=True,
        labels=True,
        boxes=True,
        img=image_bgr,
    )

    if depth is None:
        return detections_bgr

    depth_bgr = depth_to_bgr(depth)
    if depth_bgr.shape[:2] != detections_bgr.shape[:2]:
        depth_bgr = cv2.resize(
            depth_bgr,
            (detections_bgr.shape[1], detections_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return np.hstack((depth_bgr, detections_bgr))


def show_depth_rgb_detections(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    result: DetectionResult,
    *,
    window_name: str = "Depth + RGB detections",
    delay_ms: int = 250,
) -> None:
    """Show depth and YOLO detections with OpenCV."""
    cv2.imshow(window_name, plot_depth_rgb_detections(rgb, depth, result))
    cv2.waitKey(delay_ms)


def close_perception_windows() -> None:
    """Close OpenCV windows opened by perception display helpers."""
    cv2.destroyAllWindows()


def depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    """Convert a Habitat depth observation to a colored BGR depth map."""
    depth_2d = np.squeeze(depth)
    depth_norm = cv2.normalize(depth_2d, None, 0, 255, cv2.NORM_MINMAX)
    depth_u8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)


def _format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=3, threshold=20)
    return str(value)
