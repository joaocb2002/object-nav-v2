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
) -> None:
    """Show depth and YOLO detections with OpenCV."""
    cv2.imshow(window_name, plot_depth_rgb_detections(rgb, depth, result))


def show_rgb_observation(
    rgb: np.ndarray,
    *,
    window_name: str = "RGB",
) -> None:
    """Show one Habitat RGB observation in a separate OpenCV window."""
    cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def show_depth_observation(
    depth: np.ndarray | None,
    *,
    window_name: str = "Depth",
) -> None:
    """Show one Habitat depth observation in a separate OpenCV window."""
    if depth is not None:
        cv2.imshow(window_name, depth_to_bgr(depth))


def colorize_segmentation_bgr(labels: np.ndarray) -> np.ndarray:
    """Colorize a 2D MPCAT40 model-ID mask for OpenCV display."""
    mask = np.asarray(labels)
    if mask.ndim != 2:
        raise ValueError(f"labels must be a 2D array, got shape {mask.shape}")

    try:
        from hm3d_semseg.visualization import colorize_mask
    except ImportError as exc:
        raise ImportError("The hm3d-semseg visualization utilities are required.") from exc

    segmentation_rgb = colorize_mask(mask)
    return cv2.cvtColor(segmentation_rgb, cv2.COLOR_RGB2BGR)


def annotate_semantic_islands_bgr(
    labels: np.ndarray,
    colorized_bgr: np.ndarray,
    *,
    class_names: Mapping[int, str] | None = None,
    min_component_area: int = 1,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Label every visible class near its largest connected component."""
    mask = np.asarray(labels)
    image = np.asarray(colorized_bgr)
    if mask.ndim != 2:
        raise ValueError(f"labels must be a 2D array, got shape {mask.shape}")
    if image.dtype != np.uint8 or image.shape != (*mask.shape, 3):
        raise ValueError("colorized_bgr must be a uint8 HxWx3 image matching labels.")
    if min_component_area <= 0:
        raise ValueError("min_component_area must be positive.")
    if font_scale <= 0.0:
        raise ValueError("font_scale must be positive.")

    names = class_names or _default_semantic_class_names()
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    inner_thickness = 1
    outer_thickness = 3

    for class_id in np.unique(mask):
        class_id = int(class_id)
        if class_id == 255 or class_id not in names:
            continue

        component_mask = (mask == class_id).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            component_mask,
            connectivity=8,
        )
        if count <= 1:
            continue
        component_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if int(stats[component_id, cv2.CC_STAT_AREA]) < min_component_area:
            continue

        component = (components == component_id).astype(np.uint8)
        padded = cv2.copyMakeBorder(
            component,
            1,
            1,
            1,
            1,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        _, _, _, center = cv2.minMaxLoc(distance)
        text = str(names[class_id]).replace("_", " ")
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            inner_thickness,
        )
        center_x, center_y = center
        origin_x = int(round(center_x - text_width / 2.0))
        origin_y = int(round(center_y + (text_height - baseline) / 2.0))
        origin_x = int(
            np.clip(
                origin_x,
                outer_thickness,
                max(outer_thickness, image.shape[1] - text_width - outer_thickness),
            )
        )
        origin_y = int(
            np.clip(
                origin_y,
                text_height + outer_thickness,
                max(
                    text_height + outer_thickness,
                    image.shape[0] - baseline - outer_thickness,
                ),
            )
        )
        origin = (origin_x, origin_y)
        class_color = tuple(int(value) for value in image[component > 0][0])
        cv2.putText(
            annotated,
            text,
            origin,
            font,
            font_scale,
            (0, 0, 0),
            outer_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            origin,
            font,
            font_scale,
            class_color,
            inner_thickness,
            cv2.LINE_AA,
        )

    return annotated


def _default_semantic_class_names() -> Mapping[int, str]:
    try:
        from hm3d_semseg.taxonomy import ID2LABEL
    except ImportError as exc:
        raise ImportError("The hm3d-semseg taxonomy is required for class labels.") from exc
    return ID2LABEL


def show_segmentation(
    labels: np.ndarray,
    *,
    window_name: str = "SegFormer segmentation",
) -> None:
    """Show a SegFormer class-label image in a separate OpenCV window."""
    cv2.imshow(window_name, colorize_segmentation_bgr(labels))


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
