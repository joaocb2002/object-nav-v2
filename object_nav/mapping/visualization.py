from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from object_nav.mapping.voxel import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    CameraIntrinsics,
    SparseVoxelMap,
    TopDownGrid,
)

VOXEL_MAP_COLORS_BGR = {
    UNKNOWN: np.array([45, 45, 45], dtype=np.uint8),
    FREE: np.array([215, 215, 215], dtype=np.uint8),
    OCCUPIED: np.array([40, 40, 220], dtype=np.uint8),
}


def show_navigation_maps(
    image_bgr: np.ndarray,
    *,
    window_name: str = "Voxel map + Habitat map",
) -> None:
    """Show voxel and Habitat top-down maps in one OpenCV window."""
    cv2.imshow(window_name, image_bgr)


def render_voxel_topdown_from_agent_bgr(
    topdown: TopDownGrid,
    agent_state: object,
    *,
    view_size_m: float,
    pixels_per_meter: int,
    output_height: int,
) -> np.ndarray:
    """Render an egocentric top-down voxel map with the robot centered."""
    size_px = max(1, int(round(view_size_m * pixels_per_meter)))
    image = np.empty((size_px, size_px, 3), dtype=np.uint8)
    image[:] = VOXEL_MAP_COLORS_BGR[UNKNOWN]

    if topdown.data.size > 0:
        agent_position = np.asarray(agent_state.position, dtype=np.float64)
        right, forward = _agent_horizontal_basis(agent_state)
        center = (size_px - 1) / 2.0

        cols = np.arange(size_px, dtype=np.float64)
        rows = np.arange(size_px, dtype=np.float64)
        local_right = (cols - center) / pixels_per_meter
        local_forward = (center - rows) / pixels_per_meter
        world_x = (
            agent_position[0]
            + np.outer(np.ones(size_px), local_right) * right[0]
            + np.outer(local_forward, np.ones(size_px)) * forward[0]
        )
        world_z = (
            agent_position[2]
            + np.outer(np.ones(size_px), local_right) * right[1]
            + np.outer(local_forward, np.ones(size_px)) * forward[1]
        )
        col_index = np.floor(
            (world_x - topdown.origin[0]) / topdown.resolution
        ).astype(np.int32)
        row_index = np.floor(
            (world_z - topdown.origin[1]) / topdown.resolution
        ).astype(np.int32)
        valid = (
            (row_index >= 0)
            & (row_index < topdown.data.shape[0])
            & (col_index >= 0)
            & (col_index < topdown.data.shape[1])
        )
        sampled = np.full((size_px, size_px), UNKNOWN, dtype=np.int8)
        sampled[valid] = topdown.data[row_index[valid], col_index[valid]]
        image = colorize_voxel_grid_bgr(sampled)

    _draw_agent_marker(image)
    _draw_label(image, "Voxel map")
    _draw_legend(
        image,
        [
            ("free", tuple(int(v) for v in VOXEL_MAP_COLORS_BGR[FREE])),
            ("occupied", tuple(int(v) for v in VOXEL_MAP_COLORS_BGR[OCCUPIED])),
        ],
    )
    return _fit_to_height(image, output_height)


def render_voxel_camera_view_bgr(
    voxel_map: SparseVoxelMap,
    T_world_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    image_shape: tuple[int, int],
    output_height: int,
    max_depth: Optional[float] = None,
) -> np.ndarray:
    """Render observed 3D voxels from the robot camera perspective."""
    height, width = image_shape
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    observed = list(voxel_map.iter_observed_voxels())
    if not observed:
        _draw_label(image, "3D voxel view")
        return _fit_to_height(image, output_height)

    T_camera_world = np.linalg.inv(T_world_camera)
    projected = []
    for index, _ in observed:
        center_world = voxel_map.voxel_index_to_world_center(index)
        center_camera = T_camera_world @ np.array(
            [center_world[0], center_world[1], center_world[2], 1.0],
            dtype=np.float64,
        )
        x, y, z = center_camera[:3]
        if z <= 0.0 or (max_depth is not None and z > max_depth):
            continue

        u = int(round(intrinsics.fx * x / z + intrinsics.cx))
        v = int(round(intrinsics.fy * y / z + intrinsics.cy))
        if u < 0 or u >= width or v < 0 or v >= height:
            continue

        radius = int(
            round(voxel_map.voxel_size * intrinsics.fx / max(z, 0.01) * 0.5)
        )
        radius = max(1, min(radius, 5))
        color = _voxel_color_bgr(voxel_map, index, z, max_depth=max_depth)
        projected.append((z, u, v, radius, color))

    for _, u, v, radius, color in sorted(projected, reverse=True):
        cv2.rectangle(
            image,
            (max(0, u - radius), max(0, v - radius)),
            (min(width - 1, u + radius), min(height - 1, v + radius)),
            color,
            thickness=-1,
        )

    _draw_label(image, "3D voxel view")
    _draw_legend(
        image,
        [
            ("free", (210, 210, 210)),
            ("occupied", (35, 35, 235)),
        ],
    )
    return _fit_to_height(image, output_height)


def render_ground_truth_topdown_bgr(
    topdown_metric: Optional[dict],
    *,
    output_height: int,
) -> np.ndarray:
    """Render Habitat's TopDownMap metric as BGR for OpenCV display."""
    if topdown_metric is None:
        image = np.full((output_height, output_height, 3), 35, dtype=np.uint8)
        _draw_label(image, "Habitat map unavailable")
        return image

    from habitat.utils.visualizations import maps

    image_rgb = maps.colorize_draw_agent_and_fit_to_height(
        topdown_metric,
        output_height,
    )
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _draw_label(image_bgr, "Habitat map")
    return image_bgr


def _voxel_color_bgr(
    voxel_map: SparseVoxelMap,
    index: tuple[int, int, int],
    depth: float,
    *,
    max_depth: Optional[float],
) -> tuple[int, int, int]:
    if voxel_map.is_occupied(index):
        base = np.array([35, 35, 235], dtype=np.float32)
    elif voxel_map.is_free(index):
        base = np.array([210, 210, 210], dtype=np.float32)
    else:
        base = np.array([40, 210, 210], dtype=np.float32)

    if max_depth is None or max_depth <= 0.0:
        scale = 1.0
    else:
        scale = 1.0 - min(max(depth / max_depth, 0.0), 1.0) * 0.55
    color = np.clip(base * scale, 0, 255).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def colorize_voxel_grid_bgr(grid: np.ndarray) -> np.ndarray:
    """Convert a voxel top-down state grid to a BGR image."""
    image = np.empty((*grid.shape, 3), dtype=np.uint8)
    for value, color in VOXEL_MAP_COLORS_BGR.items():
        image[grid == value] = color
    return image


def _agent_horizontal_basis(agent_state: object) -> tuple[np.ndarray, np.ndarray]:
    import quaternion

    rotation = quaternion.as_rotation_matrix(agent_state.rotation)
    right_3d = rotation @ np.array([1.0, 0.0, 0.0])
    forward_3d = rotation @ np.array([0.0, 0.0, -1.0])
    right = _normalize_2d(
        np.array([right_3d[0], right_3d[2]], dtype=np.float64)
    )
    forward = _normalize_2d(
        np.array([forward_3d[0], forward_3d[2]], dtype=np.float64)
    )
    return right, forward


def _normalize_2d(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _draw_agent_marker(image: np.ndarray) -> None:
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = max(4, min(image.shape[:2]) // 32)
    points = np.array(
        [
            (center[0], center[1] - radius * 2),
            (center[0] - radius, center[1] + radius),
            (center[0] + radius, center[1] + radius),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, points, (0, 190, 255))
    cv2.polylines(image, [points], True, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_label(image: np.ndarray, text: str) -> None:
    cv2.putText(
        image,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_legend(
    image: np.ndarray,
    items: list[tuple[str, tuple[int, int, int]]],
) -> None:
    x = 12
    row_height = 15
    width = 92
    height = row_height * len(items) + 8
    y = image.shape[0] - height - 12
    cv2.rectangle(image, (x - 5, y - 9), (x + width, y + height), (25, 25, 25), -1)
    cv2.rectangle(image, (x - 5, y - 9), (x + width, y + height), (140, 140, 140), 1)

    for row, (text, color) in enumerate(items):
        top = y + row * row_height
        cv2.rectangle(image, (x, top), (x + 8, top + 8), color, -1)
        cv2.putText(
            image,
            text,
            (x + 13, top + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def _fit_to_height(image: np.ndarray, output_height: int) -> np.ndarray:
    if image.shape[0] == output_height:
        return image
    width = int(round(image.shape[1] * output_height / image.shape[0]))
    return cv2.resize(image, (width, output_height), interpolation=cv2.INTER_AREA)
