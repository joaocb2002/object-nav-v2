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
    window_name: str = "Voxel maps",
) -> None:
    """Show one navigation visualization image in an OpenCV window."""
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


def render_full_voxel_topdown_from_agent_bgr(
    topdown: TopDownGrid,
    agent_state: object,
    *,
    output_height: int,
    label: str = "Voxel map",
) -> np.ndarray:
    """Render the whole top-down map in robot-local orientation."""
    if topdown.data.size == 0:
        image = np.full((output_height, output_height, 3), 45, dtype=np.uint8)
        _draw_label(image, label)
        return image

    agent_position = np.asarray(agent_state.position, dtype=np.float64)
    right, forward = _agent_horizontal_basis(agent_state)
    local_bounds = _topdown_local_bounds(topdown, agent_position, right, forward)
    sampled = _sample_topdown_in_agent_frame(
        topdown,
        agent_position,
        right,
        forward,
        local_bounds,
    )
    image = colorize_voxel_grid_bgr(sampled)
    resized, scale = _fit_to_height_nearest(image, output_height)
    center = _agent_center_in_local_bounds(local_bounds, topdown.resolution, scale)
    _draw_agent_marker(resized, center=center)
    _draw_label(resized, label)
    _draw_legend(
        resized,
        [
            ("free", tuple(int(v) for v in VOXEL_MAP_COLORS_BGR[FREE])),
            ("occupied", tuple(int(v) for v in VOXEL_MAP_COLORS_BGR[OCCUPIED])),
        ],
    )
    return resized


def render_full_voxel_topdown_bgr(
    topdown: TopDownGrid,
    agent_state: object,
    *,
    output_height: int,
) -> np.ndarray:
    """Render the full voxel-derived top-down map with the robot pose."""
    if topdown.data.size == 0:
        image = np.full((output_height, output_height, 3), 45, dtype=np.uint8)
        _draw_label(image, "Voxel map")
        return image

    image = colorize_voxel_grid_bgr(np.flipud(topdown.data))
    resized, scale = _fit_to_height_nearest(image, output_height)
    center, direction = _agent_marker_in_topdown(agent_state, topdown, scale)
    _draw_agent_marker(resized, center=center, direction=direction)
    _draw_label(resized, "Voxel map")
    _draw_legend(
        resized,
        [
            ("free", tuple(int(v) for v in VOXEL_MAP_COLORS_BGR[FREE])),
            ("occupied", tuple(int(v) for v in VOXEL_MAP_COLORS_BGR[OCCUPIED])),
        ],
    )
    return resized


def render_voxel_camera_view_bgr(
    voxel_map: SparseVoxelMap,
    T_world_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    image_shape: tuple[int, int],
    output_height: int,
    max_depth: Optional[float] = None,
    label: str = "3D voxel view",
) -> np.ndarray:
    """Render observed 3D voxels from the robot camera perspective."""
    height, width = image_shape
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    centers_world = _occupied_voxel_centers(voxel_map)
    if len(centers_world) == 0:
        _draw_label(image, label)
        return _fit_to_height(image, output_height)

    T_camera_world = np.linalg.inv(T_world_camera)
    min_height = float(centers_world[:, 1].min())
    max_height = float(centers_world[:, 1].max())
    centers_h = np.ones((len(centers_world), 4), dtype=np.float64)
    centers_h[:, :3] = centers_world
    centers_camera = centers_h @ T_camera_world.T
    z = centers_camera[:, 2]
    valid = z > 0.0
    if max_depth is not None:
        valid &= z <= max_depth
    if np.any(valid):
        safe_z = np.where(z > 0.0, z, 1.0)
        u = np.rint(
            intrinsics.fx * centers_camera[:, 0] / safe_z + intrinsics.cx
        ).astype(np.int32)
        v = np.rint(
            intrinsics.fy * centers_camera[:, 1] / safe_z + intrinsics.cy
        ).astype(np.int32)
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)

    projected = []
    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        depths = z[valid_indices]
        radii = np.rint(
            voxel_map.voxel_size * intrinsics.fx / np.maximum(depths, 0.01) * 0.5
        ).astype(np.int32)
        radii = np.clip(radii, 1, 5)
        colors = _occupied_depth_height_colors_bgr(
            centers_world[valid_indices, 1],
            min_height,
            max_height,
            depths=depths,
            max_depth=max_depth,
        )
        projected = list(
            zip(
                depths,
                u[valid_indices],
                v[valid_indices],
                radii,
                colors,
            )
        )

    for _, u_px, v_px, radius, color in sorted(
        projected,
        key=lambda item: item[0],
        reverse=True,
    ):
        cv2.rectangle(
            image,
            (max(0, int(u_px) - int(radius)), max(0, int(v_px) - int(radius))),
            (
                min(width - 1, int(u_px) + int(radius)),
                min(height - 1, int(v_px) + int(radius)),
            ),
            tuple(int(channel) for channel in color),
            thickness=-1,
        )

    _draw_label(image, label)
    legend_height = (min_height + max_height) * 0.5
    legend_far_depth = 1.0 if max_depth is None else max_depth
    _draw_legend(
        image,
        [
            (
                "near",
                _occupied_depth_height_color_bgr(
                    legend_height,
                    min_height,
                    max_height,
                    depth=0.0,
                    max_depth=max_depth,
                ),
            ),
            (
                "far",
                _occupied_depth_height_color_bgr(
                    legend_height,
                    min_height,
                    max_height,
                    depth=legend_far_depth,
                    max_depth=max_depth,
                ),
            ),
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


def render_logodds_difference_histogram_bgr(
    histogram: Optional[object],
    *,
    output_height: int,
    width: int = 360,
    label: str = "|log-odds diff|",
) -> np.ndarray:
    """Render a compact histogram panel for voxel-map log-odds differences."""
    image = np.full((output_height, width, 3), 28, dtype=np.uint8)
    _draw_label(image, label)

    if histogram is None:
        _draw_small_text(image, "No comparison yet", (18, 62))
        return image

    counts = np.asarray(histogram.counts, dtype=np.float64)
    edges = np.asarray(histogram.bin_edges, dtype=np.float64)
    total = int(histogram.total_different)
    if counts.size == 0 or edges.size != counts.size + 1 or total == 0:
        _draw_small_text(image, "No nonzero differences", (18, 62))
        return image

    left, right = 42, width - 16
    top, bottom = 70, output_height - 42
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)
    cv2.rectangle(image, (left, top), (right, bottom), (70, 70, 70), 1)

    max_count = float(counts.max())
    bar_width = max(1, int(np.floor(plot_width / counts.size)))
    for index, count in enumerate(counts):
        if count <= 0.0 or max_count <= 0.0:
            continue
        x0 = left + int(round(index * plot_width / counts.size))
        x1 = min(right - 1, x0 + bar_width - 1)
        y0 = bottom - int(round(count / max_count * plot_height))
        color = _histogram_bar_color(index / max(counts.size - 1, 1))
        cv2.rectangle(image, (x0, y0), (x1, bottom - 1), color, -1)
        text = str(int(count))
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)[0]
        text_x = x0 + max(0, (x1 - x0 - text_size[0]) // 2)
        text_y = max(38, y0 - 4)
        _draw_small_text(
            image,
            text,
            (text_x, text_y),
            font_scale=0.28,
            color=(245, 245, 245),
        )

    _draw_small_text(image, f"changed voxels: {total}", (18, output_height - 18))
    _draw_small_text(image, f"{edges[0]:.1f}", (left - 2, bottom + 18))
    _draw_small_text(image, f"{edges[-1]:.1f}", (right - 34, bottom + 18))
    _draw_small_text(image, "linear count scale", (width - 126, top - 8))
    return image


def _histogram_bar_color(t: float) -> tuple[int, int, int]:
    scalar = np.array([[int(round(np.clip(t, 0.0, 1.0) * 255.0))]], dtype=np.uint8)
    color = cv2.applyColorMap(scalar, cv2.COLORMAP_VIRIDIS)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def _occupied_depth_height_color_bgr(
    height: float,
    min_height: float,
    max_height: float,
    *,
    depth: float,
    max_depth: Optional[float],
) -> tuple[int, int, int]:
    span = max(max_height - min_height, 1e-6)
    height_t = min(max((height - min_height) / span, 0.0), 1.0)
    if max_depth is None or max_depth <= 0.0:
        depth_t = 0.5
    else:
        depth_t = min(max(depth / max_depth, 0.0), 1.0)

    # OpenCV JET maps low values to blue and high values to red. Invert depth
    # so nearby occupied voxels are warm and far voxels are cool.
    scalar = np.array([[round((1.0 - depth_t) * 255.0)]], dtype=np.uint8)
    color = cv2.applyColorMap(scalar, cv2.COLORMAP_JET)[0, 0].astype(np.float32)
    brightness = 0.62 + 0.38 * height_t
    color *= brightness
    color += np.array([18.0, 18.0, 18.0], dtype=np.float32) * height_t
    color = np.clip(color, 0, 255).astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def _occupied_depth_height_colors_bgr(
    heights: np.ndarray,
    min_height: float,
    max_height: float,
    *,
    depths: np.ndarray,
    max_depth: Optional[float],
) -> np.ndarray:
    span = max(max_height - min_height, 1e-6)
    height_t = np.clip((heights - min_height) / span, 0.0, 1.0)
    if max_depth is None or max_depth <= 0.0:
        depth_t = np.full_like(depths, 0.5, dtype=np.float64)
    else:
        depth_t = np.clip(depths / max_depth, 0.0, 1.0)

    scalars = np.rint((1.0 - depth_t) * 255.0).astype(np.uint8).reshape(-1, 1)
    colors = cv2.applyColorMap(scalars, cv2.COLORMAP_JET)[:, 0, :].astype(
        np.float32
    )
    brightness = (0.62 + 0.38 * height_t).astype(np.float32)
    colors *= brightness[:, None]
    colors += np.array([18.0, 18.0, 18.0], dtype=np.float32) * height_t[:, None]
    return np.clip(colors, 0, 255).astype(np.uint8)


def _occupied_voxel_centers(voxel_map: SparseVoxelMap) -> np.ndarray:
    centers = []
    block_size = voxel_map.block_size
    voxel_size = voxel_map.voxel_size
    threshold = voxel_map._occupied_logodds
    for block_index, block in voxel_map.iter_allocated_blocks():
        mask = block.observed & (block.occupancy_logodds >= threshold)
        local_x, local_y, local_z = np.nonzero(mask)
        if len(local_x) == 0:
            continue

        block_origin = np.array(block_index, dtype=np.float64) * block_size
        local = np.stack((local_x, local_y, local_z), axis=1).astype(np.float64)
        centers.append((block_origin + local + 0.5) * voxel_size)

    if not centers:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(centers)


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


def _topdown_local_bounds(
    topdown: TopDownGrid,
    agent_position: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
) -> tuple[float, float, float, float]:
    height, width = topdown.data.shape
    half = topdown.resolution * 0.5
    min_x = topdown.origin[0] - half
    max_x = topdown.origin[0] + (width - 1) * topdown.resolution + half
    min_z = topdown.origin[1] - half
    max_z = topdown.origin[1] + (height - 1) * topdown.resolution + half
    corners = np.array(
        [
            [min_x, min_z],
            [min_x, max_z],
            [max_x, min_z],
            [max_x, max_z],
        ],
        dtype=np.float64,
    )
    origin = np.array([agent_position[0], agent_position[2]], dtype=np.float64)
    relative = corners - origin
    local_right = relative @ right
    local_forward = relative @ forward
    return (
        float(local_right.min()),
        float(local_right.max()),
        float(local_forward.min()),
        float(local_forward.max()),
    )


def _sample_topdown_in_agent_frame(
    topdown: TopDownGrid,
    agent_position: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    local_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    min_right, max_right, min_forward, max_forward = local_bounds
    resolution = topdown.resolution
    width = max(1, int(np.ceil((max_right - min_right) / resolution)))
    height = max(1, int(np.ceil((max_forward - min_forward) / resolution)))

    cols = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    local_right = min_right + (cols + 0.5) * resolution
    local_forward = max_forward - (rows + 0.5) * resolution
    origin = np.array([agent_position[0], agent_position[2]], dtype=np.float64)
    world_x = (
        origin[0]
        + np.outer(np.ones(height), local_right) * right[0]
        + np.outer(local_forward, np.ones(width)) * forward[0]
    )
    world_z = (
        origin[1]
        + np.outer(np.ones(height), local_right) * right[1]
        + np.outer(local_forward, np.ones(width)) * forward[1]
    )

    source_x0 = topdown.origin[0] - resolution * 0.5
    source_z0 = topdown.origin[1] - resolution * 0.5
    col_index = np.floor((world_x - source_x0) / resolution).astype(np.int32)
    row_index = np.floor((world_z - source_z0) / resolution).astype(np.int32)
    valid = (
        (row_index >= 0)
        & (row_index < topdown.data.shape[0])
        & (col_index >= 0)
        & (col_index < topdown.data.shape[1])
    )
    sampled = np.full((height, width), UNKNOWN, dtype=np.int8)
    sampled[valid] = topdown.data[row_index[valid], col_index[valid]]
    return sampled


def _agent_center_in_local_bounds(
    local_bounds: tuple[float, float, float, float],
    resolution: float,
    scale: float,
) -> tuple[int, int]:
    min_right, _, _, max_forward = local_bounds
    col = (-min_right) / resolution - 0.5
    row = max_forward / resolution - 0.5
    return int(round(col * scale)), int(round(row * scale))


def _agent_marker_in_topdown(
    agent_state: object,
    topdown: TopDownGrid,
    scale: float,
) -> tuple[tuple[int, int], np.ndarray]:
    position = np.asarray(agent_state.position, dtype=np.float64)
    col = (position[0] - topdown.origin[0]) / topdown.resolution
    row = (position[2] - topdown.origin[1]) / topdown.resolution
    row = topdown.data.shape[0] - 1 - row
    _, forward = _agent_horizontal_basis(agent_state)
    direction = np.array([forward[0], -forward[1]], dtype=np.float64)
    return (int(round(col * scale)), int(round(row * scale))), direction


def _draw_agent_marker(
    image: np.ndarray,
    *,
    center: Optional[tuple[int, int]] = None,
    direction: Optional[np.ndarray] = None,
) -> None:
    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = max(5, min(image.shape[:2]) // 38)
    if direction is None:
        direction = np.array([0.0, -1.0], dtype=np.float64)
    direction = _normalize_2d(np.asarray(direction, dtype=np.float64))
    if float(np.linalg.norm(direction)) == 0.0:
        direction = np.array([0.0, -1.0], dtype=np.float64)
    right = np.array([direction[1], -direction[0]], dtype=np.float64)
    center_arr = np.array(center, dtype=np.float64)
    points = np.array(
        [
            center_arr + direction * radius * 1.8,
            center_arr - direction * radius * 0.9 - right * radius * 0.8,
            center_arr - direction * radius * 0.9 + right * radius * 0.8,
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


def _draw_small_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    font_scale: float = 0.38,
    color: tuple[int, int, int] = (225, 225, 225),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_legend(
    image: np.ndarray,
    items: list[tuple[str, tuple[int, int, int]]],
) -> None:
    x = 12
    row_height = 15
    text_widths = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        for text, _ in items
    ]
    width = max(92, max(text_widths, default=0) + 22)
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


def _fit_to_height_nearest(
    image: np.ndarray,
    output_height: int,
) -> tuple[np.ndarray, float]:
    if image.shape[0] == output_height:
        return image, 1.0
    scale = output_height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    resized = cv2.resize(
        image,
        (width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized, scale
