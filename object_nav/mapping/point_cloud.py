from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from object_nav.mapping.habitat import (
    camera_intrinsics_from_sensor_config,
    depth_camera_transform,
    depth_observation_to_meters,
)


@dataclass(frozen=True)
class PointCloudDebugConfig:
    """Settings for RGB-D point cloud debugging."""

    pixel_stride: int = 8
    max_points: Optional[int] = 500_000
    window_name: str = "Point cloud debug"
    plot_pixels: int = 800


class HabitatPointCloudRecorder:
    """Accumulate a colored world-frame point cloud from Habitat RGB-D frames."""

    def __init__(
        self,
        habitat_config: Any,
        *,
        config: PointCloudDebugConfig = PointCloudDebugConfig(),
        rgb_uuid: str = "rgb",
        depth_uuid: str = "depth",
        depth_sensor_name: str = "depth_sensor",
    ) -> None:
        self.config = config
        self.rgb_uuid = rgb_uuid
        self.depth_uuid = depth_uuid
        self.depth_sensor_config = _get_depth_sensor_config(
            habitat_config,
            depth_sensor_name,
        )
        self.camera_intrinsics = camera_intrinsics_from_sensor_config(
            self.depth_sensor_config
        )
        self._points: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []

    def reset(self) -> None:
        """Clear accumulated points for a new episode."""
        self._points.clear()
        self._colors.clear()

    def integrate(self, env: Any, observations: Mapping[str, Any]) -> None:
        """Add one RGB-D observation to the accumulated point cloud."""
        if self.depth_uuid not in observations or self.rgb_uuid not in observations:
            return

        depth = depth_observation_to_meters(
            observations[self.depth_uuid],
            self.depth_sensor_config,
        )
        rgb = np.asarray(observations[self.rgb_uuid], dtype=np.uint8)
        T_world_camera = depth_camera_transform(env, self.depth_uuid)
        points, colors = backproject_rgbd_to_world_points(
            depth,
            rgb,
            self.camera_intrinsics,
            T_world_camera,
            pixel_stride=self.config.pixel_stride,
            min_depth=float(self.depth_sensor_config.min_depth),
            max_depth=float(self.depth_sensor_config.max_depth),
        )
        if len(points) == 0:
            return

        self._points.append(points)
        self._colors.append(colors)
        self._trim_to_max_points()

    def num_points(self) -> int:
        """Return the number of accumulated points."""
        if not self._points:
            return 0
        return int(sum(len(points) for points in self._points))

    def save(self, path: str | Path) -> Path:
        """Save the accumulated point cloud as an RGB ASCII `.ply` file."""
        ply_path = Path(path)
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        points, colors = self.as_arrays()
        write_colored_ply(ply_path, points, colors)
        return ply_path

    def save_static_preview(self, path: str | Path) -> Path:
        """Save a simple orthographic PNG preview for quick non-interactive checks."""
        image_path = Path(path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        points, colors = self.as_arrays()
        preview = render_point_cloud_summary_bgr(
            points,
            colors,
            image_size=self.config.plot_pixels,
        )
        cv2.imwrite(str(image_path), preview)
        return image_path

    def show_interactive(self, cloud: Optional[str | Path] = None) -> bool:
        """Open the point cloud in an interactive Open3D viewer if available."""
        point_cloud = cloud if cloud is not None else self.as_arrays()
        return show_interactive_point_cloud(point_cloud, title=self.config.window_name)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return accumulated points and RGB colors as arrays."""
        if not self._points:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )
        return np.vstack(self._points), np.vstack(self._colors)

    def _trim_to_max_points(self) -> None:
        if self.config.max_points is None:
            return

        points, colors = self.as_arrays()
        if len(points) <= self.config.max_points:
            return

        keep = int(self.config.max_points)
        self._points[:] = [points[-keep:]]
        self._colors[:] = [colors[-keep:]]


def backproject_rgbd_to_world_points(
    depth_meters: np.ndarray,
    rgb: np.ndarray,
    camera_intrinsics: Any,
    T_world_camera: np.ndarray,
    *,
    pixel_stride: int,
    min_depth: Optional[float],
    max_depth: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project sampled RGB-D pixels to world-frame colored points."""
    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")

    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError("depth_meters must be 2D or have a singleton channel")

    rgb_image = np.asarray(rgb, dtype=np.uint8)
    if rgb_image.shape[:2] != depth.shape:
        rgb_image = cv2.resize(
            rgb_image,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    v_coords = np.arange(0, depth.shape[0], pixel_stride)
    u_coords = np.arange(0, depth.shape[1], pixel_stride)
    uu, vv = np.meshgrid(u_coords, v_coords)
    z = depth[vv, uu]
    valid = np.isfinite(z) & (z > 0.0)
    if min_depth is not None:
        valid &= z >= min_depth
    if max_depth is not None:
        valid &= z <= max_depth
    if not np.any(valid):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    u_valid = uu[valid].astype(np.float32)
    v_valid = vv[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)
    x = (u_valid - camera_intrinsics.cx) * z_valid / camera_intrinsics.fx
    y = (v_valid - camera_intrinsics.cy) * z_valid / camera_intrinsics.fy
    points_camera = np.stack((x, y, z_valid, np.ones_like(z_valid)), axis=0)
    points_world = (T_world_camera @ points_camera)[:3].T.astype(np.float32)
    colors = rgb_image[vv[valid], uu[valid], :3].astype(np.uint8)
    return points_world, colors


def write_colored_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an RGB point cloud as ASCII PLY."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def load_colored_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an ASCII RGB PLY written by `write_colored_ply`."""
    vertex_count: Optional[int] = None
    points: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break

        for line in handle:
            values = line.split()
            if len(values) < 6:
                continue
            points.append((float(values[0]), float(values[1]), float(values[2])))
            colors.append((int(values[3]), int(values[4]), int(values[5])))
            if vertex_count is not None and len(points) >= vertex_count:
                break

    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def show_interactive_point_cloud(
    cloud: str | Path | tuple[np.ndarray, np.ndarray],
    *,
    title: str = "Point cloud viewer",
) -> bool:
    """Open a saved or in-memory RGB point cloud with Open3D when installed."""
    try:
        import open3d as o3d
    except ImportError:
        if isinstance(cloud, (str, Path)) and _try_external_point_cloud_viewer(Path(cloud)):
            return True
        print(
            "Open3D is not installed and no external PLY viewer was found. "
            "Saved PLY can be opened in CloudCompare, MeshLab, or after "
            "installing `open3d`."
        )
        return False

    if isinstance(cloud, (str, Path)):
        point_cloud = o3d.io.read_point_cloud(str(cloud))
    else:
        points, colors = cloud
        point_cloud = _open3d_point_cloud(o3d, points, colors)

    if point_cloud.is_empty():
        print("Point cloud is empty; nothing to visualize.")
        return False

    if _visualize_with_open3d(o3d, point_cloud, title):
        return True
    if isinstance(cloud, (str, Path)) and _try_external_point_cloud_viewer(Path(cloud)):
        return True

    print(
        "Open3D could not create a window. This usually means the current "
        "session cannot provide an OpenGL context."
    )
    return False


def render_point_cloud_summary_bgr(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    image_size: int = 800,
) -> np.ndarray:
    """Render top/front/side projections of a point cloud for quick inspection."""
    image = np.full((image_size, image_size * 3, 3), 28, dtype=np.uint8)
    if len(points) == 0:
        _draw_panel_label(image, "empty point cloud", (12, 28))
        return image

    panels = [
        ("top XZ", (0, 2), 0),
        ("front XY", (0, 1), image_size),
        ("side ZY", (2, 1), image_size * 2),
    ]
    for label, axes, x_offset in panels:
        panel = _render_projection(points, colors, axes, image_size)
        image[:, x_offset : x_offset + image_size] = panel
        _draw_panel_label(image, label, (x_offset + 12, 28))
    return image


def _render_projection(
    points: np.ndarray,
    colors_rgb: np.ndarray,
    axes: tuple[int, int],
    image_size: int,
) -> np.ndarray:
    panel = np.full((image_size, image_size, 3), 28, dtype=np.uint8)
    coords = points[:, axes]
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scale = (image_size - 20) / float(np.max(span))
    xy = ((coords - minimum) * scale + 10).astype(np.int32)
    xy[:, 1] = image_size - 1 - xy[:, 1]
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < image_size)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image_size)
    )
    colors_bgr = colors_rgb[:, ::-1]
    panel[xy[valid, 1], xy[valid, 0]] = colors_bgr[valid]
    return panel


def _open3d_point_cloud(
    o3d: Any,
    points: np.ndarray,
    colors: np.ndarray,
) -> Any:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same length")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
    return point_cloud


def _visualize_with_open3d(o3d: Any, point_cloud: Any, title: str) -> bool:
    visualizer = o3d.visualization.Visualizer()
    created = visualizer.create_window(window_name=title, width=1280, height=800)
    if not created:
        return False

    visualizer.add_geometry(point_cloud)
    render_options = visualizer.get_render_option()
    render_options.point_size = 2.0
    render_options.background_color = np.array([0.05, 0.05, 0.05])
    visualizer.run()
    visualizer.destroy_window()
    return True


def _try_external_point_cloud_viewer(path: Path) -> bool:
    viewer_commands = (
        ("cloudcompare", "-O", str(path)),
        ("CloudCompare", "-O", str(path)),
        ("meshlab", str(path)),
    )
    for command in viewer_commands:
        if shutil.which(command[0]) is None:
            continue
        subprocess.Popen(command)
        return True
    return False


def _draw_panel_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _get_depth_sensor_config(habitat_config: Any, sensor_name: str) -> Any:
    agents = habitat_config.habitat.simulator.agents
    agent_config = (
        agents.main_agent
        if hasattr(agents, "main_agent")
        else next(iter(agents.values()))
    )
    return agent_config.sim_sensors[sensor_name]
