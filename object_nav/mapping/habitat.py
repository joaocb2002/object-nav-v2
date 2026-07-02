from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan
from typing import Any, Mapping, Optional

import numpy as np

from object_nav.mapping.voxel import CameraIntrinsics, SparseVoxelMap, TopDownGrid


@dataclass(frozen=True)
class HabitatVoxelMapConfig:
    """Runtime settings for Habitat depth integration."""

    voxel_size: float = 0.10
    block_size: int = 16
    pixel_stride: int = 6
    max_ray_length: Optional[float] = 5.0
    obstacle_min_height: float = 0.20
    obstacle_max_height: float = 1.50
    floor_min_height: float = -0.10
    floor_max_height: float = 0.30
    local_view_size_m: float = 8.0
    local_pixels_per_meter: int = 60


class HabitatVoxelMapper:
    """Adapter that feeds Habitat RGB-D observations into a sparse voxel map."""

    def __init__(
        self,
        habitat_config: Any,
        *,
        config: HabitatVoxelMapConfig = HabitatVoxelMapConfig(),
        depth_uuid: str = "depth",
        depth_sensor_name: str = "depth_sensor",
    ) -> None:
        self.config = config
        self.depth_uuid = depth_uuid
        self.depth_sensor_name = depth_sensor_name
        self.depth_sensor_config = _get_sim_sensor_config(
            habitat_config,
            depth_sensor_name,
        )
        self.camera_intrinsics = camera_intrinsics_from_sensor_config(
            self.depth_sensor_config
        )
        self.voxel_map = self._new_voxel_map()

    def reset(self) -> None:
        """Start a fresh map for a new episode."""
        self.voxel_map = self._new_voxel_map()

    def integrate(
        self,
        env: Any,
        observations: Mapping[str, Any],
        step_index: int,
    ) -> None:
        """Integrate the current Habitat depth observation."""
        if self.depth_uuid not in observations:
            return

        depth_meters = depth_observation_to_meters(
            observations[self.depth_uuid],
            self.depth_sensor_config,
        )
        T_world_camera = depth_camera_transform(env, self.depth_uuid)
        self.voxel_map.integrate_depth(
            depth_meters,
            self.camera_intrinsics,
            T_world_camera,
            step_index,
            pixel_stride=self.config.pixel_stride,
            min_depth=float(self.depth_sensor_config.min_depth),
            max_depth=float(self.depth_sensor_config.max_depth),
        )

    def build_topdown_projection(self, env: Any) -> TopDownGrid:
        """Build a voxel-derived top-down grid around the agent's current floor."""
        agent_height = float(env.sim.get_agent_state().position[1])
        return self.voxel_map.build_topdown_occupancy(
            floor_min_z=agent_height + self.config.floor_min_height,
            floor_max_z=agent_height + self.config.floor_max_height,
            obstacle_min_z=agent_height + self.config.obstacle_min_height,
            obstacle_max_z=agent_height + self.config.obstacle_max_height,
            vertical_axis=1,
        )

    def render_maps(self, env: Any, *, output_height: int) -> np.ndarray:
        """Render the 3D voxel POV and voxel-derived top-down map side by side."""
        from object_nav.mapping.visualization import (
            render_full_voxel_topdown_from_agent_bgr,
            render_voxel_camera_view_bgr,
        )

        agent_state = env.sim.get_agent_state()
        camera_image = render_voxel_camera_view_bgr(
            self.voxel_map,
            depth_camera_transform(env, self.depth_uuid),
            self.camera_intrinsics,
            image_shape=(
                int(self.depth_sensor_config.height),
                int(self.depth_sensor_config.width),
            ),
            output_height=output_height,
            max_depth=self.config.max_ray_length,
        )
        voxel_image = render_full_voxel_topdown_from_agent_bgr(
            self.build_topdown_projection(env),
            agent_state,
            output_height=output_height,
        )
        return _hstack_same_height([camera_image, voxel_image])

    def _new_voxel_map(self) -> SparseVoxelMap:
        return SparseVoxelMap(
            voxel_size=self.config.voxel_size,
            block_size=self.config.block_size,
            max_ray_length=self.config.max_ray_length,
        )


def render_habitat_topdown_map(env: Any, *, output_height: int) -> np.ndarray:
    """Render Habitat's ground-truth TopDownMap metric as a BGR image."""
    from object_nav.mapping.visualization import render_ground_truth_topdown_bgr

    return render_ground_truth_topdown_bgr(
        env.get_metrics().get("top_down_map"),
        output_height=output_height,
    )


def show_habitat_topdown_map(
    env: Any,
    *,
    output_height: int,
    window_name: str = "Habitat map",
) -> None:
    """Show Habitat's ground-truth TopDownMap metric in its own window."""
    from object_nav.mapping.visualization import show_navigation_maps

    show_navigation_maps(
        render_habitat_topdown_map(env, output_height=output_height),
        window_name=window_name,
    )


def enable_topdown_map_measure(
    habitat_config: Any,
    *,
    map_resolution: int = 1024,
) -> None:
    """Enable Habitat's ground-truth TopDownMap metric on an unlocked config."""
    from habitat.config.default_structured_configs import (
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )

    habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=map_resolution,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=True,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            )
        }
    )


def camera_intrinsics_from_sensor_config(sensor_config: Any) -> CameraIntrinsics:
    """Build pinhole intrinsics from a Habitat camera sensor config."""
    width = float(sensor_config.width)
    height = float(sensor_config.height)
    hfov = radians(float(sensor_config.hfov))
    focal = width / (2.0 * tan(hfov / 2.0))
    return CameraIntrinsics(
        fx=focal,
        fy=focal,
        cx=(width - 1.0) / 2.0,
        cy=(height - 1.0) / 2.0,
    )


def depth_observation_to_meters(depth: np.ndarray, sensor_config: Any) -> np.ndarray:
    """Return a 2D depth image in meters from a Habitat depth observation."""
    depth_2d = np.asarray(depth, dtype=np.float32)
    if depth_2d.ndim == 3 and depth_2d.shape[-1] == 1:
        depth_2d = depth_2d[:, :, 0]
    if bool(getattr(sensor_config, "normalize_depth", False)):
        min_depth = float(sensor_config.min_depth)
        max_depth = float(sensor_config.max_depth)
        depth_2d = depth_2d * (max_depth - min_depth) + min_depth
    return depth_2d


def depth_camera_transform(env: Any, depth_uuid: str = "depth") -> np.ndarray:
    """Return ``T_world_camera`` for CV optical camera coordinates.

    Habitat camera sensors use an OpenGL-style local frame (+X right, +Y up,
    -Z forward). The sparse mapper's pinhole model uses the common optical frame
    (+X right, +Y down, +Z forward), so this adds the fixed axis conversion.
    """
    sensor_state = _find_sensor_state(env.sim.get_agent_state(), depth_uuid)
    T_world_sensor = _state_to_transform(sensor_state)
    T_sensor_camera = np.eye(4, dtype=np.float64)
    T_sensor_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    return T_world_sensor @ T_sensor_camera


def _get_sim_sensor_config(habitat_config: Any, sensor_name: str) -> Any:
    agents = habitat_config.habitat.simulator.agents
    agent_config = (
        agents.main_agent
        if hasattr(agents, "main_agent")
        else next(iter(agents.values()))
    )
    return agent_config.sim_sensors[sensor_name]


def _find_sensor_state(agent_state: Any, sensor_uuid: str) -> Any:
    sensor_states = agent_state.sensor_states
    if sensor_uuid in sensor_states:
        return sensor_states[sensor_uuid]

    fallback_uuid = f"{sensor_uuid}_sensor"
    if fallback_uuid in sensor_states:
        return sensor_states[fallback_uuid]

    available = ", ".join(sorted(sensor_states))
    raise KeyError(f"Sensor state {sensor_uuid!r} not found. Available: {available}")


def _state_to_transform(state: Any) -> np.ndarray:
    import quaternion

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion.as_rotation_matrix(state.rotation)
    transform[:3, 3] = np.asarray(state.position, dtype=np.float64)
    return transform


def _hstack_same_height(images: list[np.ndarray]) -> np.ndarray:
    import cv2

    height = max(image.shape[0] for image in images)
    resized = []
    for image in images:
        if image.shape[0] != height:
            width = int(round(image.shape[1] * height / image.shape[0]))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        resized.append(image)
    return np.hstack(resized)
