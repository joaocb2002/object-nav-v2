from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from object_nav.agents import KeyboardControls, RandomActionAgent
from object_nav.mapping import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    CameraIntrinsics,
    SparseVoxelMap,
    TopDownGrid,
    clamp_logodds,
    logodds_to_prob,
    prob_to_logodds,
    raycast_voxels,
)
from object_nav.mapping.habitat import (
    camera_intrinsics_from_sensor_config,
    depth_observation_to_meters,
)
from object_nav.mapping.point_cloud import (
    backproject_rgbd_to_world_points,
    load_colored_ply,
    write_colored_ply,
)
from object_nav.mapping.visualization import (
    render_full_voxel_topdown_from_agent_bgr,
    render_full_voxel_topdown_bgr,
    render_voxel_camera_view_bgr,
)
from object_nav.utils import make_run_output_dir


class FakeDepthSensorConfig:
    width = 640
    height = 480
    hfov = 90
    min_depth = 0.5
    max_depth = 5.0
    normalize_depth = True


class FakeAgentState:
    def __init__(self, position: tuple[float, float, float], rotation: object) -> None:
        self.position = position
        self.rotation = rotation


class SparseVoxelMapTest(unittest.TestCase):
    def test_keyboard_controls_map_keys_to_actions(self) -> None:
        controls = KeyboardControls()

        self.assertEqual(controls.action_for_key(ord("w")), "move_forward")
        self.assertEqual(controls.action_for_key(ord("a")), "turn_left")
        self.assertEqual(controls.action_for_key(ord("d")), "turn_right")
        self.assertEqual(controls.action_for_key(ord("f")), "stop")
        self.assertIsNone(controls.action_for_key(ord("x")))

    def test_random_action_agent_selects_from_actions(self) -> None:
        agent = RandomActionAgent(actions=("move_forward",))

        self.assertEqual(agent.act(), "move_forward")

    def test_habitat_intrinsics_from_sensor_config(self) -> None:
        intrinsics = camera_intrinsics_from_sensor_config(FakeDepthSensorConfig())

        self.assertAlmostEqual(intrinsics.fx, 320.0)
        self.assertAlmostEqual(intrinsics.fy, 320.0)
        self.assertAlmostEqual(intrinsics.cx, 319.5)
        self.assertAlmostEqual(intrinsics.cy, 239.5)

    def test_habitat_normalized_depth_converts_to_meters(self) -> None:
        depth = np.array([[[0.0], [1.0], [0.5]]], dtype=np.float32)

        depth_meters = depth_observation_to_meters(depth, FakeDepthSensorConfig())

        np.testing.assert_allclose(
            depth_meters,
            np.array([[0.5, 5.0, 2.75]], dtype=np.float32),
        )

    def test_voxel_camera_view_renderer_returns_bgr_image(self) -> None:
        voxel_map = SparseVoxelMap(voxel_size=1.0)
        voxel_map.integrate_ray((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), 1)

        image = render_voxel_camera_view_bgr(
            voxel_map,
            np.eye(4),
            CameraIntrinsics(fx=20.0, fy=20.0, cx=10.0, cy=10.0),
            image_shape=(20, 20),
            output_height=20,
            max_depth=5.0,
        )

        self.assertEqual(image.shape, (20, 20, 3))
        self.assertGreater(int(image.sum()), 0)

    def test_full_voxel_topdown_renderer_returns_whole_map(self) -> None:
        import quaternion

        topdown = TopDownGrid(
            data=np.array(
                [
                    [UNKNOWN, FREE, OCCUPIED, FREE],
                    [FREE, FREE, UNKNOWN, OCCUPIED],
                ],
                dtype=np.int8,
            ),
            origin=(0.0, 0.0),
            resolution=1.0,
            axes=(0, 2),
            vertical_axis=1,
        )
        agent_state = FakeAgentState(
            position=(1.0, 0.0, 1.0),
            rotation=quaternion.quaternion(1.0, 0.0, 0.0, 0.0),
        )

        image = render_full_voxel_topdown_bgr(
            topdown,
            agent_state,
            output_height=20,
        )

        self.assertEqual(image.shape, (20, 40, 3))
        self.assertGreater(int(image.sum()), 0)

    def test_full_egocentric_voxel_topdown_renderer_returns_whole_map(self) -> None:
        import quaternion

        topdown = TopDownGrid(
            data=np.array(
                [
                    [UNKNOWN, FREE, OCCUPIED, FREE],
                    [FREE, FREE, UNKNOWN, OCCUPIED],
                ],
                dtype=np.int8,
            ),
            origin=(0.0, 0.0),
            resolution=1.0,
            axes=(0, 2),
            vertical_axis=1,
        )
        agent_state = FakeAgentState(
            position=(1.0, 0.0, 1.0),
            rotation=quaternion.quaternion(1.0, 0.0, 0.0, 0.0),
        )

        image = render_full_voxel_topdown_from_agent_bgr(
            topdown,
            agent_state,
            output_height=20,
        )

        self.assertGreaterEqual(image.shape[0], 20)
        self.assertGreaterEqual(image.shape[1], 20)
        self.assertGreater(int(image.sum()), 0)

    def test_point_cloud_backprojection_uses_rgb_colors(self) -> None:
        depth = np.array([[2.0]], dtype=np.float32)
        rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)

        points, colors = backproject_rgbd_to_world_points(
            depth,
            rgb,
            CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            np.eye(4),
            pixel_stride=1,
            min_depth=0.5,
            max_depth=5.0,
        )

        np.testing.assert_allclose(points, np.array([[0.0, 0.0, 2.0]], dtype=np.float32))
        np.testing.assert_array_equal(colors, rgb.reshape(1, 3))

    def test_write_colored_ply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cloud.ply"
            write_colored_ply(
                path,
                np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                np.array([[10, 20, 30]], dtype=np.uint8),
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("element vertex 1", text)
        self.assertIn("1.00000 2.00000 3.00000 10 20 30", text)

    def test_load_colored_ply_round_trip(self) -> None:
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        colors = np.array([[10, 20, 30]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cloud.ply"
            write_colored_ply(path, points, colors)

            loaded_points, loaded_colors = load_colored_ply(path)

        np.testing.assert_allclose(loaded_points, points)
        np.testing.assert_array_equal(loaded_colors, colors)

    def test_run_output_dir_contains_identifying_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = make_run_output_dir(
                script_path="/tmp/main.py",
                scene_id="92vYG1q49FY",
                episode_id="episode/1",
                root=tmpdir,
                timestamp=datetime(2026, 7, 2, 18, 42, 10),
            )

            self.assertTrue(output_dir.is_dir())
            self.assertEqual(
                output_dir.name,
                "2026-07-02_18-42-10__scene-92vYG1q49FY__episode-episode-1__script-main",
            )

    def test_coordinate_conversion_handles_negative_coordinates(self) -> None:
        voxel_map = SparseVoxelMap(voxel_size=0.5, block_size=4)

        self.assertEqual(voxel_map.world_to_voxel_index((0.0, 0.49, -0.01)), (0, 0, -1))
        self.assertEqual(
            voxel_map.world_to_voxel_index((-0.01, -0.5, -0.51)),
            (-1, -1, -2),
        )
        self.assertEqual(voxel_map.voxel_index_to_block_index((-1, -5, 4)), (-1, -2, 1))
        self.assertEqual(voxel_map.voxel_index_to_local_index((-1, -5, 4)), (3, 3, 0))
        np.testing.assert_allclose(
            voxel_map.voxel_index_to_world_center((-1, 0, 2)),
            np.array([-0.25, 0.25, 1.25]),
        )

    def test_logodds_helpers_and_updates(self) -> None:
        p = 0.7
        self.assertTrue(math.isclose(logodds_to_prob(prob_to_logodds(p)), p))
        self.assertEqual(clamp_logodds(10.0, minimum=-2.0, maximum=2.0), 2.0)
        self.assertEqual(clamp_logodds(-10.0, minimum=-2.0, maximum=2.0), -2.0)

        voxel_map = SparseVoxelMap(voxel_size=1.0)
        index = (0, 0, 0)
        for step in range(3):
            voxel_map.integrate_ray((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), step)

        start_prob = voxel_map.occupancy_probability(index)
        end_prob = voxel_map.occupancy_probability((1, 0, 0))
        self.assertIsNotNone(start_prob)
        self.assertIsNotNone(end_prob)
        self.assertLess(start_prob, 0.5)
        self.assertGreater(end_prob, 0.5)

    def test_raycast_axis_aligned_includes_endpoint(self) -> None:
        voxels = raycast_voxels((0.1, 0.1, 0.1), (3.1, 0.1, 0.1), 1.0)

        self.assertEqual(voxels, [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])

    def test_raycast_diagonal_has_no_duplicates(self) -> None:
        voxels = raycast_voxels((0.1, 0.1, 0.1), (2.1, 2.1, 2.1), 1.0)

        self.assertEqual(voxels[0], (0, 0, 0))
        self.assertEqual(voxels[-1], (2, 2, 2))
        self.assertEqual(len(voxels), len(set(voxels)))

    def test_sparse_allocation_and_unknown_space(self) -> None:
        voxel_map = SparseVoxelMap(voxel_size=1.0, block_size=4)

        self.assertEqual(voxel_map.num_allocated_blocks(), 0)
        self.assertIsNone(voxel_map.get_voxel((100, 100, 100)))

        voxel_map.integrate_ray((0.1, 0.1, 0.1), (2.1, 0.1, 0.1), 1)

        self.assertEqual(voxel_map.num_allocated_blocks(), 1)
        self.assertEqual(voxel_map.num_observed_voxels(), 3)
        self.assertIsNotNone(voxel_map.get_voxel((3, 0, 0)))
        self.assertFalse(voxel_map.is_observed((3, 0, 0)))
        self.assertIsNone(voxel_map.get_voxel((100, 100, 100)))

    def test_depth_integration_single_point(self) -> None:
        voxel_map = SparseVoxelMap(voxel_size=1.0, block_size=4)
        depth = np.array([[3.2]], dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)

        voxel_map.integrate_depth(depth, intrinsics, np.eye(4), step_index=7)

        self.assertTrue(voxel_map.is_free((0, 0, 0)))
        self.assertTrue(voxel_map.is_free((0, 0, 1)))
        self.assertTrue(voxel_map.is_free((0, 0, 2)))
        self.assertTrue(voxel_map.is_occupied((0, 0, 3)))
        far_voxel = voxel_map.get_voxel((0, 0, 4))
        self.assertTrue(far_voxel is None or not voxel_map.is_observed((0, 0, 4)))

    def test_depth_integration_filters_invalid_depths(self) -> None:
        voxel_map = SparseVoxelMap(voxel_size=1.0)
        depth = np.array([[0.0, np.nan, -1.0, 2.0]], dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)

        voxel_map.integrate_depth(
            depth,
            intrinsics,
            np.eye(4),
            step_index=1,
            min_depth=0.5,
            max_depth=1.5,
        )
        self.assertEqual(voxel_map.num_allocated_blocks(), 0)

        voxel_map.integrate_depth(
            depth,
            intrinsics,
            np.eye(4),
            step_index=2,
            min_depth=0.5,
            max_depth=2.5,
        )
        self.assertGreater(voxel_map.num_allocated_blocks(), 0)

    def test_topdown_projection_marks_obstacle_free_and_unknown(self) -> None:
        voxel_map = SparseVoxelMap(voxel_size=1.0, block_size=4)

        voxel_map.integrate_ray((0.1, 0.0, 0.1), (0.1, 0.0, 1.1), 1)
        voxel_map.integrate_ray((2.1, 0.0, 0.1), (2.1, 0.6, 0.1), 2)

        grid = voxel_map.build_topdown_occupancy(
            floor_min_z=-0.5,
            floor_max_z=0.6,
            obstacle_min_z=0.5,
            obstacle_max_z=1.0,
            vertical_axis=1,
        )

        free_col = int(math.floor((0.5 - grid.origin[0]) / grid.resolution))
        free_row = int(math.floor((0.5 - grid.origin[1]) / grid.resolution))
        obs_col = int(math.floor((2.5 - grid.origin[0]) / grid.resolution))
        obs_row = int(math.floor((0.5 - grid.origin[1]) / grid.resolution))

        self.assertEqual(grid.data[free_row, free_col], FREE)
        self.assertEqual(grid.data[obs_row, obs_col], OCCUPIED)
        self.assertIn(UNKNOWN, grid.data)


if __name__ == "__main__":
    unittest.main()
