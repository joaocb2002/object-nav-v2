from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import floor, inf, isfinite, log
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

VoxelIndex = Tuple[int, int, int]
BlockIndex = Tuple[int, int, int]
LocalVoxelIndex = Tuple[int, int, int]

UNKNOWN = -1
FREE = 0
OCCUPIED = 1
_PACKED_VOXEL_BITS = 21
_PACKED_VOXEL_SHIFT = 1 << (_PACKED_VOXEL_BITS - 1)
_PACKED_VOXEL_MASK = (1 << _PACKED_VOXEL_BITS) - 1


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera parameters for depth back-projection.

    Depth is interpreted in meters after multiplying by ``depth_scale``. Habitat
    depth observations are typically already meters, so the default scale is 1.
    Use ``depth_scale=0.001`` for millimeter depth images.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0


@dataclass(frozen=True)
class GeometryVoxel:
    """Snapshot of one geometry voxel.

    The map stores geometry in compact NumPy arrays inside sparse blocks. This
    value object is returned for public queries; semantics can later live in a
    separate lazy layer keyed by the same voxel index.
    """

    occupancy_logodds: float
    observed: bool
    last_update_step: int


@dataclass(frozen=True)
class TopDownGrid:
    """A sparse-map-derived 2D planning projection."""

    data: np.ndarray
    origin: Tuple[float, float]
    resolution: float
    axes: Tuple[int, int]
    vertical_axis: int


class VoxelBlock:
    """Small dense block of geometry voxels allocated on first touch."""

    def __init__(self, block_size: int) -> None:
        shape = (block_size, block_size, block_size)
        self.occupancy_logodds = np.zeros(shape, dtype=np.float32)
        self.observed = np.zeros(shape, dtype=bool)
        self.last_update_step = np.full(shape, -1, dtype=np.int32)

    def voxel(self, local_index: LocalVoxelIndex) -> GeometryVoxel:
        x, y, z = local_index
        return GeometryVoxel(
            occupancy_logodds=float(self.occupancy_logodds[x, y, z]),
            observed=bool(self.observed[x, y, z]),
            last_update_step=int(self.last_update_step[x, y, z]),
        )


def prob_to_logodds(probability: float) -> float:
    """Convert an occupancy probability in (0, 1) to log-odds."""
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("probability must be between 0 and 1, exclusive")
    return log(probability / (1.0 - probability))


def logodds_to_prob(logodds: float) -> float:
    """Convert occupancy log-odds to probability."""
    return 1.0 / (1.0 + np.exp(-logodds))


def clamp_logodds(
    logodds: float,
    *,
    minimum: float = -5.0,
    maximum: float = 5.0,
) -> float:
    """Clamp log-odds to a stable numeric range."""
    return float(min(max(logodds, minimum), maximum))


def raycast_voxels(
    start_world: Sequence[float],
    end_world: Sequence[float],
    voxel_size: float,
) -> list[VoxelIndex]:
    """Return voxel indices crossed by a segment, including both endpoints.

    This is a 3D DDA / Amanatides-Woo traversal over a floor-indexed voxel grid.
    It intentionally returns the camera-origin voxel as the first element so ray
    integration can mark all voxels before the endpoint as free.
    """
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    start = np.asarray(start_world, dtype=np.float64)
    end = np.asarray(end_world, dtype=np.float64)
    direction = end - start
    current = _world_to_voxel_index(start, voxel_size)
    target = _world_to_voxel_index(end, voxel_size)

    voxels = [current]
    if current == target:
        return voxels

    step = [0, 0, 0]
    t_max = [inf, inf, inf]
    t_delta = [inf, inf, inf]

    for axis in range(3):
        delta = float(direction[axis])
        if delta > 0.0:
            step[axis] = 1
            next_boundary = (current[axis] + 1) * voxel_size
            t_max[axis] = (next_boundary - start[axis]) / delta
            t_delta[axis] = voxel_size / delta
        elif delta < 0.0:
            step[axis] = -1
            next_boundary = current[axis] * voxel_size
            t_max[axis] = (next_boundary - start[axis]) / delta
            t_delta[axis] = -voxel_size / delta

    while current != target:
        axis = _axis_with_smallest_t(t_max)
        mutable = list(current)
        mutable[axis] += step[axis]
        current = (mutable[0], mutable[1], mutable[2])
        voxels.append(current)
        t_max[axis] += t_delta[axis]

    return voxels


class SparseVoxelMap:
    """Sparse block-based occupancy map for robot navigation.

    Coordinates are metric world/map coordinates. Voxel indices use
    ``floor(world / voxel_size)`` on every axis, which handles negative world
    coordinates correctly. Unknown space is implicit: absent blocks are unknown,
    and untouched voxels inside allocated blocks have ``observed=False``.

    Depth integration expects ``T_world_camera`` to transform points from a
    computer-vision optical camera frame (+X right, +Y down, +Z forward) into
    the world/map frame. Habitat world coordinates are meter-scaled and Y-up;
    pass a transform that already accounts for Habitat's sensor orientation.
    """

    def __init__(
        self,
        *,
        voxel_size: float = 0.1,
        block_size: int = 16,
        p_occ: float = 0.70,
        p_free: float = 0.30,
        logodds_min: float = -5.0,
        logodds_max: float = 5.0,
        occupied_threshold: float = 0.65,
        free_threshold: float = 0.35,
        max_ray_length: Optional[float] = None,
        raycast_backend: str = "auto",
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.voxel_size = float(voxel_size)
        self._inv_voxel_size = 1.0 / self.voxel_size
        self.block_size = int(block_size)
        self.logodds_min = float(logodds_min)
        self.logodds_max = float(logodds_max)
        self.occupied_threshold = float(occupied_threshold)
        self.free_threshold = float(free_threshold)
        self.max_ray_length = max_ray_length
        self.raycast_backend = raycast_backend
        self.blocks: Dict[BlockIndex, VoxelBlock] = {}

        self._occ_update = prob_to_logodds(p_occ)
        self._free_update = prob_to_logodds(p_free)
        self._occupied_logodds = prob_to_logodds(occupied_threshold)
        self._free_logodds = prob_to_logodds(free_threshold)

        if raycast_backend not in {"auto", "python", "numba"}:
            raise ValueError("raycast_backend must be 'auto', 'python', or 'numba'")
        if raycast_backend in {"auto", "numba"}:
            self._warm_up_numba_backend()

    def world_to_voxel_index(self, point_world: Sequence[float]) -> VoxelIndex:
        """Convert a world point to its floor-indexed voxel."""
        return _world_to_voxel_index(point_world, self.voxel_size)

    def voxel_index_to_world_center(self, index: VoxelIndex) -> np.ndarray:
        """Return the world-space center of a voxel."""
        return (np.asarray(index, dtype=np.float64) + 0.5) * self.voxel_size

    def voxel_index_to_block_index(self, index: VoxelIndex) -> BlockIndex:
        """Return the sparse block containing a voxel index."""
        return (
            index[0] // self.block_size,
            index[1] // self.block_size,
            index[2] // self.block_size,
        )

    def voxel_index_to_local_index(self, index: VoxelIndex) -> LocalVoxelIndex:
        """Return the local index of a voxel within its block."""
        return (
            index[0] % self.block_size,
            index[1] % self.block_size,
            index[2] % self.block_size,
        )

    def get_or_create_block(self, block_index: BlockIndex) -> VoxelBlock:
        """Return a block, allocating it lazily when first touched."""
        block = self.blocks.get(block_index)
        if block is None:
            block = VoxelBlock(self.block_size)
            self.blocks[block_index] = block
        return block

    def get_voxel(self, index: VoxelIndex) -> Optional[GeometryVoxel]:
        """Return a voxel snapshot, or ``None`` when its block is unknown."""
        block = self.blocks.get(self.voxel_index_to_block_index(index))
        if block is None:
            return None
        return block.voxel(self.voxel_index_to_local_index(index))

    def get_or_create_voxel(self, index: VoxelIndex) -> GeometryVoxel:
        """Allocate the containing block and return a voxel snapshot."""
        block = self.get_or_create_block(self.voxel_index_to_block_index(index))
        return block.voxel(self.voxel_index_to_local_index(index))

    def integrate_ray(
        self,
        start_world: Sequence[float],
        end_world: Sequence[float],
        step_index: int,
    ) -> None:
        """Integrate one depth ray into the map."""
        start = np.asarray(start_world, dtype=np.float64)
        end = np.asarray(end_world, dtype=np.float64)
        self._integrate_ray_fast(
            float(start[0]),
            float(start[1]),
            float(start[2]),
            float(end[0]),
            float(end[1]),
            float(end[2]),
            step_index,
        )

    def integrate_depth(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: CameraIntrinsics,
        T_world_camera: np.ndarray,
        step_index: int,
        *,
        pixel_stride: int = 1,
        max_depth: Optional[float] = None,
        min_depth: Optional[float] = None,
    ) -> None:
        """Integrate a depth frame using frame-level aggregated voxel updates.

        Back-projection uses optical camera axes: ``X=(u-cx)z/fx``,
        ``Y=(v-cy)z/fy``, ``Z=z``. The supplied transform maps those camera
        points into the world/map frame. Invalid, zero, NaN, and filtered depths
        are ignored.

        This is the standard integration path. It raycasts sampled depth pixels,
        accumulates free/occupied evidence per voxel for the frame, and applies
        one clamped log-odds update per touched voxel.
        """
        points_world, origin = self._depth_endpoints_world(
            depth_image,
            camera_intrinsics,
            T_world_camera,
            pixel_stride=pixel_stride,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if len(points_world) == 0:
            return

        if self._integrate_frame_updates_numba(origin, points_world, step_index):
            return

        free_counts: dict[VoxelIndex, int] = defaultdict(int)
        occupied_counts: dict[VoxelIndex, int] = defaultdict(int)
        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        for end_x, end_y, end_z in points_world:
            self._accumulate_ray_updates(
                ox,
                oy,
                oz,
                float(end_x),
                float(end_y),
                float(end_z),
                free_counts,
                occupied_counts,
            )

        self._apply_frame_updates(free_counts, occupied_counts, step_index)

    def integrate_depth_reference(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: CameraIntrinsics,
        T_world_camera: np.ndarray,
        step_index: int,
        *,
        pixel_stride: int = 1,
        max_depth: Optional[float] = None,
        min_depth: Optional[float] = None,
    ) -> None:
        """Integrate a depth frame with immediate per-ray updates.

        This slower path is kept as a reference for diagnostics. It applies and
        clamps every free/occupied ray update immediately, preserving the exact
        sampled-pixel processing order.
        """
        points_world, origin = self._depth_endpoints_world(
            depth_image,
            camera_intrinsics,
            T_world_camera,
            pixel_stride=pixel_stride,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if len(points_world) == 0:
            return
        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])

        for end_x, end_y, end_z in points_world:
            self._integrate_ray_fast(
                ox,
                oy,
                oz,
                float(end_x),
                float(end_y),
                float(end_z),
                step_index,
            )

    def integrate_depth_aggregated(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: CameraIntrinsics,
        T_world_camera: np.ndarray,
        step_index: int,
        *,
        pixel_stride: int = 1,
        max_depth: Optional[float] = None,
        min_depth: Optional[float] = None,
    ) -> None:
        """Compatibility alias for the standard aggregated depth integration."""
        self.integrate_depth(
            depth_image,
            camera_intrinsics,
            T_world_camera,
            step_index,
            pixel_stride=pixel_stride,
            max_depth=max_depth,
            min_depth=min_depth,
        )

    def build_topdown_occupancy(
        self,
        *,
        floor_min_z: float = -0.10,
        floor_max_z: float = 0.30,
        obstacle_min_z: float = 0.20,
        obstacle_max_z: float = 1.50,
        resolution: Optional[float] = None,
        vertical_axis: int = 1,
    ) -> TopDownGrid:
        """Project observed geometry into a 2D planning grid.

        The ``*_z`` names describe height bands from mapping literature; they
        are applied along ``vertical_axis``. Habitat-style maps should normally
        use the default ``vertical_axis=1`` (Y-up). Cells are obstacle if any
        occupied voxel is in the obstacle band, free if a near-ground voxel is
        observed free and no obstacle is present, and unknown otherwise.
        """
        if vertical_axis not in (0, 1, 2):
            raise ValueError("vertical_axis must be 0, 1, or 2")
        grid_resolution = self.voxel_size if resolution is None else float(resolution)
        if grid_resolution <= 0.0:
            raise ValueError("resolution must be positive")

        bounds = self.get_allocated_bounds()
        if bounds is None:
            return TopDownGrid(
                data=np.full((0, 0), UNKNOWN, dtype=np.int8),
                origin=(0.0, 0.0),
                resolution=grid_resolution,
                axes=_horizontal_axes(vertical_axis),
                vertical_axis=vertical_axis,
            )

        axes = _horizontal_axes(vertical_axis)
        min_index, max_index = bounds
        min_world = self.voxel_index_to_world_center(min_index)
        max_world = self.voxel_index_to_world_center(max_index)
        origin = (float(min_world[axes[0]]), float(min_world[axes[1]]))
        max_coord = (float(max_world[axes[0]]), float(max_world[axes[1]]))
        width = int(floor((max_coord[0] - origin[0]) / grid_resolution)) + 1
        height = int(floor((max_coord[1] - origin[1]) / grid_resolution)) + 1
        data = np.full((height, width), UNKNOWN, dtype=np.int8)
        free_seen = np.zeros((height, width), dtype=bool)

        for block_index, block in self.blocks.items():
            local_x, local_y, local_z = np.nonzero(block.observed)
            if len(local_x) == 0:
                continue

            global_indices = (
                block_index[0] * self.block_size + local_x,
                block_index[1] * self.block_size + local_y,
                block_index[2] * self.block_size + local_z,
            )
            centers = [
                (axis_indices.astype(np.float64) + 0.5) * self.voxel_size
                for axis_indices in global_indices
            ]
            vertical = centers[vertical_axis]
            col = np.floor((centers[axes[0]] - origin[0]) / grid_resolution).astype(
                np.int32
            )
            row = np.floor((centers[axes[1]] - origin[1]) / grid_resolution).astype(
                np.int32
            )
            valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
            if not np.any(valid):
                continue

            logodds = block.occupancy_logodds[local_x, local_y, local_z]
            obstacle = (
                valid
                & (vertical >= obstacle_min_z)
                & (vertical <= obstacle_max_z)
                & (logodds >= self._occupied_logodds)
            )
            if np.any(obstacle):
                data[row[obstacle], col[obstacle]] = OCCUPIED

            free = (
                valid
                & (vertical >= floor_min_z)
                & (vertical <= floor_max_z)
                & (logodds <= self._free_logodds)
            )
            if np.any(free):
                free_seen[row[free], col[free]] = True

        data[(data != OCCUPIED) & free_seen] = FREE
        return TopDownGrid(
            data=data,
            origin=origin,
            resolution=grid_resolution,
            axes=axes,
            vertical_axis=vertical_axis,
        )

    def is_observed(self, voxel_index: VoxelIndex) -> bool:
        """Return whether a voxel has received any sensor evidence."""
        voxel = self.get_voxel(voxel_index)
        return False if voxel is None else voxel.observed

    def is_free(self, voxel_index: VoxelIndex) -> bool:
        """Return whether a voxel is observed and below the free threshold."""
        voxel = self.get_voxel(voxel_index)
        return False if voxel is None else self._voxel_is_free(voxel)

    def is_occupied(self, voxel_index: VoxelIndex) -> bool:
        """Return whether a voxel is observed and above the occupied threshold."""
        voxel = self.get_voxel(voxel_index)
        return False if voxel is None else self._voxel_is_occupied(voxel)

    def occupancy_probability(self, voxel_index: VoxelIndex) -> Optional[float]:
        """Return occupancy probability for observed voxels, otherwise ``None``."""
        voxel = self.get_voxel(voxel_index)
        if voxel is None or not voxel.observed:
            return None
        return float(logodds_to_prob(voxel.occupancy_logodds))

    def num_allocated_blocks(self) -> int:
        """Return the number of sparse blocks currently allocated."""
        return len(self.blocks)

    def num_observed_voxels(self) -> int:
        """Return the number of voxels touched by sensor rays."""
        return sum(int(np.count_nonzero(block.observed)) for block in self.blocks.values())

    def get_allocated_bounds(self) -> Optional[Tuple[VoxelIndex, VoxelIndex]]:
        """Return inclusive min/max voxel bounds of allocated blocks."""
        if not self.blocks:
            return None

        block_indices = np.asarray(list(self.blocks), dtype=np.int64)
        min_block = block_indices.min(axis=0)
        max_block = block_indices.max(axis=0)
        minimum = min_block * self.block_size
        maximum = (max_block + 1) * self.block_size - 1
        return (
            (int(minimum[0]), int(minimum[1]), int(minimum[2])),
            (int(maximum[0]), int(maximum[1]), int(maximum[2])),
        )

    def iter_allocated_blocks(self) -> Iterator[Tuple[BlockIndex, VoxelBlock]]:
        """Iterate over allocated sparse blocks and their block indices."""
        yield from self.blocks.items()

    def iter_observed_voxels(self) -> Iterator[Tuple[VoxelIndex, GeometryVoxel]]:
        """Iterate over observed voxels as global voxel indices and snapshots."""
        for block_index, block in self.blocks.items():
            for local in np.argwhere(block.observed):
                lx, ly, lz = (int(local[0]), int(local[1]), int(local[2]))
                index = (
                    block_index[0] * self.block_size + lx,
                    block_index[1] * self.block_size + ly,
                    block_index[2] * self.block_size + lz,
                )
                yield index, block.voxel((lx, ly, lz))

    def iter_occupied_voxels(
        self,
        threshold: Optional[float] = None,
    ) -> Iterator[Tuple[VoxelIndex, GeometryVoxel]]:
        """Iterate over observed voxels at or above an occupancy threshold."""
        min_logodds = self._occupied_logodds if threshold is None else prob_to_logodds(threshold)
        for index, voxel in self.iter_observed_voxels():
            if voxel.occupancy_logodds >= min_logodds:
                yield index, voxel

    def iter_free_voxels(
        self,
        threshold: Optional[float] = None,
    ) -> Iterator[Tuple[VoxelIndex, GeometryVoxel]]:
        """Iterate over observed voxels at or below a free-space threshold."""
        max_logodds = self._free_logodds if threshold is None else prob_to_logodds(threshold)
        for index, voxel in self.iter_observed_voxels():
            if voxel.occupancy_logodds <= max_logodds:
                yield index, voxel

    def _update_voxel(self, index: VoxelIndex, update: float, step_index: int) -> None:
        self._update_voxel_fast(index[0], index[1], index[2], update, step_index)

    def _depth_endpoints_world(
        self,
        depth_image: np.ndarray,
        camera_intrinsics: CameraIntrinsics,
        T_world_camera: np.ndarray,
        *,
        pixel_stride: int,
        min_depth: Optional[float],
        max_depth: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if pixel_stride <= 0:
            raise ValueError("pixel_stride must be positive")
        transform = np.asarray(T_world_camera, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("T_world_camera must have shape (4, 4)")

        depth = np.asarray(depth_image)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2:
            raise ValueError("depth_image must be 2D or have a singleton channel")

        origin = transform[:3, 3].astype(np.float64, copy=False)
        rotation = transform[:3, :3].astype(np.float64, copy=False)
        height, width = depth.shape
        sampled = np.asarray(depth[0:height:pixel_stride, 0:width:pixel_stride])
        z_values = sampled.astype(np.float64, copy=False) * camera_intrinsics.depth_scale
        valid = np.isfinite(z_values) & (z_values > 0.0)
        if min_depth is not None:
            valid &= z_values >= min_depth
        if max_depth is not None:
            valid &= z_values <= max_depth
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float64), origin

        v_coords = np.arange(0, height, pixel_stride, dtype=np.float64)
        u_coords = np.arange(0, width, pixel_stride, dtype=np.float64)
        uu, vv = np.meshgrid(u_coords, v_coords)
        z_valid = z_values[valid]
        points_camera = np.empty((len(z_valid), 3), dtype=np.float64)
        points_camera[:, 0] = (
            (uu[valid] - camera_intrinsics.cx) * z_valid / camera_intrinsics.fx
        )
        points_camera[:, 1] = (
            (vv[valid] - camera_intrinsics.cy) * z_valid / camera_intrinsics.fy
        )
        points_camera[:, 2] = z_valid
        return points_camera @ rotation.T + origin, origin

    def _integrate_ray_fast(
        self,
        start_x: float,
        start_y: float,
        start_z: float,
        end_x: float,
        end_y: float,
        end_z: float,
        step_index: int,
    ) -> None:
        dx = end_x - start_x
        dy = end_y - start_y
        dz = end_z - start_z
        length_sq = dx * dx + dy * dy + dz * dz
        if length_sq <= 0.0:
            return

        if self.max_ray_length is not None:
            max_length_sq = self.max_ray_length * self.max_ray_length
            if length_sq > max_length_sq:
                scale = self.max_ray_length / (length_sq**0.5)
                end_x = start_x + dx * scale
                end_y = start_y + dy * scale
                end_z = start_z + dz * scale
                dx = end_x - start_x
                dy = end_y - start_y
                dz = end_z - start_z

        voxel_size = self.voxel_size
        inv_voxel_size = self._inv_voxel_size
        current_x = int(floor(start_x * inv_voxel_size))
        current_y = int(floor(start_y * inv_voxel_size))
        current_z = int(floor(start_z * inv_voxel_size))
        target_x = int(floor(end_x * inv_voxel_size))
        target_y = int(floor(end_y * inv_voxel_size))
        target_z = int(floor(end_z * inv_voxel_size))

        if (
            current_x == target_x
            and current_y == target_y
            and current_z == target_z
        ):
            self._update_voxel_fast(
                current_x,
                current_y,
                current_z,
                self._occ_update,
                step_index,
            )
            return

        step_x = step_y = step_z = 0
        t_max_x = t_max_y = t_max_z = inf
        t_delta_x = t_delta_y = t_delta_z = inf

        if dx > 0.0:
            step_x = 1
            t_max_x = ((current_x + 1) * voxel_size - start_x) / dx
            t_delta_x = voxel_size / dx
        elif dx < 0.0:
            step_x = -1
            t_max_x = (current_x * voxel_size - start_x) / dx
            t_delta_x = -voxel_size / dx

        if dy > 0.0:
            step_y = 1
            t_max_y = ((current_y + 1) * voxel_size - start_y) / dy
            t_delta_y = voxel_size / dy
        elif dy < 0.0:
            step_y = -1
            t_max_y = (current_y * voxel_size - start_y) / dy
            t_delta_y = -voxel_size / dy

        if dz > 0.0:
            step_z = 1
            t_max_z = ((current_z + 1) * voxel_size - start_z) / dz
            t_delta_z = voxel_size / dz
        elif dz < 0.0:
            step_z = -1
            t_max_z = (current_z * voxel_size - start_z) / dz
            t_delta_z = -voxel_size / dz

        free_update = self._free_update
        occ_update = self._occ_update
        block_size = self.block_size
        blocks = self.blocks
        logodds_min = self.logodds_min
        logodds_max = self.logodds_max
        while (
            current_x != target_x
            or current_y != target_y
            or current_z != target_z
        ):
            block_index = (
                current_x // block_size,
                current_y // block_size,
                current_z // block_size,
            )
            block = blocks.get(block_index)
            if block is None:
                block = VoxelBlock(block_size)
                blocks[block_index] = block
            lx = current_x % block_size
            ly = current_y % block_size
            lz = current_z % block_size
            value = float(block.occupancy_logodds[lx, ly, lz]) + free_update
            if value < logodds_min:
                value = logodds_min
            elif value > logodds_max:
                value = logodds_max
            block.occupancy_logodds[lx, ly, lz] = value
            block.observed[lx, ly, lz] = True
            block.last_update_step[lx, ly, lz] = step_index

            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                current_x += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                current_y += step_y
                t_max_y += t_delta_y
            else:
                current_z += step_z
                t_max_z += t_delta_z

        block_index = (
            target_x // block_size,
            target_y // block_size,
            target_z // block_size,
        )
        block = blocks.get(block_index)
        if block is None:
            block = VoxelBlock(block_size)
            blocks[block_index] = block
        lx = target_x % block_size
        ly = target_y % block_size
        lz = target_z % block_size
        value = float(block.occupancy_logodds[lx, ly, lz]) + occ_update
        if value < logodds_min:
            value = logodds_min
        elif value > logodds_max:
            value = logodds_max
        block.occupancy_logodds[lx, ly, lz] = value
        block.observed[lx, ly, lz] = True
        block.last_update_step[lx, ly, lz] = step_index

    def _warm_up_numba_backend(self) -> None:
        try:
            from object_nav.mapping.raycast_numba import (
                NUMBA_AVAILABLE,
                warm_up_numba_raycast,
            )
        except Exception as exc:
            if self.raycast_backend == "numba":
                raise RuntimeError("Numba raycast backend is unavailable") from exc
            return

        if not NUMBA_AVAILABLE:
            if self.raycast_backend == "numba":
                raise RuntimeError("Numba raycast backend is unavailable")
            return
        warm_up_numba_raycast()

    def _integrate_frame_updates_numba(
        self,
        origin: np.ndarray,
        points_world: np.ndarray,
        step_index: int,
    ) -> bool:
        if self.raycast_backend == "python":
            return False

        try:
            from object_nav.mapping.raycast_numba import (
                NUMBA_AVAILABLE,
                raycast_frame_keys,
            )
        except Exception as exc:
            if self.raycast_backend == "numba":
                raise RuntimeError("Numba raycast backend is unavailable") from exc
            return False

        if not NUMBA_AVAILABLE:
            if self.raycast_backend == "numba":
                raise RuntimeError("Numba raycast backend is unavailable")
            return False

        free_keys, occupied_keys, packed_ok = raycast_frame_keys(
            origin,
            points_world,
            voxel_size=self.voxel_size,
            max_ray_length=self.max_ray_length,
        )
        if not packed_ok:
            if self.raycast_backend == "numba":
                raise RuntimeError("Numba raycast backend exceeded packed voxel range")
            return False

        free_keys, free_counts = _packed_keys_to_counts(free_keys)
        occupied_keys, occupied_counts = _packed_keys_to_counts(occupied_keys)
        self._apply_packed_frame_updates(
            free_keys,
            free_counts,
            occupied_keys,
            occupied_counts,
            step_index,
        )
        return True

    def _accumulate_ray_updates(
        self,
        start_x: float,
        start_y: float,
        start_z: float,
        end_x: float,
        end_y: float,
        end_z: float,
        free_counts: dict[VoxelIndex, int],
        occupied_counts: dict[VoxelIndex, int],
    ) -> None:
        dx = end_x - start_x
        dy = end_y - start_y
        dz = end_z - start_z
        length_sq = dx * dx + dy * dy + dz * dz
        if length_sq <= 0.0:
            return

        if self.max_ray_length is not None:
            max_length_sq = self.max_ray_length * self.max_ray_length
            if length_sq > max_length_sq:
                scale = self.max_ray_length / (length_sq**0.5)
                end_x = start_x + dx * scale
                end_y = start_y + dy * scale
                end_z = start_z + dz * scale
                dx = end_x - start_x
                dy = end_y - start_y
                dz = end_z - start_z

        voxel_size = self.voxel_size
        inv_voxel_size = self._inv_voxel_size
        current_x = int(floor(start_x * inv_voxel_size))
        current_y = int(floor(start_y * inv_voxel_size))
        current_z = int(floor(start_z * inv_voxel_size))
        target_x = int(floor(end_x * inv_voxel_size))
        target_y = int(floor(end_y * inv_voxel_size))
        target_z = int(floor(end_z * inv_voxel_size))

        if (
            current_x == target_x
            and current_y == target_y
            and current_z == target_z
        ):
            occupied_counts[(current_x, current_y, current_z)] += 1
            return

        step_x = step_y = step_z = 0
        t_max_x = t_max_y = t_max_z = inf
        t_delta_x = t_delta_y = t_delta_z = inf

        if dx > 0.0:
            step_x = 1
            t_max_x = ((current_x + 1) * voxel_size - start_x) / dx
            t_delta_x = voxel_size / dx
        elif dx < 0.0:
            step_x = -1
            t_max_x = (current_x * voxel_size - start_x) / dx
            t_delta_x = -voxel_size / dx

        if dy > 0.0:
            step_y = 1
            t_max_y = ((current_y + 1) * voxel_size - start_y) / dy
            t_delta_y = voxel_size / dy
        elif dy < 0.0:
            step_y = -1
            t_max_y = (current_y * voxel_size - start_y) / dy
            t_delta_y = -voxel_size / dy

        if dz > 0.0:
            step_z = 1
            t_max_z = ((current_z + 1) * voxel_size - start_z) / dz
            t_delta_z = voxel_size / dz
        elif dz < 0.0:
            step_z = -1
            t_max_z = (current_z * voxel_size - start_z) / dz
            t_delta_z = -voxel_size / dz

        while (
            current_x != target_x
            or current_y != target_y
            or current_z != target_z
        ):
            free_counts[(current_x, current_y, current_z)] += 1
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                current_x += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                current_y += step_y
                t_max_y += t_delta_y
            else:
                current_z += step_z
                t_max_z += t_delta_z

        occupied_counts[(target_x, target_y, target_z)] += 1

    def _apply_frame_updates(
        self,
        free_counts: dict[VoxelIndex, int],
        occupied_counts: dict[VoxelIndex, int],
        step_index: int,
    ) -> None:
        touched = set(free_counts)
        touched.update(occupied_counts)
        for x, y, z in touched:
            update = (
                free_counts.get((x, y, z), 0) * self._free_update
                + occupied_counts.get((x, y, z), 0) * self._occ_update
            )
            self._update_voxel_fast(x, y, z, update, step_index)

    def _apply_packed_frame_updates(
        self,
        free_keys: np.ndarray,
        free_counts: np.ndarray,
        occupied_keys: np.ndarray,
        occupied_counts: np.ndarray,
        step_index: int,
    ) -> None:
        if len(free_keys) == 0 and len(occupied_keys) == 0:
            return

        keys = np.concatenate((free_keys, occupied_keys))
        updates = np.concatenate(
            (
                free_counts.astype(np.float64) * self._free_update,
                occupied_counts.astype(np.float64) * self._occ_update,
            )
        )
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        unique_updates = np.zeros(len(unique_keys), dtype=np.float64)
        np.add.at(unique_updates, inverse, updates)

        x, y, z = _unpack_voxel_keys(unique_keys)
        block_size = self.block_size
        block_x = x // block_size
        block_y = y // block_size
        block_z = z // block_size
        block_keys = _pack_voxel_components(block_x, block_y, block_z)
        if block_keys is None:
            for vx, vy, vz, update in zip(x, y, z, unique_updates):
                self._update_voxel_fast(
                    int(vx),
                    int(vy),
                    int(vz),
                    float(update),
                    step_index,
                )
            return

        order = np.argsort(block_keys)
        sorted_block_keys = block_keys[order]
        sorted_x = x[order]
        sorted_y = y[order]
        sorted_z = z[order]
        sorted_updates = unique_updates[order]
        boundaries = np.flatnonzero(sorted_block_keys[1:] != sorted_block_keys[:-1]) + 1
        starts = np.concatenate((np.array([0]), boundaries))
        stops = np.concatenate((boundaries, np.array([len(sorted_block_keys)])))

        for start, stop in zip(starts, stops):
            bx, by, bz = _unpack_single_voxel_key(int(sorted_block_keys[start]))
            block_index = (bx, by, bz)
            block = self.blocks.get(block_index)
            if block is None:
                block = VoxelBlock(block_size)
                self.blocks[block_index] = block

            local_x = sorted_x[start:stop] % block_size
            local_y = sorted_y[start:stop] % block_size
            local_z = sorted_z[start:stop] % block_size
            values = block.occupancy_logodds[local_x, local_y, local_z]
            values = np.clip(
                values + sorted_updates[start:stop],
                self.logodds_min,
                self.logodds_max,
            )
            block.occupancy_logodds[local_x, local_y, local_z] = values
            block.observed[local_x, local_y, local_z] = True
            block.last_update_step[local_x, local_y, local_z] = step_index

    def _update_voxel_fast(
        self,
        x: int,
        y: int,
        z: int,
        update: float,
        step_index: int,
    ) -> None:
        block_size = self.block_size
        block_index = (x // block_size, y // block_size, z // block_size)
        block = self.blocks.get(block_index)
        if block is None:
            block = VoxelBlock(block_size)
            self.blocks[block_index] = block

        lx = x % block_size
        ly = y % block_size
        lz = z % block_size
        value = float(block.occupancy_logodds[lx, ly, lz]) + update
        if value < self.logodds_min:
            value = self.logodds_min
        elif value > self.logodds_max:
            value = self.logodds_max
        block.occupancy_logodds[lx, ly, lz] = value
        block.observed[lx, ly, lz] = True
        block.last_update_step[lx, ly, lz] = step_index

    def _voxel_is_free(self, voxel: GeometryVoxel) -> bool:
        return voxel.observed and voxel.occupancy_logodds <= self._free_logodds

    def _voxel_is_occupied(self, voxel: GeometryVoxel) -> bool:
        return voxel.observed and voxel.occupancy_logodds >= self._occupied_logodds


def _world_to_voxel_index(point_world: Sequence[float], voxel_size: float) -> VoxelIndex:
    point = np.asarray(point_world, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("point_world must be a 3D point")
    return (
        int(floor(point[0] / voxel_size)),
        int(floor(point[1] / voxel_size)),
        int(floor(point[2] / voxel_size)),
    )


def _axis_with_smallest_t(t_max: Sequence[float]) -> int:
    if t_max[0] <= t_max[1] and t_max[0] <= t_max[2]:
        return 0
    if t_max[1] <= t_max[2]:
        return 1
    return 2


def _valid_depth(
    depth_meters: float,
    *,
    min_depth: Optional[float],
    max_depth: Optional[float],
) -> bool:
    if not isfinite(depth_meters) or depth_meters <= 0.0:
        return False
    if min_depth is not None and depth_meters < min_depth:
        return False
    if max_depth is not None and depth_meters > max_depth:
        return False
    return True


def _add_voxel_array_counts(
    counts: dict[VoxelIndex, int],
    voxels: np.ndarray,
) -> None:
    if len(voxels) == 0:
        return

    packed_counts = _voxel_array_to_packed_counts(voxels)
    if packed_counts is not None:
        unique_keys, unique_counts = packed_counts
        xs, ys, zs = _unpack_voxel_keys(unique_keys)
        for x, y, z, count in zip(xs, ys, zs, unique_counts):
            counts[(int(x), int(y), int(z))] += int(count)
        return

    unique_voxels, unique_counts = np.unique(voxels, axis=0, return_counts=True)
    for voxel, count in zip(unique_voxels, unique_counts):
        counts[(int(voxel[0]), int(voxel[1]), int(voxel[2]))] += int(count)


def _voxel_array_to_packed_counts(
    voxels: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if len(voxels) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    keys = _pack_voxel_components(voxels[:, 0], voxels[:, 1], voxels[:, 2])
    if keys is None:
        return None
    unique_keys, unique_counts = np.unique(keys, return_counts=True)
    return unique_keys, unique_counts


def _packed_keys_to_counts(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(keys) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return np.unique(keys, return_counts=True)


def _pack_voxel_components(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Optional[np.ndarray]:
    if not _components_fit_packed_range(x, y, z):
        return None
    shifted_x = x.astype(np.int64, copy=False) + _PACKED_VOXEL_SHIFT
    shifted_y = y.astype(np.int64, copy=False) + _PACKED_VOXEL_SHIFT
    shifted_z = z.astype(np.int64, copy=False) + _PACKED_VOXEL_SHIFT
    return (
        (shifted_x << (_PACKED_VOXEL_BITS * 2))
        | (shifted_y << _PACKED_VOXEL_BITS)
        | shifted_z
    )


def _unpack_voxel_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = ((keys >> (_PACKED_VOXEL_BITS * 2)) & _PACKED_VOXEL_MASK) - _PACKED_VOXEL_SHIFT
    y = ((keys >> _PACKED_VOXEL_BITS) & _PACKED_VOXEL_MASK) - _PACKED_VOXEL_SHIFT
    z = (keys & _PACKED_VOXEL_MASK) - _PACKED_VOXEL_SHIFT
    return x.astype(np.int64), y.astype(np.int64), z.astype(np.int64)


def _unpack_single_voxel_key(key: int) -> VoxelIndex:
    x = ((key >> (_PACKED_VOXEL_BITS * 2)) & _PACKED_VOXEL_MASK) - _PACKED_VOXEL_SHIFT
    y = ((key >> _PACKED_VOXEL_BITS) & _PACKED_VOXEL_MASK) - _PACKED_VOXEL_SHIFT
    z = (key & _PACKED_VOXEL_MASK) - _PACKED_VOXEL_SHIFT
    return int(x), int(y), int(z)


def _components_fit_packed_range(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> bool:
    lower = -_PACKED_VOXEL_SHIFT
    upper = _PACKED_VOXEL_SHIFT - 1
    return bool(
        np.all((x >= lower) & (x <= upper))
        and np.all((y >= lower) & (y <= upper))
        and np.all((z >= lower) & (z <= upper))
    )


def _horizontal_axes(vertical_axis: int) -> Tuple[int, int]:
    axes = tuple(axis for axis in (0, 1, 2) if axis != vertical_axis)
    return (axes[0], axes[1])
