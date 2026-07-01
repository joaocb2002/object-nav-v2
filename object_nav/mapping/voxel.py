from __future__ import annotations

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
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.voxel_size = float(voxel_size)
        self.block_size = int(block_size)
        self.logodds_min = float(logodds_min)
        self.logodds_max = float(logodds_max)
        self.occupied_threshold = float(occupied_threshold)
        self.free_threshold = float(free_threshold)
        self.max_ray_length = max_ray_length
        self.blocks: Dict[BlockIndex, VoxelBlock] = {}

        self._occ_update = prob_to_logodds(p_occ)
        self._free_update = prob_to_logodds(p_free)
        self._occupied_logodds = prob_to_logodds(occupied_threshold)
        self._free_logodds = prob_to_logodds(free_threshold)

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
        ray = end - start
        length = float(np.linalg.norm(ray))
        if length <= 0.0:
            return
        if self.max_ray_length is not None and length > self.max_ray_length:
            end = start + ray * (self.max_ray_length / length)

        voxels = raycast_voxels(start, end, self.voxel_size)
        if len(voxels) == 1:
            self._update_voxel(voxels[0], self._occ_update, step_index)
            return

        for index in voxels[:-1]:
            self._update_voxel(index, self._free_update, step_index)
        self._update_voxel(voxels[-1], self._occ_update, step_index)

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
        """Integrate a depth frame using pinhole back-projection.

        Back-projection uses optical camera axes: ``X=(u-cx)z/fx``,
        ``Y=(v-cy)z/fy``, ``Z=z``. The supplied transform maps those camera
        points into the world/map frame. Invalid, zero, NaN, and filtered depths
        are ignored.
        """
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

        origin = transform[:3, 3]
        rotation = transform[:3, :3]
        height, width = depth.shape
        for v in range(0, height, pixel_stride):
            for u in range(0, width, pixel_stride):
                z = float(depth[v, u]) * camera_intrinsics.depth_scale
                if not _valid_depth(z, min_depth=min_depth, max_depth=max_depth):
                    continue

                x = (u - camera_intrinsics.cx) * z / camera_intrinsics.fx
                y = (v - camera_intrinsics.cy) * z / camera_intrinsics.fy
                point_camera = np.array([x, y, z], dtype=np.float64)
                point_world = origin + rotation @ point_camera
                self.integrate_ray(origin, point_world, step_index)

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

        for index, voxel in self.iter_observed_voxels():
            center = self.voxel_index_to_world_center(index)
            vertical = float(center[vertical_axis])
            col = int(floor((center[axes[0]] - origin[0]) / grid_resolution))
            row = int(floor((center[axes[1]] - origin[1]) / grid_resolution))
            if row < 0 or row >= height or col < 0 or col >= width:
                continue

            if obstacle_min_z <= vertical <= obstacle_max_z and self._voxel_is_occupied(voxel):
                data[row, col] = OCCUPIED
            elif (
                data[row, col] != OCCUPIED
                and floor_min_z <= vertical <= floor_max_z
                and self._voxel_is_free(voxel)
            ):
                free_seen[row, col] = True

        data[(data != OCCUPIED) & free_seen] = FREE
        return TopDownGrid(
            data=data,
            origin=origin,
            resolution=grid_resolution,
            axes=axes,
            vertical_axis=vertical_axis,
        )

    def is_observed(self, voxel_index: VoxelIndex) -> bool:
        voxel = self.get_voxel(voxel_index)
        return False if voxel is None else voxel.observed

    def is_free(self, voxel_index: VoxelIndex) -> bool:
        voxel = self.get_voxel(voxel_index)
        return False if voxel is None else self._voxel_is_free(voxel)

    def is_occupied(self, voxel_index: VoxelIndex) -> bool:
        voxel = self.get_voxel(voxel_index)
        return False if voxel is None else self._voxel_is_occupied(voxel)

    def occupancy_probability(self, voxel_index: VoxelIndex) -> Optional[float]:
        voxel = self.get_voxel(voxel_index)
        if voxel is None or not voxel.observed:
            return None
        return float(logodds_to_prob(voxel.occupancy_logodds))

    def num_allocated_blocks(self) -> int:
        return len(self.blocks)

    def num_observed_voxels(self) -> int:
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
        yield from self.blocks.items()

    def iter_observed_voxels(self) -> Iterator[Tuple[VoxelIndex, GeometryVoxel]]:
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
        min_logodds = self._occupied_logodds if threshold is None else prob_to_logodds(threshold)
        for index, voxel in self.iter_observed_voxels():
            if voxel.occupancy_logodds >= min_logodds:
                yield index, voxel

    def iter_free_voxels(
        self,
        threshold: Optional[float] = None,
    ) -> Iterator[Tuple[VoxelIndex, GeometryVoxel]]:
        max_logodds = self._free_logodds if threshold is None else prob_to_logodds(threshold)
        for index, voxel in self.iter_observed_voxels():
            if voxel.occupancy_logodds <= max_logodds:
                yield index, voxel

    def _update_voxel(self, index: VoxelIndex, update: float, step_index: int) -> None:
        block = self.get_or_create_block(self.voxel_index_to_block_index(index))
        local = self.voxel_index_to_local_index(index)
        x, y, z = local
        value = float(block.occupancy_logodds[x, y, z]) + update
        block.occupancy_logodds[x, y, z] = clamp_logodds(
            value,
            minimum=self.logodds_min,
            maximum=self.logodds_max,
        )
        block.observed[x, y, z] = True
        block.last_update_step[x, y, z] = step_index

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


def _horizontal_axes(vertical_axis: int) -> Tuple[int, int]:
    axes = tuple(axis for axis in (0, 1, 2) if axis != vertical_axis)
    return (axes[0], axes[1])
