"""Optional Numba backend for voxel ray traversal."""

from __future__ import annotations

from typing import Optional

import numpy as np

_PACKED_VOXEL_BITS = 21
_PACKED_VOXEL_SHIFT = 1 << (_PACKED_VOXEL_BITS - 1)
_PACKED_VOXEL_MASK = (1 << _PACKED_VOXEL_BITS) - 1
_PACKED_VOXEL_MIN = -_PACKED_VOXEL_SHIFT
_PACKED_VOXEL_MAX = _PACKED_VOXEL_SHIFT - 1

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional runtime dependency
    njit = None
    NUMBA_AVAILABLE = False


def raycast_frame_voxels(
    origin: np.ndarray,
    endpoints: np.ndarray,
    *,
    voxel_size: float,
    max_ray_length: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw free and occupied voxel indices for a depth frame."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")

    max_length = -1.0 if max_ray_length is None else float(max_ray_length)
    return _raycast_frame_voxels_numba(
        np.asarray(origin, dtype=np.float64),
        np.asarray(endpoints, dtype=np.float64),
        float(voxel_size),
        max_length,
    )


def raycast_frame_keys(
    origin: np.ndarray,
    endpoints: np.ndarray,
    *,
    voxel_size: float,
    max_ray_length: Optional[float],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return packed free and occupied voxel keys for a depth frame."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")

    max_length = -1.0 if max_ray_length is None else float(max_ray_length)
    return _raycast_frame_keys_numba(
        np.asarray(origin, dtype=np.float64),
        np.asarray(endpoints, dtype=np.float64),
        float(voxel_size),
        max_length,
    )


def warm_up_numba_raycast() -> None:
    """Compile the Numba raycaster before the first timed integration step."""
    if not NUMBA_AVAILABLE:
        return
    origin = np.zeros(3, dtype=np.float64)
    endpoints = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    _raycast_frame_keys_numba(origin, endpoints, 1.0, -1.0)


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _raycast_frame_voxels_numba(
        origin: np.ndarray,
        endpoints: np.ndarray,
        voxel_size: float,
        max_ray_length: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        inv_voxel_size = 1.0 / voxel_size
        start_x = origin[0]
        start_y = origin[1]
        start_z = origin[2]
        total_free = 0
        total_occupied = 0

        for ray_index in range(endpoints.shape[0]):
            end_x = endpoints[ray_index, 0]
            end_y = endpoints[ray_index, 1]
            end_z = endpoints[ray_index, 2]
            dx = end_x - start_x
            dy = end_y - start_y
            dz = end_z - start_z
            length_sq = dx * dx + dy * dy + dz * dz
            if length_sq <= 0.0:
                continue

            if max_ray_length > 0.0:
                max_length_sq = max_ray_length * max_ray_length
                if length_sq > max_length_sq:
                    scale = max_ray_length / length_sq**0.5
                    end_x = start_x + dx * scale
                    end_y = start_y + dy * scale
                    end_z = start_z + dz * scale

            current_x = int(np.floor(start_x * inv_voxel_size))
            current_y = int(np.floor(start_y * inv_voxel_size))
            current_z = int(np.floor(start_z * inv_voxel_size))
            target_x = int(np.floor(end_x * inv_voxel_size))
            target_y = int(np.floor(end_y * inv_voxel_size))
            target_z = int(np.floor(end_z * inv_voxel_size))

            total_free += (
                abs(target_x - current_x)
                + abs(target_y - current_y)
                + abs(target_z - current_z)
            )
            total_occupied += 1

        free_voxels = np.empty((total_free, 3), dtype=np.int64)
        occupied_voxels = np.empty((total_occupied, 3), dtype=np.int64)
        free_offset = 0
        occupied_offset = 0

        for ray_index in range(endpoints.shape[0]):
            end_x = endpoints[ray_index, 0]
            end_y = endpoints[ray_index, 1]
            end_z = endpoints[ray_index, 2]
            dx = end_x - start_x
            dy = end_y - start_y
            dz = end_z - start_z
            length_sq = dx * dx + dy * dy + dz * dz
            if length_sq <= 0.0:
                continue

            if max_ray_length > 0.0:
                max_length_sq = max_ray_length * max_ray_length
                if length_sq > max_length_sq:
                    scale = max_ray_length / length_sq**0.5
                    end_x = start_x + dx * scale
                    end_y = start_y + dy * scale
                    end_z = start_z + dz * scale
                    dx = end_x - start_x
                    dy = end_y - start_y
                    dz = end_z - start_z

            current_x = int(np.floor(start_x * inv_voxel_size))
            current_y = int(np.floor(start_y * inv_voxel_size))
            current_z = int(np.floor(start_z * inv_voxel_size))
            target_x = int(np.floor(end_x * inv_voxel_size))
            target_y = int(np.floor(end_y * inv_voxel_size))
            target_z = int(np.floor(end_z * inv_voxel_size))

            step_x = 0
            step_y = 0
            step_z = 0
            t_max_x = np.inf
            t_max_y = np.inf
            t_max_z = np.inf
            t_delta_x = np.inf
            t_delta_y = np.inf
            t_delta_z = np.inf

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
                free_voxels[free_offset, 0] = current_x
                free_voxels[free_offset, 1] = current_y
                free_voxels[free_offset, 2] = current_z
                free_offset += 1

                if t_max_x <= t_max_y and t_max_x <= t_max_z:
                    current_x += step_x
                    t_max_x += t_delta_x
                elif t_max_y <= t_max_z:
                    current_y += step_y
                    t_max_y += t_delta_y
                else:
                    current_z += step_z
                    t_max_z += t_delta_z

            occupied_voxels[occupied_offset, 0] = target_x
            occupied_voxels[occupied_offset, 1] = target_y
            occupied_voxels[occupied_offset, 2] = target_z
            occupied_offset += 1

        return free_voxels, occupied_voxels

    @njit(cache=True)
    def _pack_voxel_key_numba(x: int, y: int, z: int) -> tuple[np.int64, bool]:
        if (
            x < _PACKED_VOXEL_MIN
            or x > _PACKED_VOXEL_MAX
            or y < _PACKED_VOXEL_MIN
            or y > _PACKED_VOXEL_MAX
            or z < _PACKED_VOXEL_MIN
            or z > _PACKED_VOXEL_MAX
        ):
            return np.int64(0), False

        shifted_x = np.int64(x + _PACKED_VOXEL_SHIFT)
        shifted_y = np.int64(y + _PACKED_VOXEL_SHIFT)
        shifted_z = np.int64(z + _PACKED_VOXEL_SHIFT)
        return (
            (shifted_x << (_PACKED_VOXEL_BITS * 2))
            | (shifted_y << _PACKED_VOXEL_BITS)
            | shifted_z,
            True,
        )

    @njit(cache=True)
    def _raycast_frame_keys_numba(
        origin: np.ndarray,
        endpoints: np.ndarray,
        voxel_size: float,
        max_ray_length: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        inv_voxel_size = 1.0 / voxel_size
        start_x = origin[0]
        start_y = origin[1]
        start_z = origin[2]
        total_free = 0
        total_occupied = 0

        for ray_index in range(endpoints.shape[0]):
            end_x = endpoints[ray_index, 0]
            end_y = endpoints[ray_index, 1]
            end_z = endpoints[ray_index, 2]
            dx = end_x - start_x
            dy = end_y - start_y
            dz = end_z - start_z
            length_sq = dx * dx + dy * dy + dz * dz
            if length_sq <= 0.0:
                continue

            if max_ray_length > 0.0:
                max_length_sq = max_ray_length * max_ray_length
                if length_sq > max_length_sq:
                    scale = max_ray_length / length_sq**0.5
                    end_x = start_x + dx * scale
                    end_y = start_y + dy * scale
                    end_z = start_z + dz * scale

            current_x = int(np.floor(start_x * inv_voxel_size))
            current_y = int(np.floor(start_y * inv_voxel_size))
            current_z = int(np.floor(start_z * inv_voxel_size))
            target_x = int(np.floor(end_x * inv_voxel_size))
            target_y = int(np.floor(end_y * inv_voxel_size))
            target_z = int(np.floor(end_z * inv_voxel_size))

            total_free += (
                abs(target_x - current_x)
                + abs(target_y - current_y)
                + abs(target_z - current_z)
            )
            total_occupied += 1

        free_keys = np.empty(total_free, dtype=np.int64)
        occupied_keys = np.empty(total_occupied, dtype=np.int64)
        free_offset = 0
        occupied_offset = 0
        packed_ok = True

        for ray_index in range(endpoints.shape[0]):
            end_x = endpoints[ray_index, 0]
            end_y = endpoints[ray_index, 1]
            end_z = endpoints[ray_index, 2]
            dx = end_x - start_x
            dy = end_y - start_y
            dz = end_z - start_z
            length_sq = dx * dx + dy * dy + dz * dz
            if length_sq <= 0.0:
                continue

            if max_ray_length > 0.0:
                max_length_sq = max_ray_length * max_ray_length
                if length_sq > max_length_sq:
                    scale = max_ray_length / length_sq**0.5
                    end_x = start_x + dx * scale
                    end_y = start_y + dy * scale
                    end_z = start_z + dz * scale
                    dx = end_x - start_x
                    dy = end_y - start_y
                    dz = end_z - start_z

            current_x = int(np.floor(start_x * inv_voxel_size))
            current_y = int(np.floor(start_y * inv_voxel_size))
            current_z = int(np.floor(start_z * inv_voxel_size))
            target_x = int(np.floor(end_x * inv_voxel_size))
            target_y = int(np.floor(end_y * inv_voxel_size))
            target_z = int(np.floor(end_z * inv_voxel_size))

            step_x = 0
            step_y = 0
            step_z = 0
            t_max_x = np.inf
            t_max_y = np.inf
            t_max_z = np.inf
            t_delta_x = np.inf
            t_delta_y = np.inf
            t_delta_z = np.inf

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
                key, ok = _pack_voxel_key_numba(current_x, current_y, current_z)
                free_keys[free_offset] = key
                packed_ok = packed_ok and ok
                free_offset += 1

                if t_max_x <= t_max_y and t_max_x <= t_max_z:
                    current_x += step_x
                    t_max_x += t_delta_x
                elif t_max_y <= t_max_z:
                    current_y += step_y
                    t_max_y += t_delta_y
                else:
                    current_z += step_z
                    t_max_z += t_delta_z

            key, ok = _pack_voxel_key_numba(target_x, target_y, target_z)
            occupied_keys[occupied_offset] = key
            packed_ok = packed_ok and ok
            occupied_offset += 1

        return free_keys, occupied_keys, packed_ok

else:

    def _raycast_frame_voxels_numba(
        origin: np.ndarray,
        endpoints: np.ndarray,
        voxel_size: float,
        max_ray_length: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("Numba is not available")

    def _raycast_frame_keys_numba(
        origin: np.ndarray,
        endpoints: np.ndarray,
        voxel_size: float,
        max_ray_length: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        raise RuntimeError("Numba is not available")
