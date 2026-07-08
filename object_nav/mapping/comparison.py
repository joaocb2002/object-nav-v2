from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from object_nav.mapping.habitat import (
    HabitatVoxelMapper,
    depth_camera_transform,
    depth_observation_to_meters,
)
from object_nav.mapping.voxel import (
    FREE,
    OCCUPIED,
    SparseVoxelMap,
)


UNCERTAIN = 2
LOGODDS_DIFF_EPSILON = 1e-6
DEFAULT_LOGODDS_DIFF_BINS = tuple(float(i) * 0.5 for i in range(21))


@dataclass(frozen=True)
class LogOddsDifferenceHistogram:
    """Histogram of nonzero absolute log-odds differences between two maps."""

    bin_edges: tuple[float, ...]
    counts: tuple[int, ...]
    total_different: int


@dataclass(frozen=True)
class VoxelMapDifference:
    """Per-step difference summary between two voxel maps."""

    step_index: int
    map_seconds: float
    reference_seconds: float
    map_observed: int
    reference_observed: int
    union_observed: int
    only_map: int
    only_reference: int
    state_disagreements: int
    opposite_state_disagreements: int
    map_free_reference_occupied: int
    map_occupied_reference_free: int
    state_disagreement_rate: float
    mean_abs_logodds: Optional[float]
    max_abs_logodds: Optional[float]
    logodds_histogram: LogOddsDifferenceHistogram


@dataclass
class HabitatVoxelMapComparison:
    """Compare an independently updated standard map against a reference map."""

    history: list[VoxelMapDifference] = field(default_factory=list, init=False)

    def reset(self) -> None:
        """Clear per-episode comparison history."""
        self.history.clear()

    def compare(
        self,
        voxel_map: SparseVoxelMap,
        reference_map: SparseVoxelMap,
        *,
        step_index: int,
        map_seconds: float,
        reference_seconds: float,
    ) -> VoxelMapDifference:
        """Compare two already-updated maps and store the metric."""
        metric = compare_voxel_maps(
            voxel_map,
            reference_map,
            step_index=step_index,
            map_seconds=map_seconds,
            reference_seconds=reference_seconds,
        )
        self.history.append(metric)
        return metric

    def render_maps(
        self,
        env: Any,
        voxel_mapper: HabitatVoxelMapper,
        reference_mapper: "HabitatReferenceVoxelMapper",
        *,
        output_height: int,
    ) -> np.ndarray:
        """Render standard and reference voxel views side by side."""
        import cv2

        from object_nav.mapping.visualization import (
            render_full_voxel_topdown_from_agent_bgr,
            render_logodds_difference_histogram_bgr,
            render_voxel_camera_view_bgr,
        )

        T_world_camera = depth_camera_transform(env, voxel_mapper.depth_uuid)
        agent_state = env.sim.get_agent_state()
        image_shape = (
            int(voxel_mapper.depth_sensor_config.height),
            int(voxel_mapper.depth_sensor_config.width),
        )
        panels = [
            render_voxel_camera_view_bgr(
                voxel_mapper.voxel_map,
                T_world_camera,
                voxel_mapper.camera_intrinsics,
                image_shape=image_shape,
                output_height=output_height,
                max_depth=voxel_mapper.config.max_ray_length,
                label="Voxel 3D",
            ),
            render_full_voxel_topdown_from_agent_bgr(
                voxel_mapper.build_topdown_projection(env),
                agent_state,
                output_height=output_height,
                label="Voxel 2D",
            ),
            render_voxel_camera_view_bgr(
                reference_mapper.voxel_map,
                T_world_camera,
                reference_mapper.camera_intrinsics,
                image_shape=image_shape,
                output_height=output_height,
                max_depth=reference_mapper.config.max_ray_length,
                label="Reference 3D",
            ),
            render_full_voxel_topdown_from_agent_bgr(
                reference_mapper.build_topdown_projection(env),
                agent_state,
                output_height=output_height,
                label="Reference 2D",
            ),
            render_logodds_difference_histogram_bgr(
                self.history[-1].logodds_histogram if self.history else None,
                output_height=output_height,
            ),
        ]
        height = max(panel.shape[0] for panel in panels)
        resized = []
        for panel in panels:
            if panel.shape[0] == height:
                resized.append(panel)
                continue
            width = int(round(panel.shape[1] * height / panel.shape[0]))
            resized.append(cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA))
        return np.hstack(resized)


class HabitatReferenceVoxelMapper(HabitatVoxelMapper):
    """Habitat adapter for the slower immediate-update reference map."""

    def integrate(
        self,
        env: Any,
        observations: Mapping[str, Any],
        step_index: int,
    ) -> None:
        """Integrate the current Habitat depth observation with reference logic."""
        if self.depth_uuid not in observations:
            return

        depth_meters = depth_observation_to_meters(
            observations[self.depth_uuid],
            self.depth_sensor_config,
        )
        T_world_camera = depth_camera_transform(env, self.depth_uuid)
        self.voxel_map.integrate_depth_reference(
            depth_meters,
            self.camera_intrinsics,
            T_world_camera,
            step_index,
            pixel_stride=self.config.pixel_stride,
            min_depth=float(self.depth_sensor_config.min_depth),
            max_depth=float(self.depth_sensor_config.max_depth),
        )


def compare_voxel_maps(
    voxel_map: SparseVoxelMap,
    reference_map: SparseVoxelMap,
    *,
    step_index: int,
    map_seconds: float,
    reference_seconds: float,
) -> VoxelMapDifference:
    """Compare observed geometry state between the standard map and reference."""
    map_observed = 0
    reference_observed = 0
    only_map = 0
    only_reference = 0
    state_disagreements = 0
    opposite_state_disagreements = 0
    map_free_reference_occupied = 0
    map_occupied_reference_free = 0
    abs_logodds_sum = 0.0
    abs_logodds_count = 0
    max_abs_logodds: Optional[float] = None
    histogram_counts = np.zeros(len(DEFAULT_LOGODDS_DIFF_BINS) - 1, dtype=np.int64)
    different_logodds_count = 0

    for block_index in set(voxel_map.blocks) | set(reference_map.blocks):
        map_block = voxel_map.blocks.get(block_index)
        reference_block = reference_map.blocks.get(block_index)
        if map_block is None:
            reference_mask = reference_block.observed
            reference_count = int(np.count_nonzero(reference_mask))
            reference_observed += reference_count
            only_reference += reference_count
            state_disagreements += reference_count
            continue
        if reference_block is None:
            map_mask = map_block.observed
            map_count = int(np.count_nonzero(map_mask))
            map_observed += map_count
            only_map += map_count
            state_disagreements += map_count
            continue

        map_mask = map_block.observed
        reference_mask = reference_block.observed
        map_count = int(np.count_nonzero(map_mask))
        reference_count = int(np.count_nonzero(reference_mask))
        map_observed += map_count
        reference_observed += reference_count

        common = map_mask & reference_mask
        map_only_mask = map_mask & ~reference_mask
        reference_only_mask = reference_mask & ~map_mask
        map_only_count = int(np.count_nonzero(map_only_mask))
        reference_only_count = int(np.count_nonzero(reference_only_mask))
        only_map += map_only_count
        only_reference += reference_only_count
        state_disagreements += map_only_count + reference_only_count

        if not np.any(common):
            continue

        map_states = _state_array(
            map_block.occupancy_logodds,
            common,
            free_logodds=voxel_map._free_logodds,
            occupied_logodds=voxel_map._occupied_logodds,
        )
        reference_states = _state_array(
            reference_block.occupancy_logodds,
            common,
            free_logodds=reference_map._free_logodds,
            occupied_logodds=reference_map._occupied_logodds,
        )
        state_disagreements += int(
            np.count_nonzero(map_states != reference_states)
        )
        map_free_ref_occ = (map_states == FREE) & (
            reference_states == OCCUPIED
        )
        map_occ_ref_free = (map_states == OCCUPIED) & (
            reference_states == FREE
        )
        map_free_ref_occ_count = int(np.count_nonzero(map_free_ref_occ))
        map_occ_ref_free_count = int(np.count_nonzero(map_occ_ref_free))
        map_free_reference_occupied += map_free_ref_occ_count
        map_occupied_reference_free += map_occ_ref_free_count
        opposite_state_disagreements += (
            map_free_ref_occ_count + map_occ_ref_free_count
        )

        abs_diff = np.abs(
            map_block.occupancy_logodds[common]
            - reference_block.occupancy_logodds[common]
        )
        abs_logodds_sum += float(abs_diff.sum())
        abs_logodds_count += int(abs_diff.size)
        block_max = float(abs_diff.max())
        max_abs_logodds = (
            block_max if max_abs_logodds is None else max(max_abs_logodds, block_max)
        )
        different = abs_diff > LOGODDS_DIFF_EPSILON
        if np.any(different):
            different_values = abs_diff[different]
            histogram_values = np.clip(
                different_values,
                DEFAULT_LOGODDS_DIFF_BINS[0],
                DEFAULT_LOGODDS_DIFF_BINS[-1],
            )
            histogram_counts += np.histogram(
                histogram_values,
                bins=DEFAULT_LOGODDS_DIFF_BINS,
            )[0].astype(np.int64)
            different_logodds_count += int(different_values.size)

    union_count = map_observed + only_reference
    return VoxelMapDifference(
        step_index=step_index,
        map_seconds=map_seconds,
        reference_seconds=reference_seconds,
        map_observed=map_observed,
        reference_observed=reference_observed,
        union_observed=union_count,
        only_map=only_map,
        only_reference=only_reference,
        state_disagreements=state_disagreements,
        opposite_state_disagreements=opposite_state_disagreements,
        map_free_reference_occupied=map_free_reference_occupied,
        map_occupied_reference_free=map_occupied_reference_free,
        state_disagreement_rate=(
            0.0 if union_count == 0 else state_disagreements / union_count
        ),
        mean_abs_logodds=(
            None if abs_logodds_count == 0 else abs_logodds_sum / abs_logodds_count
        ),
        max_abs_logodds=max_abs_logodds,
        logodds_histogram=LogOddsDifferenceHistogram(
            bin_edges=DEFAULT_LOGODDS_DIFF_BINS,
            counts=tuple(int(count) for count in histogram_counts),
            total_different=different_logodds_count,
        ),
    )


def format_voxel_difference(metric: VoxelMapDifference) -> str:
    """Format one comparison metric for compact per-step logging."""
    speedup = (
        float("inf")
        if metric.map_seconds == 0.0
        else metric.reference_seconds / metric.map_seconds
    )
    mean_abs = (
        "n/a"
        if metric.mean_abs_logodds is None
        else f"{metric.mean_abs_logodds:.4f}"
    )
    max_abs = (
        "n/a"
        if metric.max_abs_logodds is None
        else f"{metric.max_abs_logodds:.4f}"
    )
    return (
        "Voxel comparison: "
        f"map={metric.map_seconds:.3f}s "
        f"reference={metric.reference_seconds:.3f}s "
        f"speedup={speedup:.2f}x "
        f"observed(map/reference/union)="
        f"{metric.map_observed}/{metric.reference_observed}/{metric.union_observed} "
        f"only(map/reference)={metric.only_map}/{metric.only_reference} "
        f"state_diff={metric.state_disagreements} "
        f"state_diff_rate={metric.state_disagreement_rate:.4%} "
        f"free_occ_swaps={metric.opposite_state_disagreements}"
        f"(map_free->reference_occ/map_occ->reference_free="
        f"{metric.map_free_reference_occupied}/"
        f"{metric.map_occupied_reference_free}) "
        f"logodds_changed={metric.logodds_histogram.total_different} "
        f"mean_abs_logodds={mean_abs} "
        f"max_abs_logodds={max_abs}"
    )


def format_integration_time_summary(
    map_seconds: Sequence[float],
    reference_seconds: Sequence[float],
) -> str:
    """Format per-episode integration timing statistics."""
    if not map_seconds and not reference_seconds:
        return "Voxel integration time summary: no integration steps"

    return (
        "Voxel integration time summary: "
        f"map {_format_time_stats(map_seconds)} | "
        f"reference {_format_time_stats(reference_seconds)}"
    )


def _format_time_stats(seconds: Sequence[float]) -> str:
    if not seconds:
        return "min=n/a max=n/a avg=n/a"
    values = np.asarray(seconds, dtype=np.float64)
    return (
        f"min={values.min():.3f}s "
        f"max={values.max():.3f}s "
        f"avg={values.mean():.3f}s"
    )


def _state_array(
    logodds: np.ndarray,
    mask: np.ndarray,
    *,
    free_logodds: float,
    occupied_logodds: float,
) -> np.ndarray:
    values = logodds[mask]
    states = np.full(values.shape, UNCERTAIN, dtype=np.int8)
    states[values >= occupied_logodds] = OCCUPIED
    states[values <= free_logodds] = FREE
    return states


def _get_depth_sensor_config(habitat_config: Any, sensor_name: str) -> Any:
    agents = habitat_config.habitat.simulator.agents
    agent_config = (
        agents.main_agent
        if hasattr(agents, "main_agent")
        else next(iter(agents.values()))
    )
    return agent_config.sim_sensors[sensor_name]
