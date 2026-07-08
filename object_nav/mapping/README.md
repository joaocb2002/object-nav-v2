# Sparse Voxel Mapping

`object_nav.mapping` turns Habitat depth observations into a sparse 3D geometry
map and renders debugging views from that map. The package is intentionally split
by responsibility:

- `voxel.py`: Habitat-free mapping math and storage.
- `habitat.py`: Habitat adapter code for depth units, camera pose, camera
  intrinsics, and ground-truth top-down map metrics.
- `comparison.py`: optional side-by-side comparison of the standard voxel map
  against the slower reference integration.
- `raycast_numba.py`: optional native DDA ray traversal backend.
- `visualization.py`: OpenCV renderers for the voxel map and Habitat map.
- `point_cloud.py`: optional RGB-D point-cloud debugging utilities.

The design keeps the reusable mapping core independent from Habitat. `main.py`
owns the experiment flow and calls the adapter at the points where Habitat data
exists.

## Public Imports

The package root exports the Habitat-free voxel primitives:

```python
from object_nav.mapping import SparseVoxelMap, TopDownGrid, raycast_voxels
```

Habitat and debug helpers are imported from their modules so lightweight imports
do not pull in optional runtime dependencies:

```python
from object_nav.mapping.habitat import HabitatVoxelMapper, show_habitat_topdown_map
from object_nav.mapping.point_cloud import HabitatPointCloudRecorder
```

## Runtime Flow

In `scripts/main.py`, each episode creates or resets the same pieces:

1. Habitat config is loaded with `get_config`.
2. `enable_topdown_map_measure(cfg)` adds Habitat's `TopDownMap` measure. This is
   what makes the complete ground-truth map available through `env.get_metrics()`.
3. `HabitatVoxelMapper(cfg)` reads the configured depth sensor parameters and
   creates an empty `SparseVoxelMap`.
4. At every step:
   - `print_observations(obs)` prints non-image observation values.
   - `voxel_mapper.integrate(env, obs, step)` inserts the current depth frame
     into the sparse 3D voxel map.
   - YOLO runs on `obs["rgb"]`.
   - `show_depth_rgb_detections(...)` shows Habitat depth beside RGB detections.
   - `show_navigation_maps(voxel_mapper.render_maps(...))` shows:
     - a front-facing 3D voxel view from the robot camera,
     - our voxel-derived egocentric top-down map.
   - `show_habitat_topdown_map(...)` independently shows Habitat's
     ground-truth top-down map when desired.
   - the active agent returns an action, such as `move_forward`.
   - `env.step(action)` advances Habitat.

## Core Geometry Map

`SparseVoxelMap` in `voxel.py` stores geometry only. It does not store classes,
object ids, semantic distributions, or dense global tensors.

```text
SparseVoxelMap
  voxel_size
  block_size
  blocks: dict[BlockIndex, VoxelBlock]

VoxelBlock
  occupancy_logodds: float32[block_size, block_size, block_size]
  observed: bool[block_size, block_size, block_size]
  last_update_step: int32[block_size, block_size, block_size]
```

Unknown space is implicit:

- If a block is missing, everything in that block is unknown.
- If a block exists but a local voxel has `observed=False`, that voxel is still
  unknown.
- Free and occupied voxels are allocated only when a sensor ray touches them.

This is why memory scales with explored space rather than the theoretical scene
volume.

## Coordinate Conventions

The map uses metric world coordinates. Voxel indices are computed with:

```python
floor(world_coordinate / voxel_size)
```

Using floor matters for negative coordinates. Truncation toward zero would put
points just below zero into the wrong voxel.

Habitat world coordinates are meter-scaled and Y-up. The mapper's pinhole depth
integration expects optical camera coordinates:

```text
+X right
+Y down
+Z forward
```

Habitat camera sensor states are OpenGL-like:

```text
+X right
+Y up
-Z forward
```

`depth_camera_transform(...)` in `habitat.py` applies the fixed axis conversion
before passing `T_world_camera` to the mapper.

## Depth Integration

`HabitatVoxelMapper.integrate(...)` is the runtime bridge:

1. It reads `obs["depth"]`.
2. `depth_observation_to_meters(...)` converts Habitat normalized depth back to
   meters when the sensor config has `normalize_depth=True`.
3. `camera_intrinsics_from_sensor_config(...)` derives pinhole intrinsics from
   the configured sensor width, height, and horizontal FOV.
4. `depth_camera_transform(...)` gets the current depth sensor pose from
   `env.sim.get_agent_state().sensor_states`.
5. `SparseVoxelMap.integrate_depth(...)` back-projects sampled pixels, transforms
   each endpoint into world space, raycasts from camera origin to endpoint, and
   applies frame-level aggregated log-odds updates.

For each valid depth ray:

- intermediate voxels receive free-space evidence,
- the endpoint receives occupied evidence,
- voxels behind the endpoint remain unknown.

Invalid values, NaNs, non-positive depths, and depths outside the configured
min/max range are ignored.

`SparseVoxelMap.integrate_depth(...)` is the standard optimized path. It counts
all free/occupied evidence for a frame first and then applies one clamped update
per touched voxel. It still raycasts every sampled depth endpoint, so free-space
evidence preserves the sampled sensor-ray geometry. The old immediate per-ray
path is kept as
`SparseVoxelMap.integrate_depth_reference(...)` for diagnostics and regression
checks.

The frame ray traversal has an optional native backend:

- `raycast_backend="auto"`: use Numba when installed, otherwise Python.
- `raycast_backend="python"`: force the pure-Python DDA loop.
- `raycast_backend="numba"`: require the Numba DDA loop and raise if unavailable.

With the Numba backend, numeric DDA traversal is native-accelerated and emits
packed integer voxel keys directly. Counts are compacted from those keys and
touched voxels are applied in block batches. Sparse block allocation and public
map queries remain Python-owned so the dynamic sparse map stays easy to
maintain.

Use the reference path only when comparing behavior:

```python
voxel_map.integrate_depth(depth, intrinsics, T_world_camera, step)
reference_map.integrate_depth_reference(depth, intrinsics, T_world_camera, step)
```

## Raycasting And Occupancy

`raycast_voxels(...)` implements 3D DDA / Amanatides-Woo traversal. It returns
voxel indices crossed by a segment in order, including the endpoint, without
duplicates.

Occupancy uses log-odds:

- `prob_to_logodds(p)` converts probabilities to additive evidence.
- `logodds_to_prob(l)` converts stored values back to probabilities.
- `clamp_logodds(l)` keeps values bounded.

Defaults:

- occupied update: `p_occ=0.70`
- free update: `p_free=0.30`
- occupied query threshold: `0.65`
- free query threshold: `0.35`
- clamp range: `[-5.0, 5.0]`

## Top-Down Projection

`SparseVoxelMap.build_topdown_occupancy(...)` collapses observed 3D geometry into
a 2D planning grid. With Habitat, `HabitatVoxelMapper.build_topdown_projection`
sets `vertical_axis=1` because Habitat is Y-up.

For each horizontal cell:

- if any observed occupied voxel is in the obstacle-height band, the cell is an
  obstacle,
- else if a near-floor voxel is observed free, the cell is free,
- otherwise the cell remains unknown.

This projection is our map. It is not Habitat ground truth.

Habitat's ground-truth map is different: `enable_topdown_map_measure(...)` adds
the official `TopDownMap` metric, and `render_ground_truth_topdown_bgr(...)`
renders the current `env.get_metrics()["top_down_map"]`. Habitat updates that
measure during the episode; it is not restricted to the final step.

## Habitat Metrics

The ObjectNav benchmark config used by `scripts/main.py` already includes the
standard ObjectNav measurements:

- `distance_to_goal`
- `success`
- `spl`
- `soft_spl`
- `distance_to_goal_reward`

`enable_topdown_map_measure(cfg)` adds `top_down_map` manually because it is a
visual debugging measurement, not required for the agent action loop. Habitat
also provides navigation measurements such as `collisions` and step-count style
measurements in its task registry/config store; add them through the Habitat
config when a run needs them, rather than duplicating that state in this package.

## Reference Comparison API

`comparison.py` provides optional diagnostics without owning the standard map.
The active `scripts/main.py` does not use these classes during normal runs.

- `HabitatVoxelMapper`: the standard optimized mapper, using
  `SparseVoxelMap.integrate_depth`.
- `HabitatReferenceVoxelMapper`: the slower baseline mapper, using
  `SparseVoxelMap.integrate_depth_reference`.
- `HabitatVoxelMapComparison`: a metric/rendering helper that reads both maps
  after they have been independently updated.

Minimal usage in a temporary experiment:

```python
import time

from object_nav.mapping.comparison import HabitatReferenceVoxelMapper, HabitatVoxelMapComparison, format_voxel_difference
from object_nav.mapping.habitat import HabitatVoxelMapper

voxel_mapper = HabitatVoxelMapper(cfg)
reference_mapper = HabitatReferenceVoxelMapper(cfg)
voxel_comparison = HabitatVoxelMapComparison()

...
t0 = time.perf_counter()
voxel_mapper.integrate(env, obs, step)
map_seconds = time.perf_counter() - t0

t0 = time.perf_counter()
reference_mapper.integrate(env, obs, step)
reference_seconds = time.perf_counter() - t0

voxel_metric = voxel_comparison.compare(
    voxel_mapper.voxel_map,
    reference_mapper.voxel_map,
    step_index=step,
    map_seconds=map_seconds,
    reference_seconds=reference_seconds,
)
print(format_voxel_difference(voxel_metric))
```

The formatted metric reports:

- `map`: standard integration time.
- `reference`: reference integration time.
- `speedup`: `reference / map`.
- `observed(map/reference/union)`: observed voxel counts.
- `only(map/reference)`: voxels allocated by only one map.
- `state_diff`: different discrete states (`free`, `occupied`, or uncertain).
- `free_occ_swaps`: severe disagreements where one map says free and the other
  says occupied.
- `logodds_changed`: shared voxels whose numeric log-odds differ.
- `mean_abs_logodds` and `max_abs_logodds`: continuous disagreement magnitude.

`HabitatVoxelMapComparison.render_maps(...)` can render a single combined
diagnostic panel when needed. Keep this module opt-in: production-style runs
should use only `HabitatVoxelMapper`.

## Rendered Views

`HabitatVoxelMapper` exposes separate render methods:

- `render_camera_view(...)`: render only the robot-perspective 3D voxel view.
- `render_topdown_map(...)`: render only the voxel-derived top-down map.
- `render_maps(...)`: compose those two panels side by side.

The underlying renderers are:

1. `render_voxel_camera_view_bgr(...)`
   Projects observed 3D voxel centers into the current robot camera. This is a
   direct egocentric 2D image of the 3D voxel structure.
2. `render_full_voxel_topdown_from_agent_bgr(...)`
   Samples the full voxel-derived top-down projection in a robot-oriented frame,
   keeping robot-forward upward while showing the whole explored map.
`show_habitat_topdown_map(...)` is intentionally separate. It uses Habitat's
`TopDownMap` visualization utilities to draw the official map, current agent
marker, path/goal overlays, and fog-of-war when configured.

## Point Cloud Debugging

`HabitatPointCloudRecorder` in `point_cloud.py` is an optional debugging helper
for validating RGB-D back-projection independently from voxel occupancy updates.
It is not part of the active `scripts/main.py` loop right now because it can be
memory- and time-heavy.

To re-enable it in an entrypoint, add:

```python
from object_nav.mapping.point_cloud import HabitatPointCloudRecorder
from object_nav.utils import make_run_output_dir
```

Create it beside the voxel mapper:

```python
point_cloud = HabitatPointCloudRecorder(cfg)
point_cloud_paths = []
```

Reset it once per episode:

```python
point_cloud.reset()
```

Integrate it after the voxel mapper has consumed the same observation:

```python
point_cloud.integrate(env, obs)
```

The recorder:

- converts Habitat depth back to meters,
- uses the same configured camera intrinsics as the voxel mapper,
- uses the same `depth_camera_transform(...)` frame conversion,
- samples RGB-D pixels,
- back-projects them into world coordinates,
- stores RGB colors from `obs["rgb"]`.

At the end of the episode, save into a run directory:

```python
output_dir = make_run_output_dir(
    script_path=__file__,
    scene_id=SCENE,
    episode_id=str(env.current_episode.episode_id),
)
ply_path = point_cloud.save(output_dir / "point_cloud.ply")
point_cloud_paths.append(ply_path)
```

After the Habitat environment and OpenCV windows close, inspect saved clouds
interactively:

```python
for ply_path in point_cloud_paths:
    point_cloud.show_interactive(ply_path)
```

`save(...)` writes an ASCII PLY file with `red`, `green`, and `blue` values
copied from the corresponding Habitat RGB pixels. The OpenCV-only helpers
convert RGB to BGR only when drawing PNG images.

The run directory is created under this repository's `outputs/` folder by
default. The directory name includes the timestamp, scene id, episode id, and
script name, and it is built from the repo root rather than the current working
directory. That matters because `main.py` changes directory into the local
Habitat-Lab checkout before creating the environment.

`show_interactive(...)` does not create another artifact. It tries to open a PLY
or in-memory point cloud in an Open3D interactive viewer. If Open3D cannot create
a window, open the saved PLY in CloudCompare, MeshLab, or run:

```bash
python3 tests/plot_point_cloud.py outputs/.../point_cloud.ply
```

The lower-level plotting function also works without the recorder:

```python
show_interactive_point_cloud(Path("cloud.ply"))
show_interactive_point_cloud((points, rgb_colors))
```

`save_static_preview(...)` and `render_point_cloud_summary_bgr(...)` still exist
for quick top/front/side PNG snapshots, but they are no longer the main
inspection path.

This helper is intentionally isolated so it can be removed from `main.py` later
without affecting the voxel map.

## Public Query API

The core map exposes geometry queries:

- `is_observed(index)`
- `is_free(index)`
- `is_occupied(index)`
- `occupancy_probability(index)`
- `num_allocated_blocks()`
- `num_observed_voxels()`
- `get_allocated_bounds()`
- `iter_observed_voxels()`
- `iter_occupied_voxels()`
- `iter_free_voxels()`

These operate on voxel indices, not world points. Use
`world_to_voxel_index(...)` and `voxel_index_to_world_center(...)` to convert.

## Test Coverage

`tests/test_sparse_voxel_map.py` covers the parts that are easy to break quietly:

- coordinate conversion, including negative coordinates,
- probability/log-odds conversion and clamping,
- free/occupied evidence accumulation,
- DDA ray traversal and endpoint inclusion,
- sparse allocation and implicit unknown space,
- synthetic depth integration,
- top-down projection states,
- Habitat depth normalization and intrinsics,
- agent key bindings and random action selection,
- RGB-D point cloud back-projection and PLY writing/loading,
- camera-view voxel rendering smoke behavior.

The tests do not require a running Habitat simulator. Habitat-specific runtime
objects are kept behind adapter functions so the math can be verified cheaply.

## Future Semantic Layer

Semantics should stay separate from this geometry layer. A future semantic map
can be keyed by voxel index, block index, or object instance id and allocated
with the same lazy pattern. Geometry voxels intentionally remain small:
occupancy log-odds, observed flag, and last update step only.
