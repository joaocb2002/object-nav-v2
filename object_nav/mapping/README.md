# Sparse Voxel Mapping

`object_nav.mapping` turns Habitat depth observations into a sparse 3D geometry
map and renders debugging views from that map. The package is intentionally split
into three layers:

- `voxel.py`: Habitat-free mapping math and storage.
- `habitat.py`: Habitat adapter code for depth units, camera pose, camera
  intrinsics, and ground-truth top-down map metrics.
- `visualization.py`: OpenCV renderers for the voxel map and Habitat map.

The design keeps the reusable mapping core independent from Habitat. `main.py`
owns the experiment flow and calls the adapter at the points where Habitat data
exists.

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
     - our voxel-derived egocentric top-down map,
     - Habitat's ground-truth top-down map.
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
   applies log-odds updates.

For each valid depth ray:

- intermediate voxels receive free-space evidence,
- the endpoint receives occupied evidence,
- voxels behind the endpoint remain unknown.

Invalid values, NaNs, non-positive depths, and depths outside the configured
min/max range are ignored.

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

## Rendered Views

`HabitatVoxelMapper.render_maps(...)` returns one BGR image made of three panels:

1. `render_voxel_camera_view_bgr(...)`
   Projects observed 3D voxel centers into the current robot camera. This is a
   direct egocentric 2D image of the 3D voxel structure.
2. `render_voxel_topdown_from_agent_bgr(...)`
   Samples our top-down projection in an egocentric frame, with the robot in the
   center and forward direction up.
3. `render_ground_truth_topdown_bgr(...)`
   Uses Habitat's `TopDownMap` visualization utilities to draw the official map,
   current agent marker, path/goal overlays, and fog-of-war when configured.

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
- camera-view voxel rendering smoke behavior.

The tests do not require a running Habitat simulator. Habitat-specific runtime
objects are kept behind adapter functions so the math can be verified cheaply.

## Future Semantic Layer

Semantics should stay separate from this geometry layer. A future semantic map
can be keyed by voxel index, block index, or object instance id and allocated
with the same lazy pattern. Geometry voxels intentionally remain small:
occupancy log-odds, observed flag, and last update step only.
