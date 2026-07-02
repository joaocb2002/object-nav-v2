# Scripts

`scripts/main.py` is the active ObjectNav demo entrypoint. It intentionally keeps
runtime knobs as plain constants or small dataclass defaults instead of adding a
separate project config layer.

## Main Script Knobs

Edit these directly in `scripts/main.py`:

- `CONFIG`: Habitat-Lab ObjectNav config path.
  Default: `habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml`
- `SCENE`: HM3D scene id selected through `cfg.habitat.dataset.content_scenes`.
  Default: `92vYG1q49FY`
- `NUM_EPISODES`: number of episodes sampled by Habitat's episode iterator.
  Default: `1`
- `cfg.habitat.seed`: currently set from `time.time_ns()` so sampled episodes
  vary between runs. Replace it with a fixed integer for reproducible episode
  sampling.

Habitat still owns task-level settings such as max episode steps, action
definitions, sensor resolution/FOV, depth range, and ObjectNav measurements.
Those live in the Habitat config referenced by `CONFIG`, not in this repo.

## Mapping Defaults

Active depth-to-voxel integration is configured by
`HabitatVoxelMapConfig` in `object_nav/mapping/habitat.py`:

- `voxel_size = 0.10`
- `block_size = 16`
- `pixel_stride = 6`
- `max_ray_length = 5.0`
- `obstacle_min_height = 0.20`
- `obstacle_max_height = 1.50`
- `floor_min_height = -0.10`
- `floor_max_height = 0.30`
- `local_view_size_m = 8.0`
- `local_pixels_per_meter = 60`

To change these without a config file, instantiate the mapper with an explicit
dataclass:

```python
from object_nav.mapping.habitat import HabitatVoxelMapConfig, HabitatVoxelMapper

voxel_mapper = HabitatVoxelMapper(
    cfg,
    config=HabitatVoxelMapConfig(voxel_size=0.05, pixel_stride=4),
)
```

Core occupancy defaults live in `SparseVoxelMap` in `object_nav/mapping/voxel.py`:

- `p_occ = 0.70`
- `p_free = 0.30`
- `logodds_min = -5.0`
- `logodds_max = 5.0`
- `occupied_threshold = 0.65`
- `free_threshold = 0.35`

The active script normally reaches those through `HabitatVoxelMapConfig`.

## Habitat Top-Down Map Defaults

`enable_topdown_map_measure(...)` in `object_nav/mapping/habitat.py` adds
Habitat's visual `top_down_map` metric with:

- `map_resolution = 1024`
- `map_padding = 3`
- source, border, shortest path, viewpoints, goal positions, and goal AABBs
  enabled
- fog of war enabled with `visibility_dist = 5.0` and `fov = 90`

The map is used for visualization, but `main.py` removes `top_down_map` from the
final printed metric dictionary so large arrays are not dumped to the terminal.

## Optional Point-Cloud Debugging

Point-cloud debugging is currently not active in `main.py`. If re-enabled,
defaults live in `PointCloudDebugConfig` in `object_nav/mapping/point_cloud.py`:

- `pixel_stride = 8`
- `max_points = 500_000`
- `window_name = "Point cloud debug"`
- `plot_pixels = 800`

The mapping README contains the exact import/reset/integrate/save calls for
wiring this back into an entrypoint.
