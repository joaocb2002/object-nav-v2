# Scripts

`scripts/main.py` is the active ObjectNav research/debugging entrypoint. It
intentionally keeps runtime knobs as plain constants or small dataclass defaults
instead of adding a separate project config layer.

## Main Script Knobs

Edit these directly in `scripts/main.py`:

- `HABITAT_LAB_ROOT`: adjacent Habitat-Lab checkout, resolved from this repo.
- `CONFIG`: absolute Habitat-Lab ObjectNav config path derived from that root.
- `SCENE_CONTENT_DIR`: directory containing Habitat ObjectNav per-scene
  `*.json.gz` content files. The script randomly chooses one file and strips
  `.json.gz` before assigning `cfg.habitat.dataset.content_scenes = [SCENE]`.
  Default: `<habitat-lab>/data/datasets/objectnav/hm3d/v2/train/content`
- `RUN_SEED`: seed used to choose the scene and passed to Habitat for episode
  sampling. Default: derived from `time.time_ns()`.
- `NUM_EPISODES`: number of episodes sampled by Habitat's episode iterator.
  Default: `1`
- `DISPLAY.enabled_panels`: ordered dashboard panels. Remove a name to disable
  that panel and skip its lazy display producer, or reorder names to change the
  grid order. The default is RGB, depth, SegFormer, voxel world, and Habitat
  ground-truth map.
- `SCENE`: selected by `choose_random_objectnav_scene(...)` before the Habitat
  env is created. To force one scene, replace that assignment with a fixed id
  such as `"92vYG1q49FY"`.

Habitat still owns task-level settings such as max episode steps, action
definitions, sensor resolution/FOV, depth range, and ObjectNav measurements.
Those live in the Habitat config referenced by `CONFIG`, not in this repo.

## Active SegFormer Perception

`main.py` loads one calibrated SegFormer-B5 instance before creating the Habitat
environment and reuses it for every observation. It checks the runtime camera
against the checkpoint's frozen camera profile before starting. When the
`SegFormer` dashboard panel is enabled, each step produces its colorized
41-class label mask. It does not yet blend an overlay or use segmentation for
navigation.

## Dashboard And Controls

The active images are coupled into one automatically tiled, resizable OpenCV
window by `OpenCVDashboard`. The entries in `DISPLAY.enabled_panels` are stable
panel names and define both selection and order. Each corresponding value in
the loop is a lazy function, so disabled display-only work is not evaluated.
Panel producers remain ordinary BGR renderers owned by mapping, perception, or
the script; adding a future representation does not require changing the
dashboard.

Keyboard controls are:

- `W`: move forward
- `A` / `D`: turn left / right
- up / down arrows: look up / down by the ObjectNav-configured 30° increment
- `F`: stop

The agent uses `cv2.waitKeyEx` because arrow keys are extended GUI key events.

The default checkpoint is configured by `SegFormerConfig`:

```text
<repo>/models/segformer/segformer_b5_mpcat40_calibrated/
```

Install the adjacent inference package in the `habitat` environment before
running the script:

```bash
python3 -m pip install -e "../hm3d-semseg[inference]"
```

The older YOLO pipeline remains importable but is no longer invoked by
`main.py`. Its exact behavior is recorded in `docs/yolo_contract.md`.

## Mapping Defaults

Active depth-to-voxel integration settings are configured by
`HabitatVoxelMapConfig` in `object_nav/mapping/habitat.py`:

- `voxel_size = 0.05`
- `block_size = 16`
- `pixel_stride = 4`
- `max_ray_length = 5.0`
- `obstacle_min_height = 0.20`
- `obstacle_max_height = 1.50`
- `floor_min_height = -0.10`
- `floor_max_height = 0.30`
- `local_view_size_m = 8.0`
- `local_pixels_per_meter = 60`
- `raycast_backend = "auto"`: uses the optional Numba DDA backend when
  installed, otherwise falls back to Python.

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

## Active Voxel Mapping

`scripts/main.py` uses only the standard optimized mapper:

```python
voxel_mapper = HabitatVoxelMapper(cfg)
```

At every step, the current depth observation is integrated once:

```python
voxel_mapper.integrate(env, obs, step)
```

The active visualization lazily renders the optimized map for the dashboard:

```python
"Voxel world": lambda: voxel_mapper.render_maps(...)
```

For one panel at a time, use `render_camera_view(...)` or
`render_topdown_map(...)` on `voxel_mapper`.

The slower reference mapper and comparison helpers remain available in
`object_nav.mapping.comparison` for occasional regression checks, but they are
not part of the normal `main.py` loop.

## Habitat Top-Down Map Defaults

`enable_topdown_map_measure(...)` in `object_nav/mapping/habitat.py` adds
Habitat's visual `top_down_map` metric with:

- `map_resolution = 1024`
- `map_padding = 3`
- source, border, shortest path, viewpoints, goal positions, and goal AABBs
  enabled
- fog of war enabled with `visibility_dist = 5.0` and `fov = 90`

The map is an independent dashboard panel, but `main.py` removes `top_down_map`
from the final printed metric dictionary so large arrays are not dumped to the
terminal.

## Optional Point-Cloud Debugging

Point-cloud debugging is currently not active in `main.py`. If re-enabled,
defaults live in `PointCloudDebugConfig` in `object_nav/mapping/point_cloud.py`:

- `pixel_stride = 8`
- `max_points = 500_000`
- `window_name = "Point cloud debug"`
- `plot_pixels = 800`

The mapping README contains the exact import/reset/integrate/save calls for
wiring this back into an entrypoint.

## Earlier Scripts

The numbered historical experiments now live under `scripts/experiments/`.
Their control flow remains intentionally simple, but repository and Habitat-Lab
paths are resolved from each script's location. They can therefore be launched
from any working directory. See `scripts/experiments/README.md` for their roles
and graphical-session requirements.

## Reproducibility Tools

`scripts/tools/check_yolo_contract.py` verifies the retained stock and patched
YOLO behaviors. Its default `both` mode launches a fresh Python process for each
variant because the softmax patch intentionally changes Ultralytics globally
inside its process:

```bash
python3 scripts/tools/check_yolo_contract.py --mode both --device cpu
```
