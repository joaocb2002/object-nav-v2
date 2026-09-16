# Scripts

`scripts/main.py` is the active ObjectNav research/debugging entrypoint. It
intentionally keeps runtime knobs as plain constants or small dataclass defaults
instead of adding a separate project config layer.

## Main Script Knobs

Edit these directly in `scripts/main.py`:

- `HABITAT_LAB_ROOT`: adjacent Habitat-Lab checkout, resolved from this repo.
- `OBJECTNAV_CONFIG`: absolute Habitat-Lab ObjectNav config path derived from
  that root.
- `SCENE_CONTENT_DIR`: directory containing Habitat ObjectNav per-scene
  `*.json.gz` content files. The script randomly chooses one file and strips
  `.json.gz` before assigning `cfg.habitat.dataset.content_scenes = [SCENE]`.
  Default: `<habitat-lab>/data/datasets/objectnav/hm3d/v2/train/content`
- `RUN_SEED`: seed used to choose the scene and passed to Habitat for episode
  sampling. Default: derived from `time.time_ns()`.
- `NUM_EPISODES`: number of episodes sampled by Habitat's episode iterator.
  Default: `1`
- `SEGMENTATION_OVERLAY_OPACITY`: color-mask contribution to the RGB overlay.
  `0.0` is the original RGB image, `1.0` is the fully colorized mask, and the
  light default is `0.25`.
- `SHOW_SEGMENTATION_LABELS`: toggle fixed-size semantic class names on the
  standalone SegFormer panel. Default: `True`.
- `DISPLAY.enabled_panels`: ordered dashboard panels. Remove a name to disable
  that panel and skip its lazy display producer, or reorder names to change the
  grid order. The default is RGB + segmentation, depth, standalone SegFormer,
  3D voxel view, voxel top-down map, and Habitat ground-truth map.
- `DISPLAY.fullscreen`: whether the dashboard requests OpenCV fullscreen mode
  when first opened. Default: `True`.
- `SCENE`: selected by `choose_random_objectnav_scene(...)` before the Habitat
  env is created. To force one scene, replace that assignment with a fixed id
  such as `"92vYG1q49FY"`.

Habitat still owns task-level settings such as max episode steps, action
definitions, sensor resolution/FOV, depth range, and ObjectNav measurements.
Those live in the Habitat config referenced by `OBJECTNAV_CONFIG`, not in this
repo.

## Active SegFormer Perception

`main.py` loads one calibrated SegFormer-B5 instance before creating the Habitat
environment and reuses it for every observation. It checks the runtime camera
against the checkpoint's frozen camera profile before starting. When the
`RGB + segmentation` or `SegFormer` dashboard panel is enabled, each step
produces one colorized 41-class label mask. A per-frame cache shares that result
between the lightly blended RGB panel and standalone mask panel. Segmentation is
still visualization-only and is not used for navigation.

When `SHOW_SEGMENTATION_LABELS` is enabled, each present known class gets at
most one label. The renderer finds that class's largest 8-connected component,
uses a distance transform to select its maximum-clearance point, and draws text
there. Text has a thin class-color interior and black outline. Every visible
class is labeled, including unknown and classes represented by a single pixel.
Labels may cross semantic boundaries but are clamped fully inside the image.
Disable the feature with:

```python
SHOW_SEGMENTATION_LABELS = False
```

## Dashboard And Controls

The active images are coupled into one automatically tiled OpenCV window by
`OpenCVDashboard`. `main.py` opens it with a temporary initialization image
before loading SegFormer or constructing the Habitat environment. This forces
OpenCV HighGUI/Qt initialization into a predictable startup phase, so backend
thread diagnostics cannot interrupt the first episode's terminal output. It
also requests fullscreen mode at that point.

The entries in `DISPLAY.enabled_panels` are stable panel names and define both
selection and order. Each corresponding value in the loop is a lazy function,
so disabled display-only work is not evaluated. Panel producers remain ordinary
BGR renderers owned by mapping, perception, or the script; adding a future
representation does not require changing the dashboard.

The overlay itself uses the model-independent `blend_bgr_overlay(...)` utility.
It receives an RGB-derived BGR image and any same-sized color overlay; it has no
dependency on SegFormer or semantic class definitions.

Keyboard controls are:

- `W`: move forward
- `A` / `D`: turn left / right
- up / down arrows: look up / down by the ObjectNav-configured 30° increment
- `F`: stop

The agent uses `cv2.waitKeyEx` because arrow keys are extended GUI key events.
The ObjectNav task includes both look actions, but the checked-out Habitat-Lab
implementation expects a public Habitat-Sim `Agent.sensors` attribute that is
absent from the installed Habitat-Sim 0.3.3. `main.py` therefore calls
`enable_compatible_look_actions(cfg)`, selecting repository-local action classes
that support both the public `sensors` and current private `_sensors` APIs while
preserving the `look_up` and `look_down` action names.

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

## Stable Main Configuration

`scripts/main_config.py` contains only paths that describe the expected local
checkout layout: project root, adjacent Habitat-Lab root, ObjectNav config, and
scene-content directory. It sits beside `main.py` and is imported normally when
the script is run directly. Runtime choices such as enabled panels, label
display, overlay opacity, episode count, seed, and scene selection deliberately
remain visible in `main.py`.

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

The active visualization exposes the two optimized-map views as independent
dashboard panels:

```python
"3D voxel view": lambda: voxel_mapper.render_camera_view(...)
"Voxel top-down map": lambda: voxel_mapper.render_topdown_map(...)
```

`render_maps(...)` remains available as a convenience for experiments that want
the two images pre-combined side by side.

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
