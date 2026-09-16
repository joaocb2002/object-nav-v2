# Utilities

`object_nav.utils` contains small helpers shared by scripts. These modules should
stay lightweight and free of experiment policy.

## Printing And Image Helpers

`visualization.py` provides:

- `rgb_to_bgr(rgb)`: convert Habitat RGB arrays for OpenCV display.
- `blend_bgr_overlay(image_bgr, overlay_bgr, opacity=0.25)`: transparently
  combine any two display-ready images. Zero opacity preserves the base image;
  one uses only the overlay.
- `print_config(cfg)`: print the resolved Habitat/OmegaConf config.
- `print_env(env)`: print high-level Habitat environment state.
- `print_episode(ep, verbose=False)`: print the active episode id, scene, and
  goal category.
- `DashboardConfig`: ordered enabled panels plus tile/window settings.
- `OpenCVDashboard`: lazily evaluate enabled BGR image producers and show them
  in one automatically tiled OpenCV window.
- `compose_dashboard_bgr(...)`: pure image compositor used by the dashboard and
  tests.

Typical script usage:

```python
from object_nav.utils import print_config, print_env, print_episode

print_config(cfg)
with habitat.Env(config=cfg) as env:
    print_env(env)
    ...
    print_episode(env.current_episode)
```

Dashboard panels are selected and ordered in one place. Sources are callables,
so a disabled panel does not perform display-only computation:

```python
display = OpenCVDashboard(
    DashboardConfig(enabled_panels=("RGB", "3D voxel view"))
)
display.open()
display.show(
    {
        "RGB": lambda: rgb_to_bgr(obs["rgb"]),
        "3D voxel view": lambda: voxel_mapper.render_camera_view(
            env, output_height=480
        ),
        "Disabled experiment": lambda: render_expensive_debug_image(),
    }
)
```

Panel producers must return display-ready `uint8` BGR arrays with shape
`H x W x 3`, or `None` when no image is available. `DashboardConfig.fullscreen`
defaults to `True`; set it to `False` when a normal resizable window is desired.
Calling `open()` early is optional, but the active script uses it before model
and simulator setup so OpenCV's GUI backend initializes before other threaded
native libraries. `show(...)` still opens the dashboard automatically if needed.

`blend_bgr_overlay(...)` is deliberately perception-model agnostic. It resizes a
mismatched overlay with nearest-neighbor interpolation and validates both inputs
as `uint8` BGR images, making it reusable if SegFormer is replaced.

## Habitat Dataset Helpers

`datasets.py` provides small helpers for ObjectNav content shards:

- `list_objectnav_scene_ids(content_dir)`: list scene ids from `*.json.gz`
  files, stripping the full `.json.gz` suffix.
- `choose_random_objectnav_scene(content_dir, rng=None)`: choose one scene id
  from that list.

Typical `scripts/main.py` usage:

```python
import random

from object_nav.utils import choose_random_objectnav_scene

RUN_SEED = time.time_ns() % (2**32)
SCENE_CONTENT_DIR = "data/datasets/objectnav/hm3d/v2/train/content"
SCENE = choose_random_objectnav_scene(SCENE_CONTENT_DIR, rng=random.Random(RUN_SEED))

with read_write(cfg):
    cfg.habitat.seed = RUN_SEED
    cfg.habitat.dataset.content_scenes = [SCENE]
```

This keeps each run to one scene while still letting Habitat sample the episode
from that selected scene.

## Run Output Directories

`artifacts.py` provides `make_run_output_dir(...)` for debug artifacts such as
PLY point clouds or static previews:

```python
from object_nav.utils import make_run_output_dir

output_dir = make_run_output_dir(
    script_path=__file__,
    scene_id=SCENE,
    episode_id=str(env.current_episode.episode_id),
)
```

The directory is created under the repository `outputs/` folder by default. Its
name includes timestamp, scene id, episode id, and script name. It is safe to use
from `scripts/main.py` even after that script changes the current directory into
the local Habitat-Lab checkout, because the helper resolves the repository root
from the installed `object_nav` package path.

Only create an output directory when the run actually writes artifacts. Keeping
this helper out of the active loop avoids empty run folders.
