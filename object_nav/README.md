# ObjectNav Package

`object_nav` is the importable package for the Habitat ObjectNav experiments.
Runnable experiment scripts live in `scripts/`; reusable behavior lives here.

The package is split by responsibility:

- `agents/`: action policies used by scripts, such as keyboard and random agents.
- `mapping/`: sparse geometry mapping, Habitat mapping adapters, map rendering,
  and optional point-cloud debugging.
- `perception/`: YOLO loading, detection data, and per-frame perception display.
- `utils/`: small script helpers for printing, image conversion, and run outputs.

Keep entrypoints simple. `scripts/main.py` should describe the experiment flow:
load Habitat config, create the agent/perception/mapping objects, run episodes,
render what is useful, and step the environment. Package modules should own the
details that would make that loop hard to read.

## Import Style

Use package imports from the repository root installation:

```python
from object_nav.agents import InteractiveKeyboardAgent
from object_nav.mapping.habitat import HabitatVoxelMapper
from object_nav.utils import print_episode
```

Avoid adding `sys.path` manipulation to scripts. Install the package in editable
mode when running scripts directly:

```bash
python3 -m pip install --no-build-isolation -e .
```

## Extension Guide

Add new code where the runtime dependency belongs:

- Habitat-independent math or storage goes in a core module.
- Habitat sensor/config/pose adapters go beside other Habitat adapters.
- Visualization helpers go in visualization modules.
- Optional debug tools should be importable directly from their module, not
  required by lightweight package imports.

If a new object needs to be used from `main.py`, expose one small public class or
function and document the exact call point in the package README.
