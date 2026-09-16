# Earlier experiments

These numbered scripts are the focused configuration, simulator, PointNav, and
ObjectNav probes that preceded `scripts/main.py`. They remain deliberately
simple and preserve their original control flow.

Every path is resolved from the script location, so a script can be launched
from any working directory. Scripts that use Habitat datasets still change into
the adjacent Habitat-Lab checkout because paths inside the composed Habitat
configuration are relative to that checkout.

- `01_inspect_config.py`: print the fully composed ObjectNav configuration.
- `02_load_env.py`: reset ObjectNav and dump environment internals.
- `03_load_backend.py`: construct Habitat-Sim directly with a semantic camera.
- `04_pointnav_interactive_play.py`: keyboard-controlled PointNav example.
- `05_objectnav_random_rollout.py`: three simple random ObjectNav rollouts.
- `06_objectnav_interactive_play.py`: keyboard-controlled fixed-scene ObjectNav.

Run them from the `habitat` Conda environment, for example:

```bash
python3 scripts/experiments/03_load_backend.py
```

The interactive scripts require an OpenCV-capable graphical session.
