# ObjectNav/V2

This is an iterative research repository for robot-navigation world
representations. Habitat ObjectNav is the current experiment and evaluation
surface, while the representation work is intended to remain useful for related
tasks such as PointNav, ImageNav, and EQA.

The repository is deliberately script-led. [`scripts/main.py`](scripts/main.py)
is the active research/debugging entry point: changes should stay small enough
that its per-step flow remains easy to inspect. Earlier numbered scripts live in
[`scripts/experiments/`](scripts/experiments/) and remain useful focused
references, not a second framework.

## Current pipeline

At each interactive ObjectNav step, `main.py` currently:

1. integrates Habitat depth into a sparse, block-allocated 3D occupancy map;
2. runs the calibrated 41-class SegFormer-B5 model on the RGB observation;
3. displays the enabled RGB, depth, SegFormer, voxel-world, and Habitat-map
   panels in one automatically tiled OpenCV dashboard;
4. accepts a keyboard action and eventually prints Habitat's ObjectNav metrics.

SegFormer is the active perception direction. The older YOLO detector remains in
`object_nav/perception/` so its experimental softmax behavior can be reproduced,
but it is no longer called by `main.py`.

Model behavior is defined by the contracts in [`docs/`](docs/):

- [`segformer_b5_model_contract.md`](docs/segformer_b5_model_contract.md)
- [`yolo_contract.md`](docs/yolo_contract.md)

## Repository layout

- `object_nav/mapping/`: Habitat-independent sparse occupancy plus Habitat
  adapters, comparison tools, point-cloud debugging, and rendering.
- `object_nav/perception/`: active SegFormer loading/display helpers and retained
  YOLO detection code.
- `object_nav/agents/`: keyboard and random action policies.
- `object_nav/utils/`: lightweight dataset, output, and display helpers.
- `scripts/main.py`: active iterative experiment.
- `scripts/experiments/`: older focused probes and examples with robust paths.
- `scripts/tools/`: explicit environment and model-contract checks.
- `tests/`: lightweight unit tests plus optional local-model integration checks.
- `models/`: ignored local model artifacts; only `.gitkeep` files are tracked.

## Local environment assumptions

The expected layout is:

```text
projects/
├── object-nav-v2/
├── habitat-lab/
└── hm3d-semseg/
```

The active environment is the Conda environment named `habitat`. Habitat-Lab is
an editable adjacent checkout, and its datasets are addressed relative to that
checkout. `main.py` resolves both repositories from its own location, then
changes into the Habitat-Lab root because composed Habitat dataset paths remain
relative to it.

Install the two local packages into the same environment:

```bash
conda activate habitat
python3 -m pip install --no-build-isolation -e .
python3 -m pip install -e "../hm3d-semseg[inference]"
```

The deployed SegFormer bundle must exist at:

```text
models/segformer/segformer_b5_mpcat40_calibrated/
```

The five files documented in the SegFormer contract are one unit. The runtime
checks that the bundle is complete and that its frozen camera geometry matches
the composed ObjectNav config before starting an episode.

The local `hm3d-semseg` install is currently editable rather than pinned by this
project's `pyproject.toml`. That is convenient for the present paired research
checkouts, but a portable experiment should record and install a verified
`hm3d-semseg` Git revision as well as the checkpoint hashes.

## Run and verify

From the ObjectNav/V2 repository root:

```bash
conda activate habitat
python3 scripts/main.py
```

Controls are `W` forward, `A` left, `D` right, up/down arrows to look up/down,
and `F` to stop. The ObjectNav config supplies both look actions with a 30°
tilt. The script requires local HM3D ObjectNav data and a graphical session for
its OpenCV dashboard.

Run the full regression suite with:

```bash
python3 -m unittest discover -v
```

The YOLO contract tests run when Ultralytics and
`models/yolo/yolo11x.pt` are available; otherwise they are skipped. To inspect
the stock and repository-patched behavior directly, run:

```bash
python3 scripts/tools/check_yolo_contract.py --mode both --device cpu
```

Before broad changes, read `AGENTS.md`, the model contracts, and the package
READMEs. In particular, keep semantic beliefs separate from the existing
geometry voxel storage until an explicit semantic-map experiment defines their
fusion and uncertainty behavior.
