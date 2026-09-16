# Agent Notes

This repo contains ObjectNav experiments built around Habitat/Habitat-Lab.
Keep changes small, explicit, and easy to read.

## Project Shape

- `object_nav/` is the importable project package.
- `scripts/` contains runnable experiment scripts.
- `scripts/main.py` is the active experiment; earlier focused probes live under
  `scripts/experiments/`, and reproducibility checks live under `scripts/tools/`.
- The local package should be installed in editable mode so scripts can import
  `object_nav` when run directly:

  ```bash
  python3 -m pip install --no-build-isolation -e .
  ```

- Prefer running scripts directly, for example:

  ```bash
  python3 scripts/main.py
  ```

## Import Style

- Use normal package imports from `object_nav`, for example:

  ```python
  from object_nav.utils import print_episode, rgb_to_bgr
  ```

- Do not add `sys.path` manipulation to scripts unless there is no cleaner
  option.
- Python imports must use `object_nav`, not `object-nav`.

## Coding Style

- Keep functions small and readable.
- Add explicit argument and return types for new functions.
- Add concise docstrings for public helpers.
- Prefer simple code over clever abstractions.
- Only introduce abstractions once there is a clear reuse point.
- Avoid broad refactors while making narrow script or utility changes.

## Habitat Scripts

- Preserve behavior from the original scripts unless the task asks otherwise.
- Keep `scripts/main.py` as short as practical while leaving the experiment
  sequence and important runtime choices visible. Move reusable mechanics into
  focused package helpers, but do not hide the loop behind a framework.
- Shared display/printing helpers belong in `object_nav/utils/visualization.py`.
- Keep experiment-specific constants and control flow in the script for now.
- When using Habitat types only for annotations, prefer type-only imports:

  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      import habitat
  ```

## Perception

- Perception code lives under `object_nav/perception/`.
- SegFormer is the active model in `scripts/main.py`; YOLO is retained for
  reproducibility but is not part of the active loop.
- Read `docs/segformer_b5_model_contract.md` and `docs/yolo_contract.md` before
  changing either model integration.
- The SegFormer Python API comes from the adjacent `hm3d-semseg` checkout,
  installed editable in the same Habitat environment. Do not copy that package
  into this repository or add `sys.path` manipulation.
- Treat all five files in the calibrated SegFormer checkpoint directory as one
  deployment unit, load it once, and enforce its camera contract at startup.
- YOLO weights belong under the repo-root `models/yolo/` directory and should
  not be committed.
- The default YOLO weights path is `<repo>/models/yolo/yolo11x.pt`.
- Keep heavyweight dependencies such as Ultralytics/Torch out of lightweight
  imports where practical.
- The YOLO softmax behavior is implemented as an explicit patch in
  `object_nav/perception/patches.py`.
- Keep `object_nav/perception/` simple: model-specific modules own loading and
  inference integration; `observations.py` owns experimental per-frame
  display/printing helpers.
- Ultralytics predict precision should use `quantize`, not deprecated `half`.
- Do not edit the shared Ultralytics installation. The historical softmax
  behavior is the explicit repository-local runtime patch in `patches.py` and
  is tied to Ultralytics 8.4.82.

## Verification

- At minimum, run syntax checks after code edits:

  ```bash
  python3 -m py_compile path/to/file.py
  ```

- For package import changes, also verify imports from outside the repo:

  ```bash
  cd /tmp
  python3 -c "from object_nav.utils import print_episode, rgb_to_bgr"
  ```

## Documentation Sync

- Every important behavior, API, model-contract, dependency, configuration, or
  repository-layout change must update the closest relevant Markdown file in
  the same change.
- Prefer the narrowest useful document: model details in `docs/`, package APIs
  in the package README, runnable workflow changes in `scripts/README.md`, and
  project-wide setup or intent in the root `README.md`.
- Keep documentation concise and factual, but include enough paths, commands,
  inputs, outputs, and assumptions for a developer or coding agent to reproduce
  the workflow on another machine.
- Do not document planned functionality as though it already exists.

## Git Hygiene

- The working tree may contain unrelated user changes.
- Do not revert, delete, or reformat unrelated files.
- Keep generated/cache files out of commits unless explicitly requested.
