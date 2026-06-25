# Agent Notes

This repo contains ObjectNav experiments built around Habitat/Habitat-Lab.
Keep changes small, explicit, and easy to read.

## Project Shape

- `object_nav/` is the importable project package.
- `scripts/` contains runnable experiment scripts.
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
- Shared display/printing helpers belong in `object_nav/utils/visualization.py`.
- Keep experiment-specific constants and control flow in the script for now.
- When using Habitat types only for annotations, prefer type-only imports:

  ```python
  from __future__ import annotations
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      import habitat
  ```

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

## Git Hygiene

- The working tree may contain unrelated user changes.
- Do not revert, delete, or reformat unrelated files.
- Keep generated/cache files out of commits unless explicitly requested.

