"""Dataset helpers for Habitat ObjectNav scripts."""

from __future__ import annotations

import random
from pathlib import Path


OBJECTNAV_CONTENT_SUFFIX = ".json.gz"


def list_objectnav_scene_ids(
    content_dir: str | Path,
    suffix: str = OBJECTNAV_CONTENT_SUFFIX,
) -> list[str]:
    """Return sorted scene ids from Habitat ObjectNav per-scene content files."""
    path = Path(content_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"ObjectNav content directory does not exist: {path}")

    scene_ids = sorted(
        file.name[: -len(suffix)]
        for file in path.iterdir()
        if file.is_file() and file.name.endswith(suffix)
    )
    if not scene_ids:
        raise FileNotFoundError(
            f"No ObjectNav scene files ending with {suffix!r} found in {path}"
        )

    return scene_ids


def choose_random_objectnav_scene(
    content_dir: str | Path,
    *,
    rng: random.Random | None = None,
) -> str:
    """Choose one random Habitat ObjectNav scene id from a content directory."""
    chooser = rng or random.SystemRandom()
    return chooser.choice(list_objectnav_scene_ids(content_dir))
