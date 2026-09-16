"""Stable local paths used by the active ObjectNav entrypoint."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HABITAT_LAB_ROOT = PROJECT_ROOT.parent / "habitat-lab"
OBJECTNAV_CONFIG = (
    HABITAT_LAB_ROOT
    / "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
)
SCENE_CONTENT_DIR = HABITAT_LAB_ROOT / "data/datasets/objectnav/hm3d/v2/train/content"
