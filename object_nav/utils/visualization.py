from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import habitat


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to BGR for OpenCV display."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def print_config(config: DictConfig) -> None:
    """Print a resolved Habitat/OmegaConf config as YAML."""
    print(OmegaConf.to_yaml(config, resolve=True))


def print_episode(env: habitat.Env, verbose: bool = False) -> None:
    """Print the current Habitat episode, with optional extra metadata."""
    ep = env.current_episode
    goal = getattr(ep.goals[0], "object_category", "unknown")
    fields = ["episode_id", "scene_id", "goal"]

    if verbose:
        fields.extend(
            [
                "scene_dataset_config",
                "additional_obj_config_paths",
                "start_position",
                "start_rotation",
                "info",
            ]
        )

    for field in fields:
        value = goal if field == "goal" else getattr(ep, field)
        print(f"{field}:", value)
