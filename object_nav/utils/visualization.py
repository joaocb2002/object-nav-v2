from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import habitat


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to BGR for OpenCV display."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def print_episode(env: habitat.Env) -> None:
    """Print the current Habitat episode id, scene id, and object goal."""
    ep = env.current_episode
    goal = getattr(ep.goals[0], "object_category", "unknown")

    print("\nEpisode:", ep.episode_id)
    print("Scene:", ep.scene_id)
    print("Goal:", goal)
