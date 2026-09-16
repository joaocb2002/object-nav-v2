from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Union

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import habitat


PanelProducer = Callable[[], Optional[np.ndarray]]
PanelSource = Union[np.ndarray, PanelProducer, None]


@dataclass(frozen=True)
class DashboardConfig:
    """Layout and enabled-panel settings for an OpenCV dashboard."""

    enabled_panels: tuple[str, ...]
    window_name: str = "ObjectNav dashboard"
    columns: Optional[int] = None
    tile_width: int = 480
    tile_height: int = 360
    fullscreen: bool = True


class OpenCVDashboard:
    """Render enabled, independently produced images in one tiled window."""

    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self._window_open = False

    def open(self) -> None:
        """Create and initialize the dashboard before expensive runtime setup."""
        if self._window_open:
            return
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        if self.config.fullscreen:
            cv2.setWindowProperty(
                self.config.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        loading = np.full(
            (self.config.tile_height, self.config.tile_width, 3),
            24,
            dtype=np.uint8,
        )
        cv2.putText(
            loading,
            "Initializing ObjectNav...",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self.config.window_name, loading)
        cv2.waitKey(1)
        self._window_open = True

    def show(self, sources: Mapping[str, PanelSource]) -> None:
        """Evaluate enabled panel sources and show their tiled BGR images."""
        panels = {}
        for name in self.config.enabled_panels:
            source = sources.get(name)
            image = source() if callable(source) else source
            if image is not None:
                panels[name] = image

        if not panels:
            return

        self.open()
        cv2.imshow(
            self.config.window_name,
            compose_dashboard_bgr(panels, config=self.config),
        )

    def close(self) -> None:
        """Close the dashboard window if it was opened."""
        if not self._window_open:
            return
        try:
            cv2.destroyWindow(self.config.window_name)
        except cv2.error:
            pass
        self._window_open = False


def compose_dashboard_bgr(
    panels: Mapping[str, np.ndarray],
    *,
    config: DashboardConfig,
) -> np.ndarray:
    """Compose enabled, display-ready BGR panels into one labeled grid."""
    if config.tile_width <= 0 or config.tile_height <= 0:
        raise ValueError("Dashboard tile dimensions must be positive.")

    active = [
        (name, panels[name])
        for name in config.enabled_panels
        if name in panels
    ]
    if not active:
        raise ValueError("No enabled dashboard panels were provided.")

    columns = (
        config.columns
        if config.columns is not None
        else int(ceil(sqrt(len(active))))
    )
    if columns <= 0:
        raise ValueError("Dashboard columns must be positive.")
    rows = int(ceil(len(active) / columns))
    canvas = np.full(
        (rows * config.tile_height, columns * config.tile_width, 3),
        24,
        dtype=np.uint8,
    )

    for index, (name, image) in enumerate(active):
        row, column = divmod(index, columns)
        tile = _fit_dashboard_panel(image, config.tile_width, config.tile_height)
        cv2.rectangle(tile, (0, 0), (config.tile_width - 1, 30), (0, 0, 0), -1)
        cv2.putText(
            tile,
            name,
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        top = row * config.tile_height
        left = column * config.tile_width
        canvas[top : top + config.tile_height, left : left + config.tile_width] = tile

    return canvas


def _fit_dashboard_panel(
    image_bgr: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Dashboard panels must be uint8 HxWx3 BGR images.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Dashboard panels cannot be empty.")

    scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    tile = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
    top = (tile_height - height) // 2
    left = (tile_width - width) // 2
    tile[top : top + height, left : left + width] = resized
    return tile


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to BGR for OpenCV display."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def blend_bgr_overlay(
    image_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    *,
    opacity: float = 0.25,
) -> np.ndarray:
    """Blend a color overlay onto a BGR image with configurable opacity."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0.0 and 1.0.")

    image = _require_bgr_image(image_bgr, name="image_bgr")
    overlay = _require_bgr_image(overlay_bgr, name="overlay_bgr")
    if overlay.shape[:2] != image.shape[:2]:
        overlay = cv2.resize(
            overlay,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return cv2.addWeighted(image, 1.0 - opacity, overlay, opacity, 0.0)


def _require_bgr_image(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must be a uint8 HxWx3 BGR image.")
    return np.ascontiguousarray(array)


def print_config(config: DictConfig) -> None:
    """Print a resolved Habitat/OmegaConf config as YAML."""
    print(OmegaConf.to_yaml(config, resolve=True))


def print_env(env: habitat.Env) -> None:
    """Print the main Habitat environment attributes."""
    print("Number of episodes in iterator:", len(env.episodes))

    for field in (
        # "current_episode",
        "episode_iterator",
        "episode_over",
        "episode_start_time",
        # "episodes",
        # "sim",
        "task",
    ):
        print(f"{field}:", getattr(env, field))


def print_episode(ep: habitat.Episode, verbose: bool = False) -> None:
    """Print the current Habitat episode, with optional extra metadata."""

    print("Episode class type:", type(ep))
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
