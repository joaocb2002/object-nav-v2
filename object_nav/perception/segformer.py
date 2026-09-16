"""SegFormer loading and visualization for the active ObjectNav experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGFORMER_CHECKPOINT_PATH = (
    PROJECT_ROOT / "models/segformer/segformer_b5_mpcat40_calibrated"
)
_CHECKPOINT_FILES = (
    "model.safetensors",
    "config.json",
    "checkpoint.json",
    "camera_profile.yaml",
    "calibration.json",
)


@dataclass(frozen=True)
class SegFormerConfig:
    """Runtime settings for the deployed HM3D SegFormer model."""

    checkpoint_path: str | Path = DEFAULT_SEGFORMER_CHECKPOINT_PATH
    device: Optional[str] = None


def build_segformer_segmenter(
    config: SegFormerConfig = SegFormerConfig(),
) -> Any:
    """Load the calibrated segmenter once from its complete checkpoint bundle."""
    checkpoint = Path(config.checkpoint_path).resolve()
    missing = [name for name in _CHECKPOINT_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete SegFormer checkpoint at {checkpoint}. Missing: "
            + ", ".join(missing)
        )

    try:
        from hm3d_semseg.inference import SemanticSegmenter
    except ImportError as exc:
        raise ImportError(
            "The hm3d-semseg inference package is required. Install the adjacent "
            "package in the Habitat environment with: python3 -m pip install -e "
            "'../hm3d-semseg[inference]'"
        ) from exc

    return SemanticSegmenter.from_checkpoint(checkpoint, device=config.device)


def assert_segformer_camera(
    segmenter: Any,
    objectnav_config: str | Path,
    *,
    habitat_lab_root: Optional[str | Path] = None,
) -> None:
    """Require the runtime Habitat camera to match the checkpoint contract."""
    try:
        from hm3d_semseg.camera import resolve_camera_profile
    except ImportError as exc:
        raise ImportError("The hm3d-semseg camera utilities are required.") from exc

    root = Path(habitat_lab_root) if habitat_lab_root is not None else None
    camera = resolve_camera_profile(Path(objectnav_config), root)
    segmenter.assert_camera(camera)
