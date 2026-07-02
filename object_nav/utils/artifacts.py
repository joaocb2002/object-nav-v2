from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_run_output_dir(
    *,
    script_path: str | Path,
    scene_id: str,
    episode_id: str,
    root: Optional[str | Path] = None,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Create a uniquely named output directory for one script episode run."""
    output_root = Path(root) if root is not None else PROJECT_ROOT / "outputs"
    run_time = timestamp or datetime.now()
    directory_name = "__".join(
        (
            run_time.strftime("%Y-%m-%d_%H-%M-%S"),
            f"scene-{_safe_path_part(scene_id)}",
            f"episode-{_safe_path_part(episode_id)}",
            f"script-{_safe_path_part(Path(script_path).stem)}",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / directory_name
    suffix = 2
    while output_dir.exists():
        output_dir = output_root / f"{directory_name}_{suffix:02d}"
        suffix += 1

    output_dir.mkdir()
    return output_dir


def _safe_path_part(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in str(value).strip()
    )
    return cleaned.strip("-") or "unknown"
