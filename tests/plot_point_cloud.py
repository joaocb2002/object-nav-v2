from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from object_nav.mapping.point_cloud import load_colored_ply, render_point_cloud_summary_bgr

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    ply_path = args.ply_path or find_latest_point_cloud()
    if ply_path is None:
        raise SystemExit("No .ply path given and no point_cloud.ply found under outputs/.")

    points, colors = load_colored_ply(ply_path)
    print_point_cloud_summary(ply_path, points, colors)

    if args.preview:
        save_preview(args.preview, points, colors)

    if args.viewer == "open3d":
        opened = plot_with_open3d(ply_path, args.window_name)
    else:
        opened = plot_with_external_viewer(ply_path, args.viewer)

    if not opened:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually test interactive plotting of an existing PLY point cloud.",
    )
    parser.add_argument(
        "ply_path",
        nargs="?",
        type=Path,
        help="Path to an existing .ply file. Defaults to the newest outputs/**/point_cloud.ply.",
    )
    parser.add_argument(
        "--viewer",
        choices=("open3d", "cloudcompare", "meshlab"),
        default="open3d",
        help="Interactive viewer backend to try.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional path for a static top/front/side PNG preview.",
    )
    parser.add_argument(
        "--window-name",
        default="Point cloud debug",
        help="Window title for the Open3D viewer.",
    )
    return parser.parse_args()


def find_latest_point_cloud() -> Path | None:
    candidates = sorted(
        (REPO_ROOT / "outputs").glob("**/point_cloud.ply"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def print_point_cloud_summary(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    print("PLY:", path)
    print("Points:", len(points))
    print("Has RGB colors:", len(colors) == len(points) and len(colors) > 0)
    if len(points) == 0:
        return

    print("XYZ min:", np.round(points.min(axis=0), 4))
    print("XYZ max:", np.round(points.max(axis=0), 4))
    print("XYZ mean:", np.round(points.mean(axis=0), 4))
    if len(colors) > 0:
        print("RGB min:", colors.min(axis=0))
        print("RGB max:", colors.max(axis=0))
        print("RGB mean:", np.round(colors.mean(axis=0), 2))


def save_preview(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = render_point_cloud_summary_bgr(points, colors)
    cv2.imwrite(str(path), image)
    print("Static preview:", path)


def plot_with_open3d(path: Path, window_name: str) -> bool:
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is not installed. Try `python3 -m pip install open3d`.")
        return False

    point_cloud = o3d.io.read_point_cloud(str(path))
    if point_cloud.is_empty():
        print("Open3D loaded an empty point cloud.")
        return False

    viewer = o3d.visualization.Visualizer()
    created = viewer.create_window(window_name=window_name, width=1280, height=800)
    if not created:
        print(
            "Open3D could not create an OpenGL window. This usually means the "
            "current session is headless, missing GLX/EGL support, or running "
            "through a display path that cannot provide an OpenGL context."
        )
        print("Try CloudCompare/MeshLab locally, or run this from a desktop session.")
        return False

    viewer.add_geometry(point_cloud)
    render_options = viewer.get_render_option()
    render_options.point_size = 2.0
    render_options.background_color = np.array([0.05, 0.05, 0.05])
    viewer.run()
    viewer.destroy_window()
    return True


def plot_with_external_viewer(path: Path, viewer: str) -> bool:
    commands = external_viewer_commands(path, viewer)
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        print("Opening with:", " ".join(command))
        subprocess.Popen(command)
        return True

    print(f"{viewer} was not found on PATH.")
    return False


def external_viewer_commands(path: Path, viewer: str) -> Sequence[tuple[str, ...]]:
    if viewer == "cloudcompare":
        return (
            ("cloudcompare", "-O", str(path)),
            ("CloudCompare", "-O", str(path)),
        )
    if viewer == "meshlab":
        return (("meshlab", str(path)),)
    raise ValueError(f"Unknown viewer: {viewer}")


if __name__ == "__main__":
    main()
