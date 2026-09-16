"""Smoke-check stock and repository-patched YOLO behavior in fresh processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = PROJECT_ROOT / "screenshots/2026_7_27_16-7-58/0.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("both", "upstream", "patched"),
        default="both",
        help="Behavior to verify. 'both' launches two isolated child processes.",
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run_isolated_modes(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    for mode in ("upstream", "patched"):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--mode",
                mode,
                "--image",
                str(args.image),
                "--device",
                args.device,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


def check_mode(mode: str, image_path: Path, device: str) -> None:
    import ultralytics

    from object_nav.perception import YoloConfig, build_yolo_detector
    from object_nav.perception.patches import AUDITED_ULTRALYTICS_VERSION

    if ultralytics.__version__ != AUDITED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected Ultralytics {AUDITED_ULTRALYTICS_VERSION}, "
            f"found {ultralytics.__version__}"
        )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read YOLO smoke image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    patched = mode == "patched"
    detector = build_yolo_detector(
        YoloConfig(device=device, use_softmax_patch=patched)
    )
    result = detector.detect(image_rgb)
    boxes = result.yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        raise AssertionError("YOLO smoke image produced no detections")

    columns = int(boxes.data.shape[1])
    if patched:
        class_count = len(result.yolo_result.names)
        if columns != 6 + class_count:
            raise AssertionError(
                f"Patched boxes have {columns} columns; expected {6 + class_count}"
            )
        probabilities = [detection.probs for detection in result.detections]
        if any(vector is None or len(vector) != class_count for vector in probabilities):
            raise AssertionError("Patched detections do not retain full class vectors")
        sums = np.asarray([sum(vector) for vector in probabilities if vector is not None])
        if not np.allclose(sums, 1.0, atol=1e-5):
            raise AssertionError("Patched class vectors do not sum to one")
    else:
        if columns != 6:
            raise AssertionError(f"Upstream boxes have {columns} columns; expected 6")
        if any(detection.probs is not None for detection in result.detections):
            raise AssertionError("Upstream detections unexpectedly expose class vectors")

    print(
        f"{mode}: ultralytics={ultralytics.__version__} "
        f"detections={len(result.detections)} columns={columns} device={device}"
    )


def main() -> None:
    args = parse_args()
    if args.mode == "both":
        run_isolated_modes(args)
        return
    check_mode(args.mode, args.image.resolve(), args.device)


if __name__ == "__main__":
    main()
