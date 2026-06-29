from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .config import YoloConfig
from .detections import Detection, DetectionResult


def _missing_perception_dependency_error(package: str) -> ImportError:
    return ImportError(
        f"Missing optional perception dependency '{package}'. "
        "Install the perception extras with: python3 -m pip install -e '.[perception]'"
    )


def build_yolo_detector(config: YoloConfig) -> "YOLODetector":
    """Create and load a YOLO detector from configuration."""
    detector = YOLODetector(config)
    detector.load()
    return detector


class YOLODetector:
    def __init__(self, config: YoloConfig) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._resolved_device: Optional[str] = None

    def load(self) -> None:
        if self.config.use_softmax_patch:
            try:
                from .patches import apply_yolo_softmax_patch
            except ImportError as exc:
                raise _missing_perception_dependency_error(
                    "torch/torchvision/ultralytics"
                ) from exc

            apply_yolo_softmax_patch(temperature=self.config.softmax_temperature)

        weights_path = Path(self.config.weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at '{weights_path}'. "
                "Download yolo11x.pt and place it there, or pass a custom weights_path."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise _missing_perception_dependency_error("ultralytics") from exc

        self._model = YOLO(self.config.weights_path_str())
        self._resolved_device = self._resolve_device()

    def detect(self, image: np.ndarray, *, input_color: str = "rgb") -> DetectionResult:
        """Run YOLO on one image and return parsed detections."""
        image = _ensure_3_channel(image)
        image_bgr = _ensure_bgr(image, input_color=input_color)
        yolo_result = self._predict(image_bgr)
        detections = self.parse_detections(yolo_result, image)
        return DetectionResult(tuple(detections), yolo_result)

    def _predict(self, image_bgr: np.ndarray) -> object:
        """Run raw Ultralytics inference for one BGR image."""
        if self._model is None:
            raise RuntimeError("YOLODetector not loaded. Call .load() first.")

        if self._resolved_device is None:
            self._resolved_device = self._resolve_device()

        results = self._model.predict(
            source=image_bgr,
            device=self._resolved_device,
            **self.config.predict_kwargs(),
        )

        return results[0]

    def _resolve_device(self) -> str:
        """Resolve execution device from config with robust CUDA fallback behavior."""
        try:
            import torch
        except ImportError as exc:
            raise _missing_perception_dependency_error("torch") from exc

        requested = self.config.device.strip().lower()

        # Auto-select best available backend.
        if requested == "auto":
            if not torch.cuda.is_available():
                return "cpu"
            for idx in range(torch.cuda.device_count()):
                candidate = f"cuda:{idx}"
                if self._is_cuda_device_usable(candidate):
                    return candidate
            warnings.warn("No usable CUDA device found; falling back to CPU for YOLO.")
            return "cpu"

        # If user asked for CPU, respect it.
        if requested.startswith("cpu"):
            return "cpu"

        # Normalize bare 'cuda' to index 0.
        if requested == "cuda":
            requested = "cuda:0"

        # If CUDA isn't available at all, fall back.
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available; falling back to CPU for YOLO.")
            return "cpu"

        if requested.startswith("cuda:"):
            try:
                requested_index = int(requested.split(":", 1)[1])
            except ValueError:
                warnings.warn(
                    f"Invalid CUDA device specifier '{self.config.device}'; "
                    "falling back to CPU."
                )
                return "cpu"

            if requested_index < 0:
                warnings.warn(f"Invalid CUDA device index '{requested_index}'; falling back to CPU.")
                return "cpu"

            if requested_index >= torch.cuda.device_count():
                warnings.warn(
                    "Requested CUDA device index is out of range "
                    f"(requested={requested_index}, available={torch.cuda.device_count()}); "
                    "falling back to CPU."
                )
                return "cpu"

            if self._is_cuda_device_usable(requested):
                return requested

            warnings.warn(
                f"Requested CUDA device '{requested}' is not usable with this PyTorch build; "
                "falling back to CPU."
            )
            return "cpu"

        warnings.warn(f"Unknown device '{self.config.device}'; falling back to CPU.")
        return "cpu"

    @staticmethod
    def _is_cuda_device_usable(device: str) -> bool:
        """Return True if a lightweight tensor allocation succeeds on the CUDA device."""
        import torch

        try:
            _ = torch.zeros(1, device=device)
            return True
        except Exception:
            return False

    def parse_detections(self, results: object, image: np.ndarray) -> List[Detection]:
        """Parse a YOLO Results object into structured detections."""
        boxes = results.boxes  # patched Boxes class should be active
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1)
        cls = boxes.cls.cpu().numpy().astype(int).reshape(-1)
        prob_vectors = (
            boxes.probs.cpu().numpy()
            if self.config.use_softmax_patch and boxes.probs is not None
            else None
        )

        names = results.names if results.names is not None else {}
        image_area = float(image.shape[0] * image.shape[1])

        dets: List[Detection] = []
        for i in range(len(cls)):
            x1, y1, x2, y2 = (
                float(xyxy[i, 0]),
                float(xyxy[i, 1]),
                float(xyxy[i, 2]),
                float(xyxy[i, 3]),
            )
            det_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            scale = det_area / image_area if image_area > 0.0 else 0.0
            det_probs: Optional[Tuple[float, ...]]
            if prob_vectors is not None:
                det_probs = tuple(float(p) for p in prob_vectors[i].tolist())
            else:
                det_probs = None
            cls_id = int(cls[i])
            cls_name = str(names.get(cls_id, cls_id))
            dets.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=float(conf[i]),
                    xyxy=(x1, y1, x2, y2),
                    scale=scale,
                    probs=det_probs,
                )
            )
        return dets


def _ensure_3_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("image must be an HxWxC array.")
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        return np.ascontiguousarray(image[:, :, :3])
    raise ValueError("image must have 3 channels (RGB/BGR) or 4 channels (RGBA/BGRA).")


def _ensure_bgr(image: np.ndarray, *, input_color: str) -> np.ndarray:
    if input_color not in {"rgb", "bgr"}:
        raise ValueError("input_color must be 'rgb' or 'bgr'.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 array.")
    if input_color == "bgr":
        return image
    return np.ascontiguousarray(image[:, :, ::-1])
