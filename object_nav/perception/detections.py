from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Detection:
    """Single detection output from a perception model."""

    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]  # x1,y1,x2,y2
    scale: float  # detection_area / image_area
    probs: Optional[Tuple[float, ...]] = None  # class probability vector (optional)


@dataclass(frozen=True)
class DetectionResult:
    """Parsed detections plus the raw Ultralytics result."""

    detections: Tuple[Detection, ...]
    yolo_result: object
