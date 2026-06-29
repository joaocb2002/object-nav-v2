# Perception Module

This package contains the YOLO-based perception utilities used in ObjectNav
experiments. Keep this module small: prefer Habitat and Ultralytics APIs over
custom wrappers when they already do the job.

## Files

- `config.py`: typed `YoloConfig`, default weights path, and YOLO predict kwargs.
- `detections.py`: immutable `Detection` and `DetectionResult` dataclasses.
- `yolo.py`: `YOLODetector`, detector construction, RGB/BGR input handling, and
  conversion from Ultralytics `Results` to `DetectionResult` objects.
- `observations.py`: experimental helpers for printing Habitat observations and
  showing depth + YOLO detections.
- `patches.py`: explicit Ultralytics monkey patch for softmax class probabilities.
- `ultralytics_compat.py`: small compatibility imports for Ultralytics internals.

## Basic Usage

```python
from object_nav.perception import (
    YoloConfig,
    build_yolo_detector,
    print_detections,
    print_observations,
show_depth_rgb_detections,
)

detector = build_yolo_detector(YoloConfig())
detections = detector.detect(obs["rgb"])
print_observations(obs)
print_detections(detections)
show_depth_rgb_detections(obs["rgb"], obs.get("depth"), detections)
```

`obs` is expected to contain `rgb`, may contain `depth`, and may contain other
Habitat observation values such as `objectgoal`, `compass`, and `gps`.

## Weights

The default weights path is:

```text
<repo>/models/yolo/yolo11x.pt
```

Weights are ignored by git. Download or copy `yolo11x.pt` there before running
the detector, or pass a custom `weights_path` in `YoloConfig`.

## Precision

Ultralytics deprecated the old `half` predict argument. Use `quantize` instead:

```python
YoloConfig(quantize=16)
YoloConfig(quantize="fp16")
```

Leave `quantize=None` for the default FP32 behavior.

## Plotting

Detection plotting uses Ultralytics `Results.plot()`. The only custom plotting
done here is the Habitat depth map, which is combined side by side with the
annotated YOLO image for experiments.
