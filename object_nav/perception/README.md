# Perception Module

This package contains the active SegFormer integration and the retained YOLO
experiment. SegFormer is used by `scripts/main.py`; YOLO remains available for
reproducing the earlier detector/probability work. Keep this module small and
keep heavyweight imports behind the model-specific helpers.

## Files

- `segformer.py`: load the complete calibrated checkpoint and enforce its camera
  contract.
- `config.py`: typed `YoloConfig`, default weights path, and YOLO predict kwargs.
- `detections.py`: immutable `Detection` and `DetectionResult` dataclasses.
- `yolo.py`: `YOLODetector`, detector construction, RGB/BGR input handling, and
  conversion from Ultralytics `Results` to `DetectionResult` objects.
- `observations.py`: experimental helpers for printing Habitat observations and
  displaying RGB, depth, semantic labels, connected-component class names, or
  retained YOLO detections.
- `patches.py`: explicit Ultralytics monkey patch for softmax class probabilities.
- `ultralytics_compat.py`: small compatibility imports for Ultralytics internals.

## Active SegFormer Usage

The model package is maintained in the adjacent `hm3d-semseg` repository and
must be installed into the Habitat environment. The checkpoint stays under this
repository's ignored `models/` directory.

```python
from object_nav.perception import (
    SegFormerConfig,
    assert_segformer_camera,
    build_segformer_segmenter,
    show_segmentation,
)

segmenter = build_segformer_segmenter(SegFormerConfig())
assert_segformer_camera(segmenter, objectnav_config, habitat_lab_root=habitat_root)

result = segmenter(obs["rgb"])
show_segmentation(result["labels"])
```

Loading reads calibration and camera metadata from the checkpoint bundle. Do
not pass a second temperature or bypass a camera mismatch. See
`docs/segformer_b5_model_contract.md` for the full interface and evidence.

The optional class-name renderer is model-agnostic when given an explicit
`class_names` mapping:

```python
annotated = annotate_semantic_islands_bgr(
    labels,
    colorized_bgr,
    class_names={3: "chair"},
)
```

Without a mapping it lazily uses the installed `hm3d-semseg` `ID2LABEL`
taxonomy. For each class it considers only the largest connected
component and places a fixed-size, class-colored, black-outlined label at the
component's maximum-inscribed distance-transform point. Every visible class is
labeled, even when its largest component is one pixel; text may extend beyond
that component but is clamped inside the image. The function changes only a
copy of the visualization image, never the class-ID mask.

## Retained YOLO Usage

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

The custom softmax implementation is repository-local; the installed
Ultralytics source is unmodified. See `docs/yolo_contract.md` for the audited
version, precise runtime overrides, probability semantics, and reproduction
instructions. The optional dependency is pinned to Ultralytics 8.4.82 and the
patch refuses to run against another version. Verify both isolated behaviors
with:

```bash
python3 scripts/tools/check_yolo_contract.py --mode both --device cpu
```

## Precision

Ultralytics deprecated the old `half` predict argument. Use `quantize` instead:

```python
YoloConfig(quantize=16)
YoloConfig(quantize="fp16")
```

Leave `quantize=None` for the default FP32 behavior.

## Plotting

Detection plotting uses Ultralytics `Results.plot()`. SegFormer visualization is
a direct colorization of the predicted class-ID mask. The active script reuses
that color image for both an independent dashboard panel and a lightly blended
RGB panel. Blending itself is the model-independent `blend_bgr_overlay(...)`
utility under `object_nav.utils`, so it remains usable with another perception
model.
