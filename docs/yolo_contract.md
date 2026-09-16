# Retained YOLO11x softmax experiment

This document records the exact YOLO behavior retained by ObjectNav/V2 after
SegFormer became the active perception model. Read it before reusing or changing
`object_nav/perception/patches.py`.

## Audited installation

The installation was audited on 2026-09-16 in the Conda environment `habitat`.

| Field | Audited value |
|---|---|
| Python | 3.9.25 |
| Ultralytics | 8.4.82 |
| Installed source | `/home/joaocb2002/miniconda3/envs/habitat/lib/python3.9/site-packages/ultralytics` |
| Official wheel | `ultralytics-8.4.82-py3-none-any.whl` |
| Official wheel SHA-256 | `2a42c20173af2f120df0e378d59b6d9f4d517f37c1fdade02f9b650d67fe721b` |
| Torch | 2.5.1+cu118 |
| Torchvision | 0.20.1+cu118 |

A recursive comparison of the installed `ultralytics/` package against the
official 8.4.82 wheel found no changed, added, or removed source files. The
shared Conda installation is therefore not modified. The nonstandard behavior
is a runtime monkey patch stored in this repository:

```text
object_nav/perception/patches.py
```

Importing Ultralytics alone gives upstream behavior. The patch is applied only
when `YOLODetector.load()` is called with `use_softmax_patch=True` (the default).
It changes Ultralytics classes and functions globally inside that Python process,
but it does not write to `site-packages`.

## Model identity

| Field | Value |
|---|---|
| Weights | `models/yolo/yolo11x.pt` |
| Weights SHA-256 | `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` |
| Architecture | YOLO11x detection, `yolo11x.yaml` scale `x` |
| Classes | 80 COCO classes |
| Checkpoint producer version | Ultralytics 8.2.100 |
| Checkpoint date | 2024-09-25 |
| Training data recorded in checkpoint | `coco.yaml` |

The file matches the expected structure and metadata of the published YOLO11x
COCO detector. This audit did not find an authoritative published checksum with
which to prove that the weights are byte-identical to an upstream download, so
the local SHA-256 above is the reproducible identity.

## Project configuration

`YoloConfig` currently defaults to:

| Setting | Value |
|---|---:|
| device | `cuda:0` with validated CPU fallback |
| confidence threshold | `0.25` |
| NMS IoU threshold | `0.7` |
| requested image size | `(640, 480)` interpreted as height, width |
| rectangular inference | enabled |
| maximum detections | 30 |
| precision | FP32 (`quantize=None`) |
| softmax patch | enabled |
| softmax temperature | `2.4` |

For a 4:3 Habitat frame, the requested portrait-shaped size and rectangular
letterboxing produce a verified network tensor of shape `[1, 3, 384, 480]` after
stride alignment. This is historical behavior, not a recommendation to use that
shape for a future detector experiment.

## Input and upstream preprocessing

`YOLODetector.detect(...)` accepts an `H x W x 3` or `H x W x 4` NumPy image.
The default `input_color="rgb"` matches Habitat observations. A fourth channel is
dropped and RGB is converted to BGR before calling `YOLO.predict`, because
Ultralytics accepts OpenCV-style NumPy input. Upstream prediction then:

1. applies rectangular letterboxing to the configured size and model stride;
2. converts BGR to RGB and HWC to BCHW;
3. makes the array contiguous;
4. converts it to FP32 by default and scales bytes from `0..255` to `0..1`.

Ultralytics scales retained boxes back to the original image coordinates during
postprocessing.

## Exact runtime modifications

The patch replaces four upstream behaviors.

### 1. Detection head output

Upstream `Detect._inference` concatenates decoded boxes with
`sigmoid(class_logits)`. The patch concatenates decoded boxes with the raw class
logits instead. This is necessary so temperature scaling is applied to logits,
not to already squashed scores.

Conceptually:

```diff
- concatenate(decoded_boxes, sigmoid(class_logits))
+ concatenate(decoded_boxes, class_logits)
```

### 2. Candidate filtering and NMS

The patch replaces the prediction-time NMS function. For each candidate it
computes:

```text
p(class | candidate) = softmax(class_logits / 2.4)
confidence = max_class p(class | candidate)
class_id = argmax_class p(class | candidate)
```

Both confidence-threshold filtering and NMS ranking use that maximum softmax
probability. Class-aware coordinate offsets and torchvision NMS are retained.
The complete 80-value vector is carried through NMS instead of discarding all
but the winning class.

This differs from an older local experiment found under
`Desktop/Old Code Repos/ObjectNav-Msc/`: that version ranked boxes by the maximum
sigmoid score and attached a separate temperature-softmax vector. The current
ObjectNav/V2 implementation uses the maximum softmax probability for both
filtering/ranking and the reported confidence.

### 3. Box result representation

Upstream `Boxes` accepts six values per detection (or seven with a tracking ID).
The patched class additionally accepts the extended detection row:

```text
[x1, y1, x2, y2, max_probability, class_id, p0, p1, ..., p79]
```

For YOLO11x this is 86 values. `boxes.conf` is derived from the maximum of the
stored probability vector, `boxes.cls` from its argmax, and the new
`boxes.probs` property exposes the entire vector. `YOLODetector` converts each
row into the repository's immutable `Detection`, whose `probs` tuple has length
80. With the patch disabled, standard six-column boxes are used and `probs` is
`None`. The patched `conf` and `cls` accessors return column-shaped `[N, 1]`
tensors rather than upstream's one-dimensional `[N]` tensors; the repository's
parser explicitly flattens them.

### 4. Detection predictor postprocessing

`DetectionPredictor.postprocess` is replaced so it calls the patched NMS and
constructs `Results` without truncating extended rows to six columns. Paths are
set to `None`. Newer upstream feature-extraction/`return_idxs` handling and the
end-to-end detector shortcut are not preserved by this experimental patch.

## Probability semantics and limitations

Upstream YOLO detection classification is trained with independent class
objectives and normally exposes independent sigmoid scores. Converting its raw
class logits to a vector that sums to one imposes a mutually exclusive
categorical interpretation after training. The vector is useful as an
experimental relative class distribution, but it is not automatically a
calibrated posterior.

The value `T=2.4` softens the distribution without changing the class ordering.
No calibration dataset, fitted objective, or held-out calibration metrics were
found for this YOLO temperature. It must therefore be treated as a historical
experimental hyperparameter, unlike the measured SegFormer temperature in the
SegFormer contract. Because thresholding uses the softened maximum probability,
the temperature can change which candidate boxes survive.

Additional scope limits:

- the implementation is verified for ordinary axis-aligned detection only;
- rotated NMS is unavailable through the compatibility import used by this
  patch with Ultralytics 8.4.82;
- the patch is process-global and `_PATCH_APPLIED` makes it one-shot, so the
  first requested temperature wins for that process;
- it relies on private Ultralytics internals and now fails explicitly unless
  `ultralytics.__version__ == "8.4.82"`;
- no automated model-quality or calibration evaluation for the modified scores
  exists in this repository.

## Reproduction and isolation strategy

The recommended near-term strategy is the one now represented in this repo:

1. keep the stock Ultralytics installation untouched;
2. pin the optional dependency to `ultralytics==8.4.82`;
3. keep the small, explicit runtime patch in
   `object_nav/perception/patches.py`;
4. preserve the weights by their SHA-256 identity;
5. load YOLO only through `object_nav.perception.build_yolo_detector`.

This makes the relationship with upstream obvious and isolates the experiment
from unrelated projects at the filesystem level. It does not isolate the patch
from other Ultralytics models in the same Python process, so do not mix patched
and unpatched detectors in one interpreter.

If reuse across several projects becomes active again, extract the current
adapter, patch, version guard, and regression smoke test into a small separately
versioned package that depends on exactly Ultralytics 8.4.82. A full Ultralytics
fork or vendoring the whole library is not justified by these four overrides.
An editable fork would become preferable only if upstream internals change
enough that the patch can no longer remain small and reviewable.

Recreate the audited environment behavior with:

```bash
conda activate habitat
python3 -m pip install "ultralytics==8.4.82"
python3 -m pip install --no-build-isolation -e .
```

Restore `models/yolo/yolo11x.pt`, verify its checksum, and construct the detector:

```python
from object_nav.perception import YoloConfig, build_yolo_detector

detector = build_yolo_detector(YoloConfig())
result = detector.detect(rgb_uint8)
```

Run the reproducible behavior checks for both variants. They use separate child
processes so the patched run cannot affect the upstream run:

```bash
python3 scripts/tools/check_yolo_contract.py --mode both --device cpu
python3 -m unittest -v tests.test_yolo_contract
```

The upstream check requires standard six-column boxes and no retained class
vector. The patched check requires 86-column boxes and 80-class vectors that sum
to one.

To repeat the installed-source audit without changing the environment:

```bash
python3 -m pip download --no-deps --only-binary=:all: \
  --dest /tmp/ultralytics-audit ultralytics==8.4.82
# Unpack the wheel, then compare its ultralytics/ directory with the path shown above.
diff -rq --exclude=__pycache__ --exclude='*.pyc' \
  /tmp/ultralytics-audit/upstream/ultralytics \
  "$CONDA_PREFIX/lib/python3.9/site-packages/ultralytics"
```

No output from the final command means the installed source matches the wheel.
