# Deployed SegFormer-B5 reference

This page is the compact contract for the semantic segmenter produced by this
repository. Read it before loading the checkpoint, changing inference code, or
integrating the model into ObjectNav. 

## What the model is

The accepted model is a fine-tuned **SegFormer-B5 semantic segmenter**. For each
RGB pixel it returns one categorical distribution over 41 outputs:

- model ID `0`: learned `unknown`;
- model IDs `1--40`: the 40 MPCAT40 indoor semantic classes;
- target value `255`: ignored supervision, never a model output.

It performs semantic segmentation, not instance segmentation. Two chairs may
receive the same class label, but the model does not assign separate instance
identities to them.

## Frozen model identity

| Field | Accepted value |
|---|---|
| Base checkpoint | `nvidia/segformer-b5-finetuned-ade-640-640` |
| Base revision | `739f5d4692954e4a185eac280dec1ba5a7d52f1d` |
| Project source commit | `1f7983a7069614fab936c12679cfe0a51ed9c37c` |
| Parameters | 84,624,873 |
| Output classes | 41 |
| Training objective | `0.8 * cross_entropy + 0.2 * Lovasz-Softmax` |
| Final training data | `train-all-v1`: 57,241 frames from 145 HM3D training scenes |
| Final duration | 11 passes; 39,358 optimizer steps |
| Calibration | Global scalar temperature `T = 1.3523482084274292` |

The ADE20K classifier was replaced by a randomly initialized 41-output
classifier. The pretrained encoder and compatible decoder features were then
fine-tuned on HM3D. The final refit started from the pinned ADE20K checkpoint,
not from a development probe. Cross-entropy trained all 41 outputs;
Lovasz-Softmax excluded `unknown`; target value `255` was excluded from both.

## Accepted checkpoint

The complete local scientific record is outside Git:

```text
/home/joaocb2002/hm3d-semseg-data/runs-server/segformer_b5_final/
```

The deployment checkpoint is:

```text
/home/joaocb2002/hm3d-semseg-data/runs-server/
└── segformer_b5_final/checkpoints/calibrated/
    ├── model.safetensors
    ├── config.json
    ├── checkpoint.json
    ├── camera_profile.yaml
    └── calibration.json
```

Treat these five files as one versioned unit. `model.safetensors` alone is not
a complete deployment artifact. `training_state.pt` is required only for exact
training resume and should not be copied into a normal inference bundle.

Run `sha256sum -c deployment.sha256` from the complete final-run directory
before accepting a transferred copy.

## Input and output contract

`SemanticSegmenter` accepts an RGB NumPy array with:

```text
shape: [H, W, 3]
dtype: uint8
channel order: RGB
value range: 0--255
```

Inference applies ImageNet normalization, performs no forced input resize, and
upsamples logits back to the input height and width. The returned dictionary is:

| Key | Shape | Meaning |
|---|---|---|
| `probabilities` | `[41, H, W]` | Calibrated categorical distribution per pixel |
| `labels` | `[H, W]` | `argmax` model ID as `uint8` |
| `confidence` | `[H, W]` | Maximum class probability |
| `entropy` | `[H, W]` | Entropy of the 41-class distribution |

The probability axis uses model IDs, not raw HM3D semantic IDs. Use the
repository taxonomy mapping; never infer class order from names or ADE20K.

## Minimal Python use

Install this package in the environment that runs ObjectNav, then load the
complete checkpoint directory:

```python
from pathlib import Path

from hm3d_semseg.inference import SemanticSegmenter

checkpoint = Path(
    "/home/joaocb2002/hm3d-semseg-data/runs-server/"
    "segformer_b5_final/checkpoints/calibrated"
)

segmenter = SemanticSegmenter.from_checkpoint(checkpoint, device="cuda")
result = segmenter(rgb_uint8)

probabilities = result["probabilities"]  # [41, H, W]
labels = result["labels"]                # [H, W]
confidence = result["confidence"]        # [H, W]
entropy = result["entropy"]              # [H, W]
```

When `temperature` is not passed explicitly,
`SemanticSegmenter.from_checkpoint(...)` reads it from `calibration.json`.
Do not divide logits by `T` again downstream.

## Camera contract

The model was built for observations matching the frozen ObjectNav camera
profile stored beside the checkpoint. The current profile is landscape
`640 x 480` RGB with `79` degree horizontal field of view and sensor position
`[0, 0.88, 0]` metres.

Resolve the active ObjectNav camera configuration and check compatibility once
during startup:

```python
from pathlib import Path

from hm3d_semseg.camera import resolve_camera_profile

runtime_camera = resolve_camera_profile(Path("/path/to/objectnav.yaml"))
segmenter.assert_camera(runtime_camera)
```

Do not silently bypass a mismatch. A changed field of view, projection, sensor
pose, width, or height can change the input distribution and geometric meaning
of projected pixels.

## Evaluation evidence

The frozen model was evaluated once on `official-val-v1`: 9,036 frames from 36
HM3D validation scenes excluded from training and recipe selection.

| Metric | Result |
|---|---:|
| Known-class mIoU | 43.24% |
| Overall pixel accuracy | 81.38% |
| ObjectNav-six mIoU | 59.08% |

The ObjectNav-six metric averages bed, chair, sofa, plant, toilet, and TV
monitor. It does not mean that all six classes perform equally well.

The SegFormer paper reports 51.0% single-scale B5 mIoU on ADE20K. That number is
context only: ADE20K and this HM3D protocol differ in taxonomy, domain, sampling,
training budget, and evaluation. Do not present the two results as a controlled
comparison.

## What calibration changes

Temperature scaling fitted one positive scalar on `calibration-fit-v1` (12
scenes) and was evaluated on the disjoint `calibration-evaluation-v1` (24
scenes).

| Metric on the 24-scene evaluation split | `T = 1` | `T = 1.3523` |
|---|---:|---:|
| NLL | 0.6950 | 0.6677 |
| ECE | 5.14% | 1.90% |
| Multiclass Brier score | 0.2753 | 0.2709 |

Because `T > 1` softens logits uniformly, calibration changes probabilities,
confidence, and entropy but not class ordering. Therefore it does not change
`labels`, pixel accuracy, or mIoU.

This is one global temperature. It is not class-conditional, scene-conditional,
or proof that every non-maximum class probability is perfectly calibrated. Its
validity is tied to data resembling the calibration distribution.

## Integration rules

- Load the model once and reuse it; do not rebuild it on every ObjectNav step.
- Use calibrated probabilities as uncertain evidence, not as ground truth.
- Preserve all 41 probabilities when updating a semantic map; do not retain
  only the winning class unless the downstream method explicitly requires it.
- Project pixels with depth and the matching camera pose before updating 3D
  state.
- Do not treat nearby pixels or consecutive frames as independent evidence.
  Their errors are strongly correlated, so naive repeated multiplication or
  unbounded Dirichlet accumulation can become overconfident.
- Keep `unknown` distinct from ignored or invalid geometry. The model may
  predict `unknown`; target value `255` must never enter the runtime belief
  vector.
- Preserve aspect ratio and the native output resolution unless a separately
  evaluated deployment policy says otherwise.
- Record checkpoint hash, temperature, camera-profile hash, package commit, and
  runtime device with ObjectNav results.

## Known limitations

- Single-frame RGB cannot exploit depth, temporal continuity, or multi-view 3D
  consistency by itself.
- Performance is non-uniform: structural classes are strong, while small,
  visually similar, or taxonomically broad classes remain harder.
- Frequent classes dominate pixel accuracy; use known-class mIoU and per-class
  IoU when diagnosing semantic quality.
- Examples include systematic confusions such as stool vs chair, blinds vs
  window, and shower vs wall.
- The model produces semantic regions, not object identities or persistent 3D
  tracks.

For the concise methodology, results, calibration plots, and held-out examples,
see the [final PDF report](../output/pdf/hm3d_semseg_final_report.pdf).
