from __future__ import annotations

import torch
import torchvision

from object_nav.perception.ultralytics_compat import LOGGER, nms_rotated, xywh2xyxy

_PATCH_APPLIED = False


def apply_yolo_softmax_patch(*, temperature: float = 2.4) -> None:
    """Patch Ultralytics YOLO to expose softmax class probabilities.

    This monkey-patches Ultralytics internals to:
    - apply softmax to class logits during NMS
    - preserve the full class probability vector in Results/Boxes
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import ultralytics.engine.results as results_mod
    import ultralytics.utils.ops as ops_mod
    from ultralytics.nn.modules.head import Detect
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils.ops import convert_torch2numpy_batch, scale_boxes

    # Patch Detect._inference to return raw logits (remove sigmoid)
    def _inference_no_sigmoid(self, x):
        """Decode boxes and return raw class logits (no sigmoid)."""
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"]), 1)

    Detect._inference = _inference_no_sigmoid

    # Load original Boxes class
    OriginalBoxes = results_mod.Boxes

    class PatchedBoxes(OriginalBoxes):
        """Boxes that expose class probabilities in `.probs`."""

        def __init__(self, boxes, orig_shape):
            if boxes.ndim == 1:
                boxes = boxes[None, :]
            n = boxes.shape[-1]

            self.orig_shape = orig_shape
            self.is_track = False
            self.num_classes = 0

            if n == 6:
                self.format = "xyxy_conf_cls"
            elif n == 7:
                self.format = "xyxy_conf_cls_track"
                self.is_track = True
            else:
                self.format = "xyxy_conf_cls_probs"
                self.num_classes = n - 6

            self.data = boxes

        @property
        def conf(self):
            if self.data.shape[1] > 6:
                return self.data[:, 6:].max(1, keepdim=True).values
            return self.data[:, 4:5]

        @property
        def cls(self):
            if self.data.shape[1] > 6:
                return self.data[:, 6:].argmax(1).to(torch.int)
            return self.data[:, 5].to(torch.int)

        @property
        def probs(self):
            return self.data[:, 6:] if self.data.shape[1] > 6 else None

    results_mod.Boxes = PatchedBoxes

    def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
        end2end=False,
    ):
        # multi_label, max_time_img, end2end are kept for signature compatibility
        assert 0 <= conf_thres <= 1
        assert 0 <= iou_thres <= 1

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        if classes is not None:
            classes = torch.tensor(classes, device=prediction.device)

        bs = prediction.shape[0]
        nc = nc or (prediction.shape[1] - 4)
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc
        # Filter by max softmax probability (logits -> softmax)
        cls_logits = prediction[:, 4:mi]
        cls_probs = torch.softmax(cls_logits / temperature, dim=1)
        # First-stage filter keeps only candidates likely to pass NMS
        xc = cls_probs.max(1).values > conf_thres
        prediction = prediction.transpose(-1, -2)

        if not rotated:
            if in_place:
                prediction[..., :4] = xywh2xyxy(prediction[..., :4])
            else:
                prediction = torch.cat(
                    (xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1
                )

        output = [torch.zeros((0, 6 + nc + nm), device=prediction.device)] * bs

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            if labels and len(labels[xi]) and not rotated:
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
                v[:, :4] = xywh2xyxy(lb[:, 1:5])
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
                x = torch.cat((x, v), 0)

            if not x.shape[0]:
                continue

            box, cls_logits, mask = x.split((4, nc, nm), 1)

            # cls_logits are raw logits after patching Detect._inference
            cls_probs = torch.softmax(cls_logits / temperature, dim=1)
            max_prob, j = cls_probs.max(1, keepdim=True)

            # Second-stage filter keeps consistency with standard Ultralytics NMS flow
            x = torch.cat((box, max_prob, j.float(), cls_probs, mask), 1)[
                max_prob.view(-1) > conf_thres
            ]
            if classes is not None:
                x = x[(x[:, 5:6] == classes).any(1)]

            n = x.shape[0]
            if not n:
                continue

            if n > max_nms:
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            scores = x[:, 4]

            if rotated:
                if nms_rotated is None:
                    raise RuntimeError("nms_rotated is not available in this Ultralytics version.")
                boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
                i = nms_rotated(boxes, scores, iou_thres)
            else:
                boxes = x[:, :4] + c
                i = torchvision.ops.nms(boxes, scores, iou_thres)

            output[xi] = x[i[:max_det]]

        return output

    ops_mod.non_max_suppression = non_max_suppression

    def patched_postprocess(self, preds, img, orig_imgs, **kwargs):
        preds = ops_mod.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            if len(pred) > 0:
                pred[:, :4] = scale_boxes(
                    img[i].shape[1:], pred[:, :4], orig_imgs[i].shape[:2]
                )
            results.append(
                results_mod.Results(
                    orig_imgs[i],
                    path=None,
                    names=self.model.names,
                    boxes=pred,
                )
            )
        return results

    DetectionPredictor.postprocess = patched_postprocess

    if LOGGER is not None:
        LOGGER.info("Applied YOLO softmax patch for class probabilities.")

    _PATCH_APPLIED = True
