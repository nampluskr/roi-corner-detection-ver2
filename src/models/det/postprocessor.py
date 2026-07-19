# src/models/det/postprocessor.py: convert det variant raw outputs into standard corners

import torch
from ultralytics.utils.nms import non_max_suppression

from src.models.base.base_postprocessor import BasePostprocessor

NUM_CORNER_CLASSES = 4


class DetPostprocessor(BasePostprocessor):
    """Selects the highest-confidence cell per corner class and decodes its center offset to (N,4,2)."""

    def __init__(self, grid_stride=16, image_size=224):
        self.grid_h = image_size // grid_stride
        self.grid_w = image_size // grid_stride

    def __call__(self, raw_output):
        cls_logits = raw_output["cls"]
        box_raw = raw_output["box"]
        n = cls_logits.shape[0]
        device = cls_logits.device

        cls_prob = torch.sigmoid(cls_logits).reshape(n, 4, -1)
        best = cls_prob.argmax(dim=-1)
        gy = best // self.grid_w
        gx = best % self.grid_w
        offset = torch.sigmoid(box_raw[:, 0:2])

        idx = torch.arange(n, device=device)
        corners = torch.zeros(n, 4, 2, device=device)
        for c in range(4):
            dx = offset[idx, 0, gy[:, c], gx[:, c]]
            dy = offset[idx, 1, gy[:, c], gx[:, c]]
            corners[:, c, 0] = (gx[:, c].float() + dx) / self.grid_w
            corners[:, c, 1] = (gy[:, c].float() + dy) / self.grid_h
        return corners


class TorchDetPostprocessor(BasePostprocessor):
    """Selects the highest-scoring box per corner class and decodes its center to (N,4,2)."""

    def __init__(self, image_size=224, label_offset=1):
        self.image_size = image_size
        self.label_offset = label_offset

    def __call__(self, raw_output):
        n = len(raw_output)
        corners = torch.full((n, NUM_CORNER_CLASSES, 2), 0.5)
        for i, pred in enumerate(raw_output):
            boxes, labels, scores = pred["boxes"], pred["labels"], pred["scores"]
            for c in range(NUM_CORNER_CLASSES):
                mask = labels == (c + self.label_offset)
                if not mask.any():
                    continue
                box = boxes[mask][scores[mask].argmax()]
                corners[i, c, 0] = (box[0] + box[2]) / 2 / self.image_size
                corners[i, c, 1] = (box[1] + box[3]) / 2 / self.image_size
        return corners


class YoloDetPostprocessor(BasePostprocessor):
    """Runs Ultralytics NMS on a decoded eval-mode tensor and decodes it to (N,4,2) corners."""

    def __init__(self, image_size=224, conf_thres=0.001, iou_thres=0.5, max_det=10):
        self.image_size = image_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def __call__(self, decoded):
        results = non_max_suppression(decoded, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                       max_det=self.max_det, nc=NUM_CORNER_CLASSES)
        corners = torch.full((len(results), NUM_CORNER_CLASSES, 2), 0.5)
        for i, pred in enumerate(results):
            boxes, scores, labels = pred[:, :4], pred[:, 4], pred[:, 5]
            for c in range(NUM_CORNER_CLASSES):
                mask = labels == c
                if not mask.any():
                    continue
                box = boxes[mask][scores[mask].argmax()]
                corners[i, c, 0] = (box[0] + box[2]) / 2 / self.image_size
                corners[i, c, 1] = (box[1] + box[3]) / 2 / self.image_size
        return corners


class DetrDetPostprocessor(BasePostprocessor):
    """Selects the highest-scoring query per corner class and decodes its box center."""

    def __call__(self, raw_output):
        logits = raw_output.logits
        boxes = raw_output.pred_boxes
        scores = torch.softmax(logits, dim=-1)[:, :, :NUM_CORNER_CLASSES]
        n = logits.shape[0]
        corners = torch.zeros((n, NUM_CORNER_CLASSES, 2), device=boxes.device)
        idx = torch.arange(n, device=boxes.device)
        for c in range(NUM_CORNER_CLASSES):
            best = scores[:, :, c].argmax(dim=1)
            corners[:, c] = boxes[idx, best, 0:2].clamp(0.0, 1.0)
        return corners
