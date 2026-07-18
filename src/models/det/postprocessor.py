# src/models/det/postprocessor.py: convert raw classification/box maps into standard corners

import torch

from src.models.base.base_postprocessor import BasePostprocessor


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
