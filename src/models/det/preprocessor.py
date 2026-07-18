# src/models/det/preprocessor.py: convert standard corners into per-cell classification and box/point targets

import torch

from src.models.base.base_preprocessor import BasePreprocessor

BOX_CHANNELS = {"box": 4, "point": 2}


class DetTarget(dict):
    """Dict-based target whose __len__ reports batch size instead of key count for BaseLoss weighting."""

    def __len__(self):
        return self["cls"].shape[0]


class DetPreprocessor(BasePreprocessor):
    """Assigns each of the 4 corners to one grid cell, producing (N,4,Gh,Gw) cls and (N,C,Gh,Gw) box targets."""

    def __init__(self, grid_stride=16, image_size=224, head="box", box_size=0.1):
        if head not in BOX_CHANNELS:
            raise ValueError("Unknown det head: %s. Supported: %s"
                              % (head, ", ".join(BOX_CHANNELS)))
        self.grid_h = image_size // grid_stride
        self.grid_w = image_size // grid_stride
        self.head = head
        self.box_size = box_size

    def __call__(self, corners):
        device = corners.device
        n = corners.shape[0]
        channels = BOX_CHANNELS[self.head]
        cls_target = torch.zeros(n, 4, self.grid_h, self.grid_w, device=device)
        box_target = torch.zeros(n, channels, self.grid_h, self.grid_w, device=device)
        idx = torch.arange(n, device=device)

        for c in range(4):
            x = corners[:, c, 0].clamp(0.0, 1.0 - 1e-6)
            y = corners[:, c, 1].clamp(0.0, 1.0 - 1e-6)
            gx = (x * self.grid_w).long()
            gy = (y * self.grid_h).long()
            dx = x * self.grid_w - gx.float()
            dy = y * self.grid_h - gy.float()

            cls_target[idx, c, gy, gx] = 1.0
            box_target[idx, 0, gy, gx] = dx
            box_target[idx, 1, gy, gx] = dy
            if self.head == "box":
                box_target[idx, 2, gy, gx] = self.box_size
                box_target[idx, 3, gy, gx] = self.box_size

        pos_mask = cls_target.amax(dim=1, keepdim=True)
        return DetTarget(cls=cls_target, box=box_target, pos_mask=pos_mask)
