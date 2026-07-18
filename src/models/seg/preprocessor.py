# src/models/seg/preprocessor.py: rasterize standard corners into a seg mask target

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.models.base.base_preprocessor import BasePreprocessor


class SegPreprocessor(BasePreprocessor):
    """Rasterizes (N, 4, 2) normalized corners into a (N, 1, mask_size, mask_size) binary mask."""

    def __init__(self, mask_size):
        self.mask_size = mask_size

    def __call__(self, corners):
        device = corners.device
        corners = corners.detach().cpu().numpy()
        masks = np.zeros((corners.shape[0], 1, self.mask_size, self.mask_size), dtype=np.float32)
        for i, quad in enumerate(corners):
            points = [(float(x) * self.mask_size, float(y) * self.mask_size) for x, y in quad]
            image = Image.new("L", (self.mask_size, self.mask_size), 0)
            ImageDraw.Draw(image).polygon(points, outline=1, fill=1)
            masks[i, 0] = np.array(image, dtype=np.float32)
        return torch.from_numpy(masks).to(device)
