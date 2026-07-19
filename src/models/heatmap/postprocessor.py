# src/models/heatmap/postprocessor.py: convert raw heatmap logits into standard corners

import torch

from src.models.base.base_postprocessor import BasePostprocessor


class HeatmapPostprocessor(BasePostprocessor):
    """Applies soft-argmax to four corner heatmap logits and returns normalized corners."""

    def __init__(self, beta=10.0):
        self.beta = beta

    def __call__(self, raw_output):
        n, c, height, width = raw_output.shape
        logits = raw_output.reshape(n, c, height * width) * self.beta
        probs = torch.softmax(logits, dim=2)
        xs = torch.linspace(0.0, 1.0, width, device=raw_output.device, dtype=raw_output.dtype)
        ys = torch.linspace(0.0, 1.0, height, device=raw_output.device, dtype=raw_output.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        x = (probs * grid_x.reshape(1, 1, height * width)).sum(dim=2)
        y = (probs * grid_y.reshape(1, 1, height * width)).sum(dim=2)
        return torch.stack([x, y], dim=2)
