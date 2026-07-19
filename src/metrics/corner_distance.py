# src/metrics/corner_distance.py: normalized-coordinate corner distance metrics

import numpy as np

from src.metrics.base_metric import BaseMetric


class CornerDistanceMetric(BaseMetric):
    """Base metric for Euclidean distances between corresponding corners."""

    def distances(self, preds, targets):
        pred = np.asarray(preds, dtype=np.float64).reshape(4, 2)
        target = np.asarray(targets, dtype=np.float64).reshape(4, 2)
        return np.linalg.norm(pred - target, axis=1)


class MeanCornerDistance(CornerDistanceMetric):
    """Computes dataset mean of sample-wise mean corner distances."""

    def __call__(self, preds, targets):
        return float(self.distances(preds, targets).mean())


class MaxCornerDistance(CornerDistanceMetric):
    """Computes dataset mean of sample-wise maximum corner distances."""

    def __call__(self, preds, targets):
        return float(self.distances(preds, targets).max())


class PCK(CornerDistanceMetric):
    """Computes the fraction of corners within one normalized distance threshold."""

    def __init__(self, threshold):
        self.threshold = threshold
        super().__init__()

    def __call__(self, preds, targets):
        return float((self.distances(preds, targets) <= self.threshold).mean())
