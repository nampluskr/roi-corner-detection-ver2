# src/metrics/success_rate.py: validity rate for corner predictions

import numpy as np

from src.metrics.base_metric import BaseMetric


class SuccessRate(BaseMetric):
    """Computes the fraction of samples with finite predicted corner values."""

    def update(self, preds, targets):
        for pred in preds:
            self.total += float(np.isfinite(pred).all())
            self.count += 1

    def __call__(self, preds, targets):
        return float(np.isfinite(preds).all())
