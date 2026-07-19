# src/core/evaluator.py: common test-split evaluator for final corner predictions

import json
import os

from tqdm import tqdm

from src.metrics.corner_distance import MeanCornerDistance, MaxCornerDistance, PCK
from src.metrics.polygon_iou import PolygonIoU
from src.metrics.success_rate import SuccessRate


def build_default_metrics():
    """Return fresh metric instances for one complete corner evaluation."""
    return {
        "iou": PolygonIoU(),
        "mcd": MeanCornerDistance(),
        "maxcd": MaxCornerDistance(),
        "pck_002": PCK(0.02),
        "pck_005": PCK(0.05),
        "sr": SuccessRate(),
    }


DEFAULT_METRICS = build_default_metrics


class Evaluator:
    """Evaluates one wrapper on a labeled dataloader and saves scalar results."""

    def __init__(self, wrapper, metrics=None, output_dir=None):
        self.wrapper = wrapper
        self.metrics = metrics or build_default_metrics()
        self.output_dir = output_dir

    def evaluate(self, dataloader):
        for metric in self.metrics.values():
            metric.reset()
        for images, targets in tqdm(dataloader, desc="test", leave=False, ascii=True):
            preds = self.wrapper.predict_step(images)
            target_array = targets.cpu().numpy()
            for metric in self.metrics.values():
                metric.update(preds, target_array)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def save(self, results, output_dir=None):
        output_dir = output_dir or self.output_dir
        if output_dir is None:
            raise ValueError("output_dir is required to save evaluation results")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return path
