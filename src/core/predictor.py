# src/core/predictor.py: prediction CSV writer for final corner predictions

import csv
import os

import numpy as np
from tqdm import tqdm


COLUMN_NAMES = [
    "index", "success", "failure_reason",
    "target_x1", "target_y1", "target_x2", "target_y2", "target_x3", "target_y3", "target_x4", "target_y4",
    "pred_x1", "pred_y1", "pred_x2", "pred_y2", "pred_x3", "pred_y3", "pred_x4", "pred_y4",
]


class Predictor:
    """Collects final corner predictions from a labeled dataloader and saves them as CSV."""

    def __init__(self, wrapper, output_dir=None):
        self.wrapper = wrapper
        self.output_dir = output_dir

    def predict(self, dataloader):
        rows = []
        index = 0
        for images, targets in tqdm(dataloader, desc="predict", leave=False, ascii=True):
            preds = self.wrapper.predict_step(images)
            target_array = targets.cpu().numpy()
            for pred, target in zip(preds, target_array):
                success = bool(np.isfinite(pred).all())
                rows.append(self.build_row(index, pred, target, success))
                index += 1
        return rows

    def build_row(self, index, pred, target, success):
        target_values = np.asarray(target, dtype=np.float32).reshape(8).tolist()
        pred_values = np.asarray(pred, dtype=np.float32).reshape(8).tolist()
        return [index, success, "" if success else "invalid_prediction"] + target_values + pred_values

    def save(self, rows, output_dir=None):
        output_dir = output_dir or self.output_dir
        if output_dir is None:
            raise ValueError("output_dir is required to save predictions")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "predictions.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMN_NAMES)
            writer.writerows(rows)
        return path
