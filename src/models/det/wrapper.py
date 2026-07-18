# src/models/det/wrapper.py: composes det model variants with their preprocessor/postprocessor/loss

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.det.model import DetModel, TorchDetModel
from src.models.det.preprocessor import DetPreprocessor
from src.models.det.postprocessor import DetPostprocessor
from src.models.det.torch_preprocessor import TorchDetPreprocessor
from src.models.det.torch_postprocessor import TorchDetPostprocessor
from src.losses.base_loss import BaseLoss
from src.losses.focal_loss import FocalLoss
from src.losses.smoothl1_loss import SmoothL1Loss
from src.metrics.polygon_iou import PolygonIoU


class DetWrapper(BaseWrapper):
    """Wraps DetModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, backbone="custom", head="box", model=None, neck_channels=256,
                 grid_stride=16, box_size=0.1, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None):
        # model kwarg accepted for CLI compatibility with get_wrapper_kwargs; DetModel has no variants
        model = DetModel(in_channels=in_channels, backbone=backbone, neck_channels=neck_channels,
                          grid_stride=grid_stride, head=head)
        preprocessor = preprocessor or DetPreprocessor(
            grid_stride=model.grid_stride, image_size=image_size,
            head=head, box_size=box_size)
        postprocessor = postprocessor or DetPostprocessor(
            grid_stride=model.grid_stride, image_size=image_size)
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                          scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        backbone_ids = {id(p) for p in self.model.extractor.parameters()}
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        self.set_optimizer(self.optimizer or AdamW([
            {"params": self.model.extractor.parameters(), "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ]))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"cls": FocalLoss(), "box": SmoothL1Loss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})


class TorchDetWrapper(BaseWrapper):
    """Wraps TorchDetModel with native torchvision detection train/eval semantics."""

    def __init__(self, backbone=None, head="box", model=None, box_size=0.1, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 metrics=None, device=None):
        # backbone/head kwargs accepted for CLI compatibility with get_wrapper_kwargs; unused here
        net = TorchDetModel(model=model)
        preprocessor = preprocessor or TorchDetPreprocessor(
            image_size=image_size, box_size=box_size, label_offset=net.label_offset)
        postprocessor = postprocessor or TorchDetPostprocessor(
            image_size=image_size, label_offset=net.label_offset)
        super().__init__(net, preprocessor, postprocessor, optimizer=optimizer,
                          scheduler=scheduler, losses=None, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or AdamW(self.model.parameters(), lr=1e-4))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        native_targets = self.preprocessor(targets)

        self.optimizer.zero_grad()
        loss_dict = self.model(list(images), native_targets)
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()

        for name, value in loss_dict.items():
            self.losses.setdefault(name, BaseLoss()).update(value.item(), len(images))
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        raw_output = self.model(list(images))
        self.compute_metrics(raw_output, targets)
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        raw_output = self.model(list(images.to(self.device, non_blocking=True)))
        preds = self.postprocessor(raw_output)
        return preds.cpu().numpy()
