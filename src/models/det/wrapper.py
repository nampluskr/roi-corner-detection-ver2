# src/models/det/wrapper.py: composes det model variants with their preprocessor/postprocessor/loss

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.det.model import DetModel, DetrDetModel, TorchDetModel, YoloDetModel
from src.models.det.preprocessor import (
    DetPreprocessor, DetrDetPreprocessor, TorchDetPreprocessor, YoloDetPreprocessor)
from src.models.det.postprocessor import (
    DetPostprocessor, DetrDetPostprocessor, TorchDetPostprocessor, YoloDetPostprocessor)
from src.losses.base_loss import BaseLoss
from src.losses.focal_loss import FocalLoss
from src.losses.smoothl1_loss import SmoothL1Loss
from src.metrics.polygon_iou import PolygonIoU

YOLODET_LOSS_NAMES = ("box", "cls", "dfl")


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


class YoloDetWrapper(BaseWrapper):
    """Wraps YoloDetModel with native Ultralytics v8DetectionLoss train/eval semantics."""

    def __init__(self, backbone=None, head="box", model=None, box_size=0.3, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 metrics=None, device=None):
        # backbone/head kwargs accepted for CLI compatibility with get_wrapper_kwargs; unused here
        # box_size defaults larger than TorchDetWrapper's 0.1: dense-anchor assigner needs
        # enough positive matches for target_scores_sum to stay well above 1
        net = YoloDetModel(model=model)
        preprocessor = preprocessor or YoloDetPreprocessor(box_size=box_size)
        postprocessor = postprocessor or YoloDetPostprocessor(image_size=image_size)
        super().__init__(net, preprocessor, postprocessor, optimizer=optimizer,
                          scheduler=scheduler, losses=None, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or AdamW(self.model.parameters(), lr=1e-4))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def build_batch(self, images, targets):
        batch = self.preprocessor(targets)
        batch["img"] = images
        return batch

    def update_yolo_losses(self, loss_detach, count):
        for name, value in zip(YOLODET_LOSS_NAMES, loss_detach):
            self.losses.setdefault(name, BaseLoss()).update(value.item(), count)

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        batch = self.build_batch(images, targets)

        self.optimizer.zero_grad()
        raw_output = self.model.net(images)
        loss, loss_detach = self.model.net.loss(batch, preds=raw_output)
        loss.sum().backward()
        self.optimizer.step()

        self.update_yolo_losses(loss_detach, len(images))
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        batch = self.build_batch(images, targets)

        decoded, raw_dict = self.model.net(images)
        _, loss_detach = self.model.net.loss(batch, preds=raw_dict)
        self.update_yolo_losses(loss_detach, len(images))

        preds = self.postprocessor(decoded).to(self.device)
        self.update_metrics(preds.cpu().numpy(), targets.cpu().numpy())
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        decoded, _ = self.model.net(images.to(self.device, non_blocking=True))
        preds = self.postprocessor(decoded)
        return preds.cpu().numpy()


class DetrDetWrapper(BaseWrapper):
    """Wraps DetrDetModel with Hugging Face native Hungarian loss semantics."""

    def build_optimizer(self):
        """Return DETR fine-tuning parameter groups with conservative pretrained-model learning rates."""
        backbone_params, classifier_params, other_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("net.model.backbone"):
                backbone_params.append(param)
            elif name.startswith("net.class_labels_classifier"):
                classifier_params.append(param)
            else:
                other_params.append(param)
        return AdamW([
            {"params": backbone_params, "lr": 1e-5},
            {"params": other_params, "lr": 1e-4},
            {"params": classifier_params, "lr": 1e-4},
        ], weight_decay=1e-4)

    def __init__(self, backbone=None, head="box", model=None, box_size=0.3, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 metrics=None, device=None, grad_clip=1.0):
        # backbone/head/image_size kwargs accepted for CLI compatibility; unused by HF DETR here
        net = DetrDetModel(model=model)
        preprocessor = preprocessor or DetrDetPreprocessor(box_size=box_size)
        postprocessor = postprocessor or DetrDetPostprocessor()
        super().__init__(net, preprocessor, postprocessor, optimizer=optimizer,
                          scheduler=scheduler, losses=None, metrics=metrics, device=device)
        self.grad_clip = grad_clip
        self.set_optimizer(self.optimizer or self.build_optimizer())
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def update_detr_losses(self, loss_dict, count):
        for name, value in loss_dict.items():
            self.losses.setdefault(name, BaseLoss()).update(value.item(), count)

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        labels = self.preprocessor(targets)

        self.optimizer.zero_grad()
        output = self.model(images, labels=labels)
        output.loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.update_detr_losses(output.loss_dict, len(images))
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        labels = self.preprocessor(targets)

        output = self.model(images, labels=labels)
        self.update_detr_losses(output.loss_dict, len(images))
        preds = self.postprocessor(output)
        self.update_metrics(preds.cpu().numpy(), targets.cpu().numpy())
        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        output = self.model(images.to(self.device, non_blocking=True))
        preds = self.postprocessor(output)
        return preds.cpu().numpy()
