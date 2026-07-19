# src/models/reg/wrapper.py: composes CustomRegModel/TorchRegModel with RegPreprocessor/RegPostprocessor and WingLoss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.reg.model import CustomRegModel, TorchRegModel
from src.models.reg.preprocessor import RegPreprocessor
from src.models.reg.postprocessor import RegPostprocessor
from src.losses.wing_loss import WingLoss
from src.metrics.polygon_iou import PolygonIoU


def build_model(in_channels, dropout, backbone, head):
    backbone = backbone or "custom"
    if backbone == "custom":
        return CustomRegModel(in_channels=in_channels, dropout=dropout, head=head)
    return TorchRegModel(backbone=backbone, dropout=dropout, head=head)


class RegWrapper(BaseWrapper):
    """Wraps CustomRegModel/TorchRegModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, dropout=0.2, backbone="custom", head="gap",
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None, warmup_epochs=1):
        model = build_model(in_channels, dropout, backbone, head)
        preprocessor = preprocessor or RegPreprocessor()
        postprocessor = postprocessor or RegPostprocessor()
        applied_warmup_epochs = 0 if isinstance(model, CustomRegModel) else warmup_epochs
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device,
                         warmup_epochs=applied_warmup_epochs)
        self.applied_warmup_epochs = applied_warmup_epochs
        if self.optimizer is None:
            phase = 1 if applied_warmup_epochs > 0 else 2
            self.set_optimizer(self.build_optimizer(phase))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_losses(self.losses or {"loss": WingLoss(apply_sigmoid=True)})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def build_optimizer(self, phase):
        if isinstance(self.model, CustomRegModel):
            return AdamW(self.model.parameters(), lr=1e-4)
        non_backbone_params = self._non_backbone_params()
        if phase == 1:
            return AdamW(non_backbone_params, lr=1e-4)
        return AdamW([
            {"params": self.model.extractor.parameters(), "lr": 1e-5},
            {"params": non_backbone_params, "lr": 1e-4},
        ])

    def build_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                                 threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def _non_backbone_params(self):
        backbone_ids = {id(p) for p in self.model.extractor.parameters()}
        return [p for p in self.model.parameters() if id(p) not in backbone_ids]
