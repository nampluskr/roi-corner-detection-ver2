# src/models/reg/wrapper.py: composes RegModel/RegPreprocessor/RegPostprocessor and WingLoss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.reg.model import RegModel
from src.models.reg.preprocessor import RegPreprocessor
from src.models.reg.postprocessor import RegPostprocessor
from src.losses.wing_loss import WingLoss
from src.metrics.polygon_iou import PolygonIoU


class RegWrapper(BaseWrapper):
    """Wraps RegModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, dropout=0.2, backbone="custom", head="coord_gap",
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None):
        model = RegModel(in_channels=in_channels, dropout=dropout, backbone=backbone, head=head)
        preprocessor = preprocessor or RegPreprocessor()
        postprocessor = postprocessor or RegPostprocessor()
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
        self.set_losses(self.losses or {"loss": WingLoss(apply_sigmoid=True)})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})
