# src/models/seg/wrapper.py: composes SegModel/SegPreprocessor/SegPostprocessor and BCE+Dice loss

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.seg.model import SUPPORTED_TORCHSEG_MODELS, SegModel, TorchSegModel
from src.models.seg.preprocessor import SegPreprocessor
from src.models.seg.postprocessor import SegPostprocessor
from src.losses.bce_loss import BCELoss
from src.losses.dice_loss import DiceLoss
from src.metrics.polygon_iou import PolygonIoU


class SegWrapper(BaseWrapper):
    """Wraps seg models behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, in_channels=3, backbone="custom", head="mask", model=None, image_size=224,
                 optimizer=None, scheduler=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None, warmup_epochs=1):
        # head kwarg accepted for CLI compatibility with get_wrapper_kwargs; seg has one head type
        net = self.build_model(model=model, in_channels=in_channels, backbone=backbone)
        preprocessor = preprocessor or SegPreprocessor(image_size // net.mask_stride)
        postprocessor = postprocessor or SegPostprocessor()
        is_custom = isinstance(net, SegModel) and backbone == "custom"
        applied_warmup_epochs = 0 if (is_custom or not hasattr(net, "extractor")) else warmup_epochs
        super().__init__(net, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device,
                         warmup_epochs=applied_warmup_epochs)
        self.applied_warmup_epochs = applied_warmup_epochs
        if self.optimizer is None:
            phase = 1 if applied_warmup_epochs > 0 else 2
            self.set_optimizer(self.build_optimizer(phase))
        if self.scheduler is None:
            self.set_scheduler(self.build_scheduler(self.optimizer))
        self.set_losses(self.losses or {"bce": BCELoss(), "dice": DiceLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def build_model(self, model=None, in_channels=3, backbone="custom"):
        if model in (None, "unet"):
            return SegModel(in_channels=in_channels, backbone=backbone)
        if model in SUPPORTED_TORCHSEG_MODELS:
            return TorchSegModel(model=model)
        supported = ("unet",) + SUPPORTED_TORCHSEG_MODELS
        raise ValueError("Unknown seg model: %s. Supported: %s" % (model, ", ".join(supported)))

    def build_optimizer(self, phase):
        if not hasattr(self.model, "extractor") or self.applied_warmup_epochs == 0:
            return AdamW(self.model.parameters(), lr=1e-4)
        backbone_ids = {id(p) for p in self.model.extractor.parameters()}
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        if phase == 1:
            return AdamW(head_params, lr=1e-4)
        return AdamW([
            {"params": self.model.extractor.parameters(), "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ])

    def build_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2,
                                 threshold=1e-4, threshold_mode="abs", min_lr=1e-7)
