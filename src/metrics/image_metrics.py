import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.prepare import prepare_metric_pair


class ImageMetrics(BaseMetric):
    def __init__(self, metric, device, name=None, use_roi=True):
        super().__init__(name=name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.metric = metric.to(self.device)
        self.use_roi = use_roi

    @torch.no_grad()
    def __call__(self, pred=None, target=None, **batch):
        if pred is None:
            pred = batch.get("recon")
        if target is None:
            target = batch.get("lensed")
        if pred is None or target is None:
            raise KeyError("batch must contain recon and lensed")

        pred, target = prepare_metric_pair(pred, target, use_roi=self.use_roi)
        pred = pred.to(self.device)
        target = target.to(self.device)

        if hasattr(self.metric, "reset"):
            self.metric.reset()
        value = self.metric(pred, target)
        return value.item()
