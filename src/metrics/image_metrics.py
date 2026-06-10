import torch

from src.metrics.base_metric import BaseMetric


class ImageMetrics(BaseMetric):
    def __init__(self, metric, device, name=None):
        super().__init__(name=name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.metric = metric.to(self.device)

    @torch.no_grad()
    def __call__(self, pred=None, target=None, **batch):
        if pred is None:
            pred = batch.get("recon")
        if target is None:
            target = batch.get("lensed")
        if pred is None or target is None:
            raise KeyError("batch must contain recon and lensed")

        pred = pred.detach().clamp(0, 1).to(self.device)
        target = target.detach().clamp(0, 1).to(self.device)

        if hasattr(self.metric, "reset"):
            self.metric.reset()
        value = self.metric(pred, target)
        return value.item()
