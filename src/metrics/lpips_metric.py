import torch

from src.metrics.base_metric import BaseMetric


class LPIPSMetric(BaseMetric):
    def __init__(self, metric, device, name="LPIPS"):
        super().__init__(name=name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.metric = metric.to(self.device)
        self.metric.eval()

    @staticmethod
    def _to_lpips_range(x: torch.Tensor) -> torch.Tensor:
        return x.detach().clamp(0, 1) * 2 - 1

    @torch.no_grad()
    def __call__(self, pred=None, target=None, **batch):
        if pred is None:
            pred = batch.get("recon")
        if target is None:
            target = batch.get("lensed")
        if pred is None or target is None:
            raise KeyError("batch must contain recon and lensed")

        pred = self._to_lpips_range(pred).to(self.device)
        target = self._to_lpips_range(target).to(self.device)

        value = self.metric(pred, target)
        return value.mean().item()
