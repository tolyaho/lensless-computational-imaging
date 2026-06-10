from src.metrics.base_metric import BaseMetric


class LPIPSMetric(BaseMetric):
    def __init__(self, metric, device, name="LPIPS"):
        super().__init__(metric, device, name)

    def __call__(self, pred, target, **batch):
        """lpips wants [-1, 1]; the pipeline stays in [0, 1]."""
        raise NotImplementedError
