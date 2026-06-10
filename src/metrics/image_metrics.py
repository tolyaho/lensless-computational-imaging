from src.metrics.base_metric import BaseMetric


class ImageMetrics(BaseMetric):
    def __init__(self, metric, device, name=None):
        super().__init__(metric, device, name)

    def __call__(self, pred, target, **batch):
        raise NotImplementedError
