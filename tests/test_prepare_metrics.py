import torch

from src.metrics.prepare import prepare_metric_pair


def test_prepare_metric_pair_crops_to_scene_roi():
    pred = torch.rand(1, 3, 380, 507)
    target = torch.rand(1, 3, 380, 507)

    pred_roi, target_roi = prepare_metric_pair(pred, target, use_roi=True)

    assert pred_roi.shape == (1, 3, 200, 266)
    assert target_roi.shape == pred_roi.shape
