import torch

from src.utils.roi import crop_to_digicam_roi


def test_crops_full_sensor_frame():
    x = torch.randn(1, 3, 380, 507)
    out = crop_to_digicam_roi(x)
    assert out.shape == (1, 3, 200, 266)


def test_leaves_already_cropped_tensors_unchanged():
    x = torch.randn(1, 3, 200, 266)
    out = crop_to_digicam_roi(x)
    assert out.shape == x.shape
    assert torch.allclose(out, x)
