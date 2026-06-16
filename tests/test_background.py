import torch

from src.utils.background import subtract_background


def test_removes_constant_per_channel_pedestal():
    x = torch.full((1, 3, 4, 4), 0.3)
    x[0, :, 0, 0] = 0.9

    out = subtract_background(x, quantile=0.5)

    assert torch.allclose(out[0, :, 1, 1], torch.zeros(3), atol=1e-6)
    assert torch.allclose(out[0, :, 0, 0], torch.full((3,), 0.6), atol=1e-6)


def test_background_is_estimated_per_channel():
    x = torch.zeros((1, 3, 2, 2))
    x[0, 0] = 0.1
    x[0, 1] = 0.5
    x[0, 2] = 0.9

    out = subtract_background(x, quantile=0.5)

    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_never_returns_negative_values():
    x = torch.rand((2, 3, 8, 8))

    out = subtract_background(x, quantile=0.6)

    assert (out >= 0).all()


def test_higher_quantile_subtracts_more_background():
    x = torch.arange(10, dtype=torch.float32).reshape(1, 1, 1, 10) / 10.0

    low = subtract_background(x, quantile=0.2)
    high = subtract_background(x, quantile=0.8)

    assert high.sum() < low.sum()


def test_batch_items_are_independent():
    x = torch.rand((4, 3, 5, 6))
    out = subtract_background(x, quantile=0.5)

    x_shifted = x.clone()
    x_shifted[0] += 10.0
    out_shifted = subtract_background(x_shifted, quantile=0.5)

    assert out.shape == x.shape
    assert torch.allclose(out_shifted[1:], out[1:], atol=1e-6)
