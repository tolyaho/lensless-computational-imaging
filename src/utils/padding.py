import torch


def center_pad(x: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """pad to padded admm space with the image centered."""
    raise NotImplementedError


def center_crop(x: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """crop back to the image window."""
    raise NotImplementedError


def padded_shape(height: int, width: int) -> tuple[int, int]:
    """roughly 2h x 2w, maybe fft-friendly."""
    raise NotImplementedError
