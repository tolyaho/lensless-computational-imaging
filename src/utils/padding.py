import torch
import torch.nn.functional as F


def center_pad(x: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = target_shape
    h, w = x.shape[-2:]

    if h > target_h or w > target_w:
        raise ValueError(f"target {target_shape} is smaller than input {(h, w)}")

    pad_h, pad_w = target_h - h, target_w - w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    return F.pad(x, (left, right, top, bottom))


def center_crop(x: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = target_shape
    h, w = x.shape[-2:]

    if h < target_h or w < target_w:
        raise ValueError(f"target {target_shape} is larger than input {(h, w)}")

    top = (h - target_h) // 2
    left = (w - target_w) // 2

    return x[..., top : top + target_h, left : left + target_w]


def padded_shape(height: int, width: int) -> tuple[int, int]:
    return 2 * height, 2 * width
