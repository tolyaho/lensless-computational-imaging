import numpy as np
import torch

from lensless_helpers.psf import simulate_psf_from_mask


def _psf_to_chw(psf: torch.Tensor) -> torch.Tensor:
    if psf.ndim == 4:
        psf = psf.squeeze(0)
    if psf.ndim != 3:
        raise ValueError(f"expected psf with shape [H, W, C] or [1, H, W, C], got {tuple(psf.shape)}")
    return psf.permute(2, 0, 1).contiguous().float()


def mask_vals_to_psf(mask_vals: np.ndarray, **kwargs) -> torch.Tensor:
    psf = simulate_psf_from_mask(mask_vals, **kwargs)
    return _psf_to_chw(psf)


def mask_to_sensor_psf(mask: torch.Tensor, sensor_shape: tuple[int, int] | None = None) -> torch.Tensor:
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    return mask_vals_to_psf(arr)


def lcd_to_psf_simple(mask: torch.Tensor, shape: tuple[int, int] | None = None) -> torch.Tensor:
    return mask_to_sensor_psf(mask, sensor_shape=shape)
