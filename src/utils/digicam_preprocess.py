import numpy as np
import torch

from lensless_helpers.preprocessor import convert_image_to_float, force_rgb, get_cropped_lensed


def rotate_lensless(lensless: torch.Tensor) -> torch.Tensor:
    return torch.rot90(lensless, k=2, dims=(-2, -1))


def align_lensed(lensed: torch.Tensor, lensless_hw: tuple[int, int]) -> torch.Tensor:
    lensed_hwc = lensed.permute(1, 2, 0).numpy()
    if lensed_hwc.max() <= 1.0:
        lensed_hwc = (lensed_hwc * 255.0).round().astype(np.uint8)

    dummy_lensless = np.zeros(lensless_hw + (3,), dtype=np.float32)
    aligned_hwc = get_cropped_lensed(
        convert_image_to_float(force_rgb(lensed_hwc)),
        dummy_lensless,
    )
    return torch.from_numpy(aligned_hwc).permute(2, 0, 1).contiguous().float()
