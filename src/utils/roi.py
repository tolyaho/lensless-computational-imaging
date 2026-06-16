import torch

from lensless_helpers.preprocessor import ALIGNMENT

_ROI_TOP, _ROI_LEFT = ALIGNMENT["top_left"]
_ROI_H = ALIGNMENT["height"]
_ROI_W = ALIGNMENT["width"]


def crop_to_digicam_roi(x: torch.Tensor) -> torch.Tensor:
    """crop to the displayed scene window in sensor coordinates."""
    h, w = x.shape[-2:]
    if h < _ROI_TOP + _ROI_H or w < _ROI_LEFT + _ROI_W:
        return x
    return x[
        ...,
        _ROI_TOP : _ROI_TOP + _ROI_H,
        _ROI_LEFT : _ROI_LEFT + _ROI_W,
    ].contiguous()
