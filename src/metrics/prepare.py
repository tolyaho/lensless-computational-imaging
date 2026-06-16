import torch
import torch.nn.functional as F

from src.utils.roi import crop_to_digicam_roi


def prepare_metric_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    use_roi: bool = True,
    detach: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """shared loss/metric prep."""
    if detach:
        pred = pred.detach()
        target = target.detach()

    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    if pred.shape[-2:] != target.shape[-2:]:
        # match lensed frame
        pred = F.interpolate(
            pred,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if use_roi:
        pred = crop_to_digicam_roi(pred)
        target = crop_to_digicam_roi(target)

    return pred.contiguous(), target.contiguous()
