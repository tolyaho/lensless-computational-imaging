import torch


def subtract_background(image: torch.Tensor, quantile: float = 0.6) -> torch.Tensor:
    """subtract a per-channel DC level from a high quantile estimate."""
    b, c = image.shape[0], image.shape[1]
    background = torch.quantile(image.reshape(b, c, -1), quantile, dim=2)
    return (image - background[:, :, None, None]).clamp_min(0.0)
