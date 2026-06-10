from torch import nn


class ReconstructionLoss(nn.Module):
    """mse plus weighted lpips on reconstructions."""

    def __init__(self, lpips_weight=0.1, *args, **kwargs):
        super().__init__()
        self.lpips_weight = lpips_weight
        raise NotImplementedError

    def forward(self, recon, target, **batch):
        raise NotImplementedError
