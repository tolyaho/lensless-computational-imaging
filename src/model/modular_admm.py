import torch
from torch import nn


class ModularADMM(nn.Module):
    """pre/admm/post wrapper."""

    def __init__(
        self,
        admm_core: nn.Module,
        pre_processor: nn.Module | None = None,
        post_processor: nn.Module | None = None,
    ):
        super().__init__()
        self.admm_core = admm_core
        self.pre_processor = pre_processor
        self.post_processor = post_processor

    def forward(self, lensless: torch.Tensor, mask: torch.Tensor, **batch):
        y = lensless

        if self.pre_processor is not None:
            y = self.pre_processor(y)

        out = self.admm_core(lensless=y, mask=mask, **batch)
        recon = out["recon"]

        if self.post_processor is not None:
            recon = self.post_processor(recon)

        return {"recon": recon}
