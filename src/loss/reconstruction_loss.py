import lpips
import torch
from torch import nn

from src.metrics.prepare import prepare_metric_pair


class ReconstructionLoss(nn.Module):
    def __init__(self, mse_weight=1.0, lpips_weight=0.0, use_roi=True):
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.use_roi = use_roi

        if lpips_weight > 0:
            self.lpips = lpips.LPIPS(net="vgg")
            for p in self.lpips.parameters():
                p.requires_grad = False
        else:
            self.lpips = None

    def forward(self, recon=None, target=None, lensed=None, **batch):
        if target is None:
            target = lensed

        if recon is None:
            recon = batch.get("recon")
        if target is None:
            target = batch.get("lensed")

        if recon is None or target is None:
            raise KeyError("ReconstructionLoss needs recon and lensed/target")

        recon, target = prepare_metric_pair(
            recon,
            target,
            use_roi=self.use_roi,
            detach=False,
        )

        mse_loss = torch.nn.functional.mse_loss(recon, target)
        loss = self.mse_weight * mse_loss

        out = {
            "loss": loss,
            "mse_loss": mse_loss.detach(),
        }

        if self.lpips is not None:
            # LPIPS expects [-1, 1], while the rest of the pipeline uses [0, 1].
            recon_lpips = recon * 2 - 1
            target_lpips = target.detach() * 2 - 1
            lpips_loss = self.lpips(recon_lpips, target_lpips).mean()
            loss = loss + self.lpips_weight * lpips_loss

            out["loss"] = loss
            out["lpips_loss"] = lpips_loss.detach()

        return out
