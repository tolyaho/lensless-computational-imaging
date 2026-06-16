import torch
import torch.nn.functional as F
from torch import nn

from src.model.drunet import DRUNetProcessor
from src.utils.fft import prepare_psf
from src.utils.padding import center_crop, center_pad, padded_shape


def inverse_softplus(x: float) -> float:
    value = torch.as_tensor(x, dtype=torch.float32)
    return torch.log(torch.expm1(value)).item()


class LearnedWienerInversion(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        init_lambda: float = 1e-3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.raw_lambda = nn.Parameter(
            torch.full((1, channels, 1, 1), inverse_softplus(init_lambda))
        )
        self.scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 1.0)
        )
        self.bias = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.0)
        )

    def forward(
        self,
        lensless: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h, w = lensless.shape[-2:]
        pad_shape = padded_shape(h, w)

        y = center_pad(lensless, pad_shape)
        y_fft = torch.fft.fft2(y)
        psf_fft = prepare_psf(mask, pad_shape).to(
            device=y.device,
            dtype=y_fft.dtype,
        )

        lam = F.softplus(self.raw_lambda).to(
            device=y.device,
            dtype=y.dtype,
        ) + self.eps

        denom = psf_fft.abs().square() + lam
        inv_filter = psf_fft.conj() / denom

        x_fft = inv_filter * y_fft
        x = torch.fft.ifft2(x_fft).real
        x = center_crop(x, (h, w))

        scale = self.scale.to(device=y.device, dtype=y.dtype)
        bias = self.bias.to(device=y.device, dtype=y.dtype)
        x = scale * x + bias
        return x


class FlatNetLite(nn.Module):
    def __init__(
        self,
        inversion: nn.Module | None = None,
        enhancer: nn.Module | None = None,
        clamp_output: bool = True,
    ):
        super().__init__()
        self.inversion = inversion or LearnedWienerInversion()
        self.enhancer = enhancer or DRUNetProcessor(
            channels=3,
            features=(32, 64, 128, 256),
            residual_scale=0.1,
            clamp_output=False,
        )
        self.clamp_output = clamp_output

    def forward(
        self,
        lensless: torch.Tensor,
        mask: torch.Tensor,
        **batch,
    ):
        x0 = self.inversion(lensless, mask)
        recon = self.enhancer(x0)

        if self.clamp_output:
            recon = torch.clamp(recon, 0.0, 1.0)

        return {
            "recon": recon,
            "flatnet_intermediate": x0,
        }
