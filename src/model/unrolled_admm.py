import torch
import torch.nn.functional as F
from torch import nn

from src.utils.fft import fft_convolve, fft_convolve_adjoint, prepare_psf
from src.utils.padding import center_crop, center_pad, padded_shape
from src.utils.tv import (
    circular_grad_ffts,
    grad_x,
    grad_x_adjoint,
    grad_y,
    grad_y_adjoint,
    soft_threshold,
    tv_squared_adjoint,
)


def inverse_softplus(x: float) -> float:
    value = torch.as_tensor(x, dtype=torch.float32)
    return torch.log(torch.expm1(value)).item()


class UnrolledADMM(nn.Module):

    def __init__(
        self,
        num_iters=20,
        init_mu=1e-4,
        init_tau=2e-4,
        eps=1e-6,
        cg_iters=10,
        clamp_output=False,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
        self.cg_iters = cg_iters
        self.clamp_output = clamp_output

        self.raw_mu = nn.Parameter(
            torch.full((num_iters,), inverse_softplus(init_mu))
        )
        self.raw_tau = nn.Parameter(
            torch.full((num_iters,), inverse_softplus(init_tau))
        )

    def forward(self, lensless: torch.Tensor, mask: torch.Tensor, **batch):
        h, w = lensless.shape[-2], lensless.shape[-1]
        out_shape = (h, w)
        pad_shape = padded_shape(h, w)

        y = center_pad(lensless, pad_shape)
        measurement_mask = center_pad(
            torch.ones(1, 1, h, w, device=y.device, dtype=y.dtype),
            pad_shape,
        )

        psf_fft = prepare_psf(mask, pad_shape).to(device=y.device)
        dx_fft, dy_fft = circular_grad_ffts(
            pad_shape,
            device=y.device,
            dtype=y.dtype,
        )

        x = torch.zeros_like(y)
        zx = torch.zeros_like(y)
        zy = torch.zeros_like(y)
        ux = torch.zeros_like(y)
        uy = torch.zeros_like(y)

        data_rhs = fft_convolve_adjoint(measurement_mask * y, psf_fft)

        for i in range(self.num_iters):
            mu = F.softplus(self.raw_mu[i]) + self.eps
            tau = F.softplus(self.raw_tau[i]) + self.eps
            tv_thresh = tau / mu

            rhs = data_rhs + mu * (
                grad_x_adjoint(zx - ux) + grad_y_adjoint(zy - uy)
            )
            x = self._solve_x(
                rhs,
                mu,
                psf_fft,
                measurement_mask,
                dx_fft,
                dy_fft,
                x0=x,
            )

            gx = grad_x(x)
            gy = grad_y(x)

            zx = soft_threshold(gx + ux, tv_thresh)
            zy = soft_threshold(gy + uy, tv_thresh)

            ux = ux + gx - zx
            uy = uy + gy - zy

        x = center_crop(x, out_shape)

        if self.clamp_output:
            x = x.clamp(0, 1)

        return {"recon": x}

    def _normal_op(self, x, mu, psf_fft, measurement_mask, dx_fft, dy_fft):
        hx = fft_convolve(x, psf_fft)
        data = fft_convolve_adjoint(measurement_mask * hx, psf_fft)
        tv = mu * tv_squared_adjoint(x, dx_fft, dy_fft)
        return data + tv + self.eps * x

    def _solve_x(self, rhs, mu, psf_fft, measurement_mask, dx_fft, dy_fft, x0=None):
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.clone()

        r = rhs - self._normal_op(x, mu, psf_fft, measurement_mask, dx_fft, dy_fft)
        p = r.clone()
        rs_old = self._dot(r, r)

        for _ in range(self.cg_iters):
            ap = self._normal_op(p, mu, psf_fft, measurement_mask, dx_fft, dy_fft)
            alpha = rs_old / (self._dot(p, ap) + self.eps)

            x = x + alpha * p
            r = r - alpha * ap

            rs_new = self._dot(r, r)

            if torch.sqrt(rs_new.max()).item() < 1e-7:
                break

            p = r + (rs_new / (rs_old + self.eps)) * p
            rs_old = rs_new

        return x

    @staticmethod
    def _dot(a, b):
        dims = tuple(range(1, a.ndim))
        return (a * b).sum(dim=dims, keepdim=True)
