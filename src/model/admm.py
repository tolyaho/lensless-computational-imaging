import torch
from torch import nn

from src.utils.fft import fft_convolve, fft_convolve_adjoint, prepare_psf
from src.utils.padding import center_crop, center_pad, padded_shape
from src.utils.tv import (
    grad_x,
    grad_x_adjoint,
    grad_y,
    grad_y_adjoint,
    soft_threshold,
)


class ADMM(nn.Module):

    def __init__(
        self,
        num_iters=100,
        mu=1e-4,
        tau=2e-4,
        eps=1e-6,
        cg_iters=10,
        clamp_output=True,
    ):
        super().__init__()
        self.num_iters = num_iters
        self.mu = mu
        self.tau = tau
        self.eps = eps
        self.cg_iters = cg_iters
        self.clamp_output = clamp_output

    def forward(self, lensless: torch.Tensor, mask: torch.Tensor, **batch):
        h, w = lensless.shape[-2], lensless.shape[-1]
        pad_shape = padded_shape(h, w)

        y = center_pad(lensless, pad_shape)
        measurement_mask = center_pad(
            torch.ones(1, 1, h, w, device=y.device, dtype=y.dtype),
            pad_shape,
        )
        psf_fft = prepare_psf(mask, pad_shape).to(
            device=y.device, dtype=torch.complex64
        )
        dx_fft, dy_fft = self._grad_ffts(
            pad_shape,
            device=y.device,
            dtype=y.dtype,
        )

        x = torch.zeros_like(y)
        zx = torch.zeros_like(y)
        zy = torch.zeros_like(y)
        ux = torch.zeros_like(y)
        uy = torch.zeros_like(y)

        tv_thresh = self.tau / self.mu
        data_rhs = fft_convolve_adjoint(measurement_mask * y, psf_fft)

        for _ in range(self.num_iters):
            rhs = data_rhs + self.mu * (
                grad_x_adjoint(zx - ux) + grad_y_adjoint(zy - uy)
            )
            x = self._solve_x(
                rhs, psf_fft, measurement_mask, dx_fft, dy_fft, x0=x
            )

            gx = grad_x(x)
            gy = grad_y(x)

            zx = soft_threshold(gx + ux, tv_thresh)
            zy = soft_threshold(gy + uy, tv_thresh)

            ux = ux + gx - zx
            uy = uy + gy - zy

        x = center_crop(x, (h, w))

        if self.clamp_output:
            x = x.clamp(0, 1)

        return {"recon": x}

    def _normal_op(self, x, psf_fft, measurement_mask, dx_fft, dy_fft):
        hx = fft_convolve(x, psf_fft)
        data = fft_convolve_adjoint(measurement_mask * hx, psf_fft)
        tv = torch.fft.ifft2(
            self.mu
            * (dx_fft.abs().square() + dy_fft.abs().square())
            * torch.fft.fft2(x, dim=(-2, -1)),
            dim=(-2, -1),
        ).real
        return data + tv + self.eps * x

    def _solve_x(
        self, rhs, psf_fft, measurement_mask, dx_fft, dy_fft, x0=None
    ):
        if x0 is None:
            x = torch.zeros_like(rhs)
        else:
            x = x0.clone()

        r = rhs - self._normal_op(x, psf_fft, measurement_mask, dx_fft, dy_fft)
        p = r.clone()
        rs_old = self._dot(r, r)

        for _ in range(self.cg_iters):
            ap = self._normal_op(p, psf_fft, measurement_mask, dx_fft, dy_fft)
            alpha = rs_old / (self._dot(p, ap) + self.eps)

            x = x + alpha * p
            r = r - alpha * ap

            rs_new = self._dot(r, r)

            if torch.sqrt(rs_new.max()) < 1e-7:
                break

            p = r + (rs_new / (rs_old + self.eps)) * p
            rs_old = rs_new

        return x

    @staticmethod
    def _dot(a, b):
        dims = tuple(range(1, a.ndim))
        return (a * b).sum(dim=dims, keepdim=True)

    @staticmethod
    def _grad_ffts(shape, device, dtype):
        h, w = shape

        kx = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
        ky = torch.zeros(1, 1, h, w, device=device, dtype=dtype)

        kx[..., 0, 0] = -1
        kx[..., 0, -1] = 1

        ky[..., 0, 0] = -1
        ky[..., -1, 0] = 1

        dx_fft = torch.fft.fft2(kx, dim=(-2, -1))
        dy_fft = torch.fft.fft2(ky, dim=(-2, -1))

        return dx_fft, dy_fft
