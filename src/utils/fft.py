import torch

from src.utils.padding import center_pad


def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected 2d, 3d, or 4d tensor, got shape {tuple(x.shape)}")
    return x


def prepare_psf(psf: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """psf to padded fft space."""
    psf = _to_bchw(psf)
    _, _, h, w = psf.shape
    target_h, target_w = shape

    if h > target_h or w > target_w:
        raise ValueError(f"psf {(h, w)} is larger than target {shape}")

    if (h, w) != (target_h, target_w):
        psf = center_pad(psf, shape)

    psf_sum = psf.sum(dim=(-2, -1), keepdim=True)
    psf = psf / psf_sum.clamp_min(1e-12)

    # center at fft origin
    psf = torch.fft.ifftshift(psf, dim=(-2, -1))
    return torch.fft.fft2(psf, dim=(-2, -1))


def _fft_shifted(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1))


def _ifft_unshifted(x_fft: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(x_fft, dim=(-2, -1)), dim=(-2, -1)).real


def fft_convolve(x: torch.Tensor, psf_fft: torch.Tensor) -> torch.Tensor:
    """circular convolution."""
    if x.shape[-2:] != psf_fft.shape[-2:]:
        raise ValueError(
            f"x shape {x.shape[-2:]} and psf shape {psf_fft.shape[-2:]} do not match"
        )

    return _ifft_unshifted(_fft_shifted(x) * psf_fft)


def fft_convolve_adjoint(z: torch.Tensor, psf_fft: torch.Tensor) -> torch.Tensor:
    """adjoint convolution."""
    if z.shape[-2:] != psf_fft.shape[-2:]:
        raise ValueError(
            f"z shape {z.shape[-2:]} and psf shape {psf_fft.shape[-2:]} do not match"
        )

    return _ifft_unshifted(_fft_shifted(z) * psf_fft.conj())
