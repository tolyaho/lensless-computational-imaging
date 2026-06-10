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
    psf = _to_bchw(psf)
    _, _, h, w = psf.shape
    target_h, target_w = shape

    if h > target_h or w > target_w:
        raise ValueError(f"psf {(h, w)} is larger than target {shape}")

    if (h, w) != (target_h, target_w):
        psf = center_pad(psf, shape)

    # psf peak at the fft origin
    psf = torch.fft.ifftshift(psf, dim=(-2, -1))
    return torch.fft.fft2(psf, dim=(-2, -1))


def fft_convolve(x: torch.Tensor, psf_fft: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != psf_fft.shape[-2:]:
        raise ValueError(f"x shape {x.shape[-2:]} and psf shape {psf_fft.shape[-2:]} do not match")

    x_fft = torch.fft.fft2(x, dim=(-2, -1))
    y_fft = x_fft * psf_fft
    return torch.fft.ifft2(y_fft, dim=(-2, -1)).real


def fft_convolve_adjoint(z: torch.Tensor, psf_fft: torch.Tensor) -> torch.Tensor:
    if z.shape[-2:] != psf_fft.shape[-2:]:
        raise ValueError(f"z shape {z.shape[-2:]} and psf shape {psf_fft.shape[-2:]} do not match")

    z_fft = torch.fft.fft2(z, dim=(-2, -1))
    out_fft = z_fft * psf_fft.conj()
    return torch.fft.ifft2(out_fft, dim=(-2, -1)).real
