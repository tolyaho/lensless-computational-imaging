import torch


def prepare_psf(psf: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """move the psf into fft space."""
    raise NotImplementedError


def fft_convolve(x: torch.Tensor, psf_fft: torch.Tensor) -> torch.Tensor:
    """apply circular convolution in padded space.

    expects x as [B, C, H, W] and psf_fft already prepared for the same spatial size.
    """
    raise NotImplementedError


def fft_convolve_adjoint(z: torch.Tensor, psf_fft: torch.Tensor) -> torch.Tensor:
    """adjoint convolution via the conjugate spectrum."""
    raise NotImplementedError
