import torch


def grad_x(x: torch.Tensor) -> torch.Tensor:
    return torch.roll(x, shifts=-1, dims=-1) - x


def grad_y(x: torch.Tensor) -> torch.Tensor:
    return torch.roll(x, shifts=-1, dims=-2) - x


def grad_x_adjoint(v: torch.Tensor) -> torch.Tensor:
    return torch.roll(v, shifts=1, dims=-1) - v


def grad_y_adjoint(v: torch.Tensor) -> torch.Tensor:
    return torch.roll(v, shifts=1, dims=-2) - v


def soft_threshold(x: torch.Tensor, lam: float | torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.relu(x.abs() - lam)


def tv_squared_adjoint(
    x: torch.Tensor,
    dx_fft: torch.Tensor,
    dy_fft: torch.Tensor,
) -> torch.Tensor:
    """circular tv normal op."""
    x_fft = torch.fft.fft2(x, dim=(-2, -1))
    out_fft = (dx_fft.abs().square() + dy_fft.abs().square()) * x_fft
    return torch.fft.ifft2(out_fft, dim=(-2, -1)).real


def circular_grad_ffts(
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """circular gradient kernels."""
    h, w = shape

    kx = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
    ky = torch.zeros(1, 1, h, w, device=device, dtype=dtype)

    kx[..., 0, 0] = -1
    kx[..., 0, 1] = 1

    ky[..., 0, 0] = -1
    ky[..., 1, 0] = 1

    dx_fft = torch.fft.fft2(kx, dim=(-2, -1))
    dy_fft = torch.fft.fft2(ky, dim=(-2, -1))

    return dx_fft, dy_fft
