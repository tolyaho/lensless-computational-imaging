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
