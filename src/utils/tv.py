import torch


def grad_x(x: torch.Tensor) -> torch.Tensor:
    """circular horizontal difference."""
    raise NotImplementedError


def grad_y(x: torch.Tensor) -> torch.Tensor:
    """circular vertical difference."""
    raise NotImplementedError


def grad_x_adjoint(v: torch.Tensor) -> torch.Tensor:
    """adjoint of grad_x."""
    raise NotImplementedError


def grad_y_adjoint(v: torch.Tensor) -> torch.Tensor:
    """adjoint of grad_y."""
    raise NotImplementedError


def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
    """small shrinkage step for tv variables."""
    raise NotImplementedError
