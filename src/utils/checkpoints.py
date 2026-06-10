from pathlib import Path

import torch
from torch import nn


def save_checkpoint(path: Path, model: nn.Module, **extra):
    raise NotImplementedError


def load_checkpoint(path: Path, model: nn.Module, device: str = "cpu"):
    raise NotImplementedError
