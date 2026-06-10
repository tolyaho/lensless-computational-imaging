import torch
from torch import nn


class RandomScale1D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        scale = torch.randn(1)
        x = scale * x
        return x
