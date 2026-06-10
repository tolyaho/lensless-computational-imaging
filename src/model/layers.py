from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
