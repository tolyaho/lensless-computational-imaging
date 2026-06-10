from torch import nn


class UNet(nn.Module):
    """small image-to-image cnn for pre/post processing."""

    def __init__(self, in_channels=3, out_channels=3, *args, **kwargs):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
