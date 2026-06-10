from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError

    def forward(self, lensless, mask, **batch):
        raise NotImplementedError
