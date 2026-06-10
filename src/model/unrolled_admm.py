from torch import nn


class UnrolledADMM(nn.Module):

    def __init__(self, num_iters=20, *args, **kwargs):
        super().__init__()
        self.num_iters = num_iters
        raise NotImplementedError

    def forward(self, lensless, mask, **batch):
        raise NotImplementedError
