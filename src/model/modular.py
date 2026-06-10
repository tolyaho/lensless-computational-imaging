from torch import nn


class ModularLeADMM(nn.Module):

    def __init__(self, variant="prepost", num_iters=5, *args, **kwargs):
        super().__init__()
        self.variant = variant
        self.num_iters = num_iters
        raise NotImplementedError

    def forward(self, lensless, mask, **batch):
        raise NotImplementedError
