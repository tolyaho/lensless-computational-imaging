from torch import nn


class ADMM(nn.Module):

    def __init__(self, num_iters=100, mu=1e-4, tau=2e-4, *args, **kwargs):
        super().__init__()
        self.num_iters = num_iters
        self.mu = mu
        self.tau = tau
        raise NotImplementedError

    def forward(self, lensless, mask, **batch):
        raise NotImplementedError
