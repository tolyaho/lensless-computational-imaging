from torch import nn


class UnrolledADMM(nn.Module):
    """trainable leadmm with a fixed iteration count."""

    def __init__(self, num_iters=20, *args, **kwargs):
        super().__init__()
        self.num_iters = num_iters
        raise NotImplementedError

    def forward(self, lensless, mask, **batch):
        """unrolled admm; mu/tau stay positive via softplus or exp."""
        raise NotImplementedError
