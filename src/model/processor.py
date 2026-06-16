import torch
from torch import nn


class ResidualCNNProcessor(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        hidden_channels: int = 32,
        num_layers: int = 4,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_scale = residual_scale

        layers = [
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]

        layers.append(
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.net(x)
