import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
    ):
        super().__init__()

        groups = min(groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv = ConvBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.down(x))


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DRUNetProcessor(nn.Module):
    """residual U-Net processor."""

    def __init__(
        self,
        channels: int = 3,
        features: tuple[int, int, int, int] = (32, 64, 128, 256),
        residual_scale: float = 0.1,
        clamp_output: bool = False,
    ):
        super().__init__()

        self.residual_scale = residual_scale
        self.clamp_output = clamp_output

        f1, f2, f3, f4 = features

        self.enc1 = ConvBlock(channels, f1)
        self.enc2 = DownsampleBlock(f1, f2)
        self.enc3 = DownsampleBlock(f2, f3)
        self.enc4 = DownsampleBlock(f3, f4)

        self.dec4 = UpsampleBlock(f4, f3, f3)
        self.dec3 = UpsampleBlock(f3, f2, f2)
        self.dec2 = UpsampleBlock(f2, f1, f1)

        self.out = nn.Conv2d(f1, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)

        correction = self.out(d2)
        # small residual step
        out = inp + correction * self.residual_scale

        if self.clamp_output:
            out = torch.clamp(out, min=0.0, max=1.0)

        return out
