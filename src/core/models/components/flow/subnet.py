"""Subnet factories for FrEIA affine coupling (used by FlowCLAS)."""

import torch
import torch.nn as nn

from .actnorm import ActNorm


def build_subnet(subnet_type: str, hidden_ratio: float = 1.0, **kwargs):
    """Return ``(in_channels, out_channels) -> nn.Module`` for the given ``subnet_type``."""
    if subnet_type == "simple":
        return lambda in_channels, out_channels: ConvNet2d(
            [in_channels] + [out_channels] * 2,
            [kwargs["kernel_size"]] * 2,
        )
    if subnet_type == "residual":
        return lambda in_channels, out_channels: ResidualSubnet(in_channels, out_channels, hidden_ratio=hidden_ratio, **kwargs)
    raise ValueError(f"Unknown subnet_type {subnet_type!r}. Expected 'simple' or 'residual'.")


class ResidualSubnet(nn.Module):
    """Lightweight pre-activation residual CNN; final layer init to zero (near-identity coupling)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        hidden_channels = int(in_channels * hidden_ratio)

        self.gn1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        self.gn2 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)

        self.gn3 = nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)

        if in_channels != hidden_channels:
            self.downsample = nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            )
        else:
            self.downsample = None

        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.conv_out.weight)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        residual = x
        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.gn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.conv_out(out)


class ConvNet2d(nn.Module):
    """Small ConvNet for coupling subnets; optional ActNorm + LeakyReLU stack."""

    def __init__(
        self,
        channels: list[int],
        kernel_size: list[int],
        leaky: float = 0.0,
        init_zeros: bool = True,
        actnorm: bool = False,
        weight_std: float | None = None,
    ):
        super().__init__()
        net: list[nn.Module] = []
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size[i],
                padding=kernel_size[i] // 2,
                bias=(not actnorm),
            )
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(ActNorm((channels[i + 1],) + (1, 1)))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(
            nn.Conv2d(
                channels[i - 1],
                channels[i],
                kernel_size[i - 1],
                padding=kernel_size[i - 1] // 2,
            )
        )
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return self.net(x)
