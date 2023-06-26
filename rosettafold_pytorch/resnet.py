import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ResBlock2D(nn.Module):
    def __init__(self, channel, kernel_size, dilation, p_dropout=0.15):
        super().__init__()
        self.layer = Residual(
            nn.Sequential(
                nn.Conv2d(
                    channel,
                    channel,
                    kernel_size,
                    dilation=dilation,
                    padding="same",
                    bias=False,
                ),
                nn.InstanceNorm2d(channel, affine=True, eps=1e-6),
                nn.ELU(),
                nn.Dropout(p_dropout),
                nn.Conv2d(
                    channel,
                    channel,
                    kernel_size,
                    dilation=dilation,
                    padding="same",
                    bias=False,
                ),
                nn.InstanceNorm2d(channel, affine=True, eps=1e-6),
            )
        )

    def forward(self, x):
        return F.elu(self.layer(x))


class ResNet(nn.Module):
    def __init__(
        self,
        n_res_blocks,
        in_channels,
        intermediate_channels,
        out_channels,
        dilations=[1, 2, 4, 8],
        p_dropout=0.15,
    ):
        super().__init__()

        layers = []
        # input projection
        layers += [
            nn.Conv2d(in_channels, intermediate_channels, 1, bias=False),
            nn.InstanceNorm2d(intermediate_channels, affine=True, eps=1e-6),
            nn.ELU(),
        ]
        # residual blocks
        for block_idx in range(n_res_blocks):
            dilation = dilations[block_idx % len(dilations)]
            layers.append(
                ResBlock2D(
                    intermediate_channels,
                    kernel_size=3,
                    dilation=dilation,
                    p_dropout=p_dropout,
                )
            )
        # output projection
        layers.append(nn.Conv2d(intermediate_channels, out_channels, 1))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
