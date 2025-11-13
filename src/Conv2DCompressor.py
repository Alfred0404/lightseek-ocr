"""
Conv2D Compressor
Compresses feature maps using strided convolution
"""

import torch
from torch import nn


class Conv2DCompressor(nn.Module):
    """Compress feature maps using strided convolution"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=16, stride=16):
        super(Conv2DCompressor, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        """
        Compress feature map

        Args:
            x: Input tensor (B, C_in, H, W)

        Returns:
            Compressed tensor (B, C_out, H_out, W_out)
        """
        return self.conv(x)
