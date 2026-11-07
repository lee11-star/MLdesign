"""Local-contrast features computed from the same retinal RGB input."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import ConvNeXtFoveaNet


def retinal_contrast(images, sigma_fraction=1/30, gain=4.):
    # Normalized filtering prevents black borders pulling down the local mean.
    with torch.autocast(images.device.type, enabled=False):
        images = images.float()
        mask = (images.amax(1, keepdim=True) > 8/255).float()
        sigma = images.shape[-1]*sigma_fraction
        radius = math.ceil(3*sigma)
        grid = torch.arange(-radius, radius+1, device=images.device, dtype=torch.float32)
        kernel = torch.exp(-grid.square()/(2*sigma**2))
        kernel = kernel/kernel.sum()
        values = torch.cat([images*mask, mask], dim=1)
        values = F.conv2d(values, kernel.view(1, 1, 1, -1).expand(4, 1, 1, -1),
                          padding=(0, radius), groups=4)
        values = F.conv2d(values, kernel.view(1, 1, -1, 1).expand(4, 1, -1, 1),
                          padding=(radius, 0), groups=4)
        local = values[:, :3]/values[:, 3:].clamp_min(1e-6)
        return (gain*(images-local)+.5).clamp(0, 1)*mask


class ContrastConvNeXtFoveaNet(ConvNeXtFoveaNet):
    def __init__(self, pretrained=None):
        super().__init__(pretrained)
        original = self.encoder.features[0][0]
        expanded = nn.Conv2d(6, original.out_channels, original.kernel_size,
                             original.stride, original.padding, bias=original.bias is not None)
        with torch.no_grad():
            expanded.weight.copy_(original.weight.repeat(1, 2, 1, 1)/2)
            if original.bias is not None:
                expanded.bias.copy_(original.bias)
        self.encoder.features[0][0] = expanded
        self.mean = self.mean.repeat(1, 2, 1, 1)
        self.std = self.std.repeat(1, 2, 1, 1)

    def forward(self, images):
        return super().forward(torch.cat([images, retinal_contrast(images)], dim=1))
