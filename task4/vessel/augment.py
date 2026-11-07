import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def affine_batch(features, target, weights, output_size):
    """Transform a larger context crop into an aligned training patch."""
    batch, channels, height, width = features.shape
    angle = torch.rand(batch, device=features.device) * (2 * math.pi)
    scale = torch.empty(batch, device=features.device).uniform_(0.9, 1.1)
    factor = scale * (output_size / width)
    theta = torch.zeros(batch, 2, 3, device=features.device)
    cosine, sine = angle.cos() * factor, angle.sin() * factor
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    grid = F.affine_grid(theta, (batch, channels, output_size, output_size), align_corners=False)
    transformed = F.grid_sample(features, grid, mode='bilinear', align_corners=False)
    annotations = F.grid_sample(torch.cat([target, weights], dim=1), grid, mode='nearest', align_corners=False)
    return transformed, annotations[:, :1], annotations[:, 1:]
