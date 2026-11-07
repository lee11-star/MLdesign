"""DARK-inspired smoothed full-Hessian subpixel heatmap decoding."""
import math

import torch
from torch.nn import functional as F


def decode_dark(logits, images, region, sigma=.025):
    with torch.autocast(logits.device.type, enabled=False):
        logits = logits.float()
        batch, height, width = logits.shape
        deviation = sigma*(width-1)
        radius = math.ceil(3*deviation)
        axis = torch.arange(-radius, radius+1, device=logits.device, dtype=torch.float32)
        kernel = torch.exp(-axis.square()/(2*deviation**2))
        kernel = kernel/kernel.sum()
        probability = (logits-logits.flatten(1).amax(1)[:, None, None]).exp()[:, None]
        smoothed = F.conv2d(F.pad(probability, (radius, radius, 0, 0), mode='reflect'), kernel.view(1, 1, 1, -1))
        smoothed = F.conv2d(F.pad(smoothed, (0, 0, radius, radius), mode='reflect'), kernel.view(1, 1, -1, 1))
        field = smoothed[:, 0].clamp_min(1e-20).log()
        y, x = torch.meshgrid(torch.linspace(0, 1, height, device=logits.device),
                              torch.linspace(0, 1, width, device=logits.device), indexing='ij')
        support = ((x >= region[0]) & (x <= region[2]) & (y >= region[1]) & (y <= region[3]))
        brightness = F.interpolate(images.float().amax(1, keepdim=True), (height, width), mode='area')[:, 0]
        allowed = support[None] & (brightness > 8/255)
        peak = field.masked_fill(~allowed, -1e9).flatten(1).argmax(1)
        px, py = peak % width, peak // width
        xx, yy = px.clamp(1, width-2), py.clamp(1, height-2)
        i = torch.arange(batch, device=logits.device)
        center = field[i, yy, xx]
        left, right = field[i, yy, xx-1], field[i, yy, xx+1]
        up, down = field[i, yy-1, xx], field[i, yy+1, xx]
        gx, gy = (right-left)/2, (down-up)/2
        hxx, hyy = left+right-2*center, up+down-2*center
        hxy = (field[i, yy+1, xx+1]-field[i, yy+1, xx-1]-
               field[i, yy-1, xx+1]+field[i, yy-1, xx-1])/4
        determinant = hxx*hyy-hxy.square()
        valid = ((determinant > 1e-8) & (hxx < 0) & (hyy < 0) & (px == xx) & (py == yy))
        for sy in [-1, 0, 1]:
            for sx in [-1, 0, 1]:
                valid &= allowed[i, yy+sy, xx+sx]
        divisor = torch.where(valid, determinant, 1.)
        dx = torch.where(valid, -(hyy*gx-hxy*gy)/divisor, 0.).clamp(-1, 1)
        dy = torch.where(valid, -(hxx*gy-hxy*gx)/divisor, 0.).clamp(-1, 1)
        return torch.stack([(px+dx)/(width-1), (py+dy)/(height-1)], dim=1)
