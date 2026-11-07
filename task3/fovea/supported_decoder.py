"""Training-derived support and subpixel decoding for the original 384 model."""
import numpy as np
import torch
from torch.nn import functional as F

from .data import normalize_xy
from .model import decode_heatmaps


def training_region(frame, padding=.08):
    xy = normalize_xy(frame[['Fovea_X', 'Fovea_Y']], frame[['width', 'height']])
    # Both eyes and horizontal TTA use the same symmetric horizontal support.
    left = max(0., min(xy[:, 0].min(), 1-xy[:, 0].max()) - padding)
    return [float(left), float(max(0., xy[:, 1].min() - padding)),
            float(1-left), float(min(1., xy[:, 1].max() + padding))]


def decode_supported(logits, images, region, method='quadratic'):
    if method == 'dark':
        from .dark import decode_dark
        return decode_dark(logits, images, region)
    logits = logits.float()
    batch, height, width = logits.shape
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=logits.device),
                            torch.linspace(0, 1, width, device=logits.device), indexing='ij')
    x0, y0, x1, y1 = region
    support = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    brightness = F.interpolate(images.amax(1, keepdim=True), (height, width), mode='area').squeeze(1)
    allowed = support[None] & (brightness > 8/255)
    if not allowed.flatten(1).any(1).all():
        raise ValueError('No non-black pixels inside the localization region')
    masked = logits.masked_fill(~allowed, -1e9)
    if method == 'legacy':
        return decode_heatmaps(masked)
    peak = masked.flatten(1).argmax(1)
    px, py = (peak % width).clamp(1, width-2), (peak // width).clamp(1, height-2)
    indices = torch.arange(batch, device=logits.device)
    center = logits[indices, py, px]
    left, right = logits[indices, py, px-1], logits[indices, py, px+1]
    up, down = logits[indices, py-1, px], logits[indices, py+1, px]
    dx = (.5 * (left-right) / (left+right-2*center).clamp(max=-1e-6)).clamp(-.75, .75)
    dy = (.5 * (up-down) / (up+down-2*center).clamp(max=-1e-6)).clamp(-.75, .75)
    # A parabola requires valid stencil neighbors on both sides of the peak.
    dx = torch.where(allowed[indices, py, px-1] & allowed[indices, py, px+1], dx, 0)
    dy = torch.where(allowed[indices, py-1, px] & allowed[indices, py+1, px], dy, 0)
    return torch.stack([(px+dx)/(width-1), (py+dy)/(height-1)], 1)


@torch.inference_mode()
def infer_supported(model, images, region, batch_size=8, method='quadratic'):
    model.eval()
    predictions = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start+batch_size].float() / 255
        with torch.autocast(images.device.type, enabled=images.device.type == 'cuda'):
            xy = decode_supported(model(batch), batch, region, method)
            mirrored = decode_supported(model(batch.flip(-1)), batch.flip(-1), region, method)
            mirrored[:, 0] = 1-mirrored[:, 0]
            xy = (xy+mirrored)/2
        predictions.append(xy.cpu().numpy())
    return np.concatenate(predictions)
