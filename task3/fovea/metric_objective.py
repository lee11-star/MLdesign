"""Differentiable image-pixel coordinate objective and consistent inference."""
import numpy as np
import torch
from torch.nn import functional as F


def pixel_coordinate_mse(prediction, target, original_sizes, transform=None):
    if transform is not None:
        with torch.autocast(prediction.device.type, enabled=False):
            inverse = torch.linalg.inv(transform[:, :, :2].float())
            error = torch.einsum('bij,bj->bi', inverse, prediction.float()-target.float())
            return ((error*original_sizes.float())**2).mean()
    return (((prediction - target) * original_sizes) ** 2).mean()


def expectation_xy(logits, images, region, transform=None, temperature=1.):
    with torch.autocast(logits.device.type, enabled=False):
        logits = logits.float()
        batch, height, width = logits.shape
        yy, xx = torch.meshgrid(torch.linspace(0, 1, height, device=logits.device),
                                torch.linspace(0, 1, width, device=logits.device), indexing='ij')
        grid = torch.stack([xx, yy], -1)
        original = grid[None].expand(batch, -1, -1, -1)
        if transform is not None:
            transform = transform.float()
            inverse = torch.linalg.inv(transform[:, :, :2])
            centered = original * 2-1-transform[:, None, None, :, 2]
            original = (torch.einsum('bij,bhwj->bhwi', inverse, centered) + 1)/2
        x0, y0, x1, y1 = region
        allowed = ((original[..., 0] >= x0) & (original[..., 0] <= x1) &
                   (original[..., 1] >= y0) & (original[..., 1] <= y1))
        brightness = F.interpolate(images.float().amax(1, keepdim=True), (height, width), mode='area').squeeze(1)
        allowed = allowed & (brightness > 8/255)
        if not allowed.flatten(1).any(1).all():
            raise ValueError('No valid pixels for coordinate expectation')
        probability = (logits / temperature).masked_fill(~allowed, -1e9).flatten(1).softmax(1)
        return probability @ grid.reshape(-1, 2)


@torch.inference_mode()
def infer_expectation(model, images, region, batch_size=8, temperature=1., shifts=None):
    model.eval()
    predictions = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start+batch_size].float()/255
        views = []
        for sx, sy in (shifts or [(0, 0)]):
            shifted = batch if sx == 0 and sy == 0 else torch.roll(batch, (sy, sx), dims=(-2, -1))
            if sy > 0:
                shifted[..., :sy, :] = 0
            elif sy < 0:
                shifted[..., sy:, :] = 0
            if sx > 0:
                shifted[..., :, :sx] = 0
            elif sx < 0:
                shifted[..., :, sx:] = 0
            dx, dy = sx/batch.shape[-1], sy/batch.shape[-2]
            translated_region = [region[0]+dx, region[1]+dy, region[2]+dx, region[3]+dy]
            flipped_region = [1-translated_region[2], translated_region[1], 1-translated_region[0], translated_region[3]]
            with torch.autocast(images.device.type, enabled=images.device.type == 'cuda'):
                xy = expectation_xy(model(shifted), shifted, translated_region, temperature=temperature)
                flipped = expectation_xy(model(shifted.flip(-1)), shifted.flip(-1), flipped_region, temperature=temperature)
                flipped[:, 0] = 1-flipped[:, 0]
                xy = (xy+flipped)/2
            views.append(xy-torch.tensor([dx, dy], device=xy.device))
        xy = torch.stack(views).mean(0)
        predictions.append(xy.cpu().numpy())
    return np.concatenate(predictions)
