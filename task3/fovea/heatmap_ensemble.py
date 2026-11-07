"""Consensus of co-registered heatmaps, preserving original-pixel coordinates."""
import numpy as np
import torch
from torch.nn import functional as F

from .data import restore_xy
from .metric_objective import expectation_xy
from .training import image_tensor, seed_everything


def combine_logits(logits, weights, size=128):
    maps = [F.interpolate(value.float()[:, None], (size, size), mode='bilinear', align_corners=True)[:, 0]
            for value in logits]
    return sum(weight*value for weight, value in zip(weights, maps))


@torch.inference_mode()
def infer_heatmap_views(frame, bundle, device='cuda'):
    from .optimized import load_model
    seed_everything(42)
    device = torch.device(device)
    models, images = [], {}
    items = bundle['coarse']
    region = items[0]['region']
    for item in items:
        model, saved = load_model(item['checkpoint'], device)
        assert not saved['config'].get('preprocessing')
        assert item['region'] == region
        if bundle['split'] == 'validation':
            assert set(frame.id).isdisjoint(saved['training_ids'])
        if 'expected_training_ids' in bundle:
            assert set(saved['training_ids']) == set(bundle['expected_training_ids'])
        size = saved['config']['image_size']
        if size not in images:
            images[size] = image_tensor(frame.path, size, device)
        models.append((model, saved, images[size]))
    batch_size = models[0][1]['config']['batch_size']
    fused_predictions, coordinate_predictions = [], []
    weights = [item['weight'] for item in items]
    for start in range(0, len(frame), batch_size):
        fused_views, coordinate_views = [], []
        for flipped in [False, True]:
            logits, coordinates = [], []
            view_region = [1-region[2], region[1], 1-region[0], region[3]] if flipped else region
            reference = None
            for item, (model, saved, source) in zip(items, models):
                batch = source[start:start+batch_size].float()/255
                if flipped:
                    batch = batch.flip(-1)
                with torch.autocast(device.type, enabled=device.type == 'cuda'):
                    value = model(batch)
                logits.append(value)
                coordinates.append(expectation_xy(value, batch, view_region, temperature=item.get('temperature', .25)))
                reference = batch
            combined = combine_logits(logits, weights)
            xy = expectation_xy(combined, reference, view_region, temperature=.25)
            old_xy = sum(weight*value for weight, value in zip(weights, coordinates))
            if flipped:
                xy[:, 0] = 1-xy[:, 0]
                old_xy[:, 0] = 1-old_xy[:, 0]
            fused_views.append(xy)
            coordinate_views.append(old_xy)
        fused_predictions.append(torch.stack(fused_views).mean(0).cpu().numpy())
        coordinate_predictions.append(torch.stack(coordinate_views).mean(0).cpu().numpy())
    sizes = frame[['width', 'height']].to_numpy()
    return restore_xy(np.concatenate(fused_predictions), sizes), restore_xy(np.concatenate(coordinate_predictions), sizes)


def infer_fused(frame, bundle, device='cuda'):
    fused, coordinates = infer_heatmap_views(frame, bundle, device)
    weight = bundle.get('fusion_weight', 1.)
    return weight*fused+(1-weight)*coordinates
