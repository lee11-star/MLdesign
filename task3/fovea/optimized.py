"""Inference for selected coarse and local models, using original image pixels."""
import numpy as np
import torch

from .data import normalize_xy, restore_xy
from .model import create_model
from .refinement import crop_spans, infer_refiner
from .training import image_tensor, infer, seed_everything
from .supported_decoder import infer_supported
from .metric_objective import infer_expectation
from .field_of_view import prepare_field_frame, restore_field_xy, BOX_COLUMNS


def load_model(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = create_model(saved['config'].get('architecture', 'resnet18')).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved


def infer_bundle(frame, bundle, device='cuda'):
    if bundle.get('fusion') == 'logits':
        from .heatmap_ensemble import infer_fused
        return infer_fused(frame, bundle, device)
    seed_everything(42)
    device = torch.device(device)
    sizes = frame[['width', 'height']].to_numpy()
    coarse_predictions = []
    for item in bundle['coarse']:
        model, saved = load_model(item['checkpoint'], device)
        if bundle['split'] == 'validation':
            assert set(frame.id).isdisjoint(saved['training_ids'])
        field = None
        if saved['config'].get('preprocessing') == 'retinal_field':
            field = prepare_field_frame(frame, saved['config']['field_threshold'], saved['config']['field_padding'])
            images = image_tensor(field.path, saved['config']['image_size'], device, field[BOX_COLUMNS].to_numpy())
        else:
            images = image_tensor(frame.path, saved['config']['image_size'], device)
        decoder = item.get('decoder', bundle.get('decoder'))
        if decoder == 'expectation':
            xy = infer_expectation(model, images, item['region'], saved['config']['batch_size'],
                                   temperature=item.get('temperature', saved['config'].get('temperature', 1.)),
                                   shifts=item.get('shifts'))
        elif decoder is not None:
            xy = infer_supported(model, images, item['region'], saved['config']['batch_size'], decoder)
        else:
            xy = infer(model, images, saved['config']['batch_size'])
        predicted = restore_xy(xy, sizes) if field is None else restore_field_xy(xy, field)
        coarse_predictions.append(predicted * item['weight'])
        del model, images
    coarse = np.sum(coarse_predictions, axis=0)
    if bundle.get('frozen_probe'):
        from .frozen_probe import cached_features, predict_probe
        probe = bundle['frozen_probe']
        features = cached_features(probe['features'], frame.id)
        with np.load(probe['head']) as head:
            predicted = predict_probe(head, features, sizes)
        coarse = (1-probe['weight'])*coarse+probe['weight']*predicted
    if not bundle.get('refiners'):
        return coarse
    refined_predictions = []
    for item in bundle['refiners']:
        model, saved = load_model(item['checkpoint'], device)
        if bundle['split'] == 'validation':
            assert set(frame.id).isdisjoint(saved['training_ids'])
        config = saved['config']
        images = image_tensor(frame.path, config['source_size'], device)
        spans = crop_spans(torch.tensor(sizes, dtype=torch.float32, device=device), config['crop_fraction'])
        centers = torch.tensor(normalize_xy(coarse, sizes), dtype=torch.float32, device=device)
        predicted = restore_xy(infer_refiner(model, images, centers, spans, config), sizes)
        refined_predictions.append(predicted * item['weight'])
        del model, images
    refined = np.sum(refined_predictions, axis=0)
    weight = bundle['refinement_weight']
    return (1 - weight) * coarse + weight * refined
