"""Local fovea refinement with explicit full-image/crop coordinate transforms."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import normalize_xy, restore_xy, pixel_metrics, prediction_preview
from .model import FoveaNet, augment_batch, heatmap_loss
from .training import seed_everything, image_tensor, infer


def crop_spans(original_sizes, fraction):
    return original_sizes.min(1).values[:, None] * fraction / original_sizes


def restore_crop_xy(local_xy, centers, spans):
    return centers + (local_xy - .5) * spans


def crop_images(images, centers, spans, size):
    position = (torch.arange(size, device=images.device, dtype=torch.float32) + .5) / size * 2 - 1
    yy, xx = torch.meshgrid(position, position, indexing='ij')
    grid = torch.stack([xx, yy], -1)[None] * spans[:, None, None] + (centers[:, None, None] * 2 - 1)
    return F.grid_sample(images.float() / 255, grid, align_corners=False)


@torch.inference_mode()
def infer_refiner(model, images, centers, spans, config):
    coordinates = []
    for start in range(0, len(images), config['batch_size']):
        end = start + config['batch_size']
        crops = crop_images(images[start:end], centers[start:end], spans[start:end], config['image_size'])
        # Reuse the existing byte-image inference contract, including flip TTA.
        local = infer(model, (crops.clamp(0, 1) * 255).round().to(torch.uint8), config['batch_size'])
        local = torch.as_tensor(local, device=images.device)
        coordinates.append(restore_crop_xy(local, centers[start:end], spans[start:end]).cpu().numpy())
    return np.concatenate(coordinates)


def fit_refiner(frame, config, output, validation=None, coarse_xy=None, epochs=None):
    seed_everything(config['seed'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    device = torch.device(config['device'])
    model = FoveaNet(config['pretrained']).to(device)
    images = image_tensor(frame.path, config['source_size'], device)
    labels = torch.tensor(normalize_xy(frame[['Fovea_X', 'Fovea_Y']], frame[['width', 'height']]), dtype=torch.float32, device=device)
    spans = crop_spans(torch.tensor(frame[['width', 'height']].to_numpy(), dtype=torch.float32, device=device), config['crop_fraction'])
    if validation is not None:
        val_sizes = validation[['width', 'height']].to_numpy()
        val_actual = validation[['Fovea_X', 'Fovea_Y']].to_numpy()
        val_images = image_tensor(validation.path, config['source_size'], device)
        val_spans = crop_spans(torch.tensor(val_sizes, dtype=torch.float32, device=device), config['crop_fraction'])
        centers = torch.tensor(normalize_xy(coarse_xy, val_sizes), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['lr'] * .15},
        {'params': list(model.lateral.parameters()) + list(model.head.parameters()), 'lr': config['lr']}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['lr'] * .005)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    history, best, best_epoch = [], float('inf'), 0
    for epoch in range(1, (epochs or config['epochs']) + 1):
        started = time.monotonic()
        model.train()
        losses = []
        for indices in torch.randperm(len(frame), device=device).split(config['batch_size']):
            # Jitter uses training labels only. Validation crops always use model predictions.
            center = labels[indices] + (torch.rand(len(indices), 2, device=device) - .5) * .6 * spans[indices]
            cropped = crop_images(images[indices], center, spans[indices], config['image_size'])
            local_xy = .5 + (labels[indices] - center) / spans[indices]
            cropped, local_xy = augment_batch(cropped, local_xy)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                loss = heatmap_loss(model(cropped), local_xy, sigma=config['sigma'])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {'epoch': epoch, 'loss': float(np.mean(losses))}
        if validation is not None:
            predicted = restore_xy(infer_refiner(model, val_images, centers, val_spans, config), val_sizes)
            # Fixed 50/50 coarse/local shrinkage reduces large local errors.
            combined = .5 * coarse_xy + .5 * predicted
            metrics = pixel_metrics(val_actual, combined)
            row.update(metrics)
            row['local_rmse'] = pixel_metrics(val_actual, predicted)['rmse']
            if metrics['rmse'] < best:
                best, best_epoch = metrics['rmse'], epoch
                torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epoch,
                            'metrics': metrics, 'training_ids': frame.id.tolist(),
                            'validation_ids': validation.id.tolist()}, output / 'best.pt')
                predictions = validation[['id', 'Fovea_X', 'Fovea_Y', 'width', 'height']].copy()
                predictions[['coarse_x', 'coarse_y']] = coarse_xy
                predictions[['local_x', 'local_y']] = predicted
                predictions[['pred_x', 'pred_y']] = combined
                predictions.to_csv(output / 'predictions.csv', index=False)
        row['seconds'] = round(time.monotonic() - started, 2)
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        if epoch == 1 or epoch % 5 == 0:
            text = f'{output.name}: epoch {epoch}/{epochs or config["epochs"]} loss={row["loss"]:.4f}'
            if validation is not None:
                text += f' combined_RMSE={row["rmse"]:.2f}px local_RMSE={row["local_rmse"]:.2f}px best={best:.2f}@{best_epoch}'
            print(text + f' ({row["seconds"]:.1f}s)', flush=True)
    if validation is None:
        torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epochs or config['epochs'],
                    'training_ids': frame.id.tolist()}, output / 'final.pt')
        return {'epochs': epochs or config['epochs'], 'checkpoint': str(output / 'final.pt')}
    predictions = pd.read_csv(output / 'predictions.csv')
    prediction_preview(validation, predictions[['pred_x', 'pred_y']].to_numpy(), output / 'predictions.jpg')
    return {'best_epoch': best_epoch, 'metrics': pixel_metrics(val_actual, predictions[['pred_x', 'pred_y']]),
            'checkpoint': str(output / 'best.pt'), 'config': config}
