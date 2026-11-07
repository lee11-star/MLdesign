import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_images, normalize_xy, restore_xy, pixel_metrics, prediction_preview
from .model import FoveaNet, augment_batch, heatmap_loss, decode_heatmaps


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def image_tensor(paths, size, device, boxes=None):
    return torch.from_numpy(load_images(paths, size, boxes)).permute(0, 3, 1, 2).to(device)


@torch.inference_mode()
def infer(model, images, batch_size=8, tta=True):
    model.eval()
    coordinates = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].float() / 255
        with torch.autocast(device_type=images.device.type, enabled=images.device.type == 'cuda'):
            xy = decode_heatmaps(model(batch))
            if tta:
                flipped = decode_heatmaps(model(batch.flip(-1)))
                flipped[:, 0] = 1 - flipped[:, 0]
                xy = (xy + flipped) / 2
        coordinates.append(xy.cpu().numpy())
    return np.concatenate(coordinates)


def fit(frame, config, output, validation=None, epochs=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config['seed'])
    device = torch.device(config['device'])
    model = FoveaNet(config['pretrained']).to(device)
    images = image_tensor(frame.path, config['image_size'], device)
    labels = torch.as_tensor(normalize_xy(frame[['Fovea_X', 'Fovea_Y']], frame[['width', 'height']]), dtype=torch.float32, device=device)
    if validation is not None:
        val_images = image_tensor(validation.path, config['image_size'], device)
        val_sizes = validation[['width', 'height']].to_numpy()
        val_xy = validation[['Fovea_X', 'Fovea_Y']].to_numpy()
    groups = [{'params': model.encoder.parameters(), 'lr': config['lr'] * .15},
              {'params': list(model.lateral.parameters()) + list(model.head.parameters()), 'lr': config['lr']}]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    # Same schedule in validation and full-data refit, stopped at the selected epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['lr'] * .005)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    history, best, best_epoch = [], float('inf'), 0
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for epoch in range(1, (epochs or config['epochs']) + 1):
        started = time.monotonic()
        model.train()
        permutation = torch.randperm(len(frame), device=device)
        losses = []
        for indices in permutation.split(config['batch_size']):
            batch = images[indices].float() / 255
            batch, xy = augment_batch(batch, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = heatmap_loss(model(batch), xy, sigma=config.get('sigma', .025))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {'epoch': epoch, 'loss': float(np.mean(losses))}
        if validation is not None:
            predicted = restore_xy(infer(model, val_images, config['batch_size']), val_sizes)
            metrics = pixel_metrics(val_xy, predicted)
            row.update(metrics)
            if metrics['rmse'] < best:
                best, best_epoch = metrics['rmse'], epoch
                checkpoint = {'state_dict': model.state_dict(), 'config': config, 'epoch': epoch,
                              'metrics': metrics, 'training_ids': frame.id.tolist(),
                              'validation_ids': validation.id.tolist()}
                torch.save(checkpoint, output / 'best.pt')
                predictions = validation[['id', 'Fovea_X', 'Fovea_Y', 'width', 'height']].copy()
                predictions[['pred_x', 'pred_y']] = predicted
                predictions['distance'] = np.linalg.norm(predicted - val_xy, axis=1)
                predictions.to_csv(output / 'predictions.csv', index=False)
        row['seconds'] = round(time.monotonic() - started, 2)
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        if epoch == 1 or epoch % 5 == 0:
            message = f'{output.name}: epoch {epoch}/{epochs or config["epochs"]} loss={row["loss"]:.4f}'
            if validation is not None:
                message += f' RMSE={row["rmse"]:.2f}px best={best:.2f}@{best_epoch}'
            print(message + f' ({row["seconds"]:.1f}s)', flush=True)
    if validation is None:
        torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epochs or config['epochs'],
                    'training_ids': frame.id.tolist()}, output / 'final.pt')
        return {'epochs': epochs or config['epochs'], 'checkpoint': str(output / 'final.pt')}
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    predicted = restore_xy(infer(model, val_images, config['batch_size']), val_sizes)
    prediction_preview(validation, predicted, output / 'predictions.jpg')
    return {'best_epoch': best_epoch, 'metrics': pixel_metrics(val_xy, predicted), 'checkpoint': str(output / 'best.pt')}
