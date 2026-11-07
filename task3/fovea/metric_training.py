import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import normalize_xy
from .distribution_loss import heatmap_js_loss
from .averaging import update_ema
from .metric_objective import expectation_xy, pixel_coordinate_mse
from .model import create_model, augment_batch, heatmap_loss
from .supported_decoder import training_region
from .training import image_tensor, seed_everything


def fit_metric(frame, warm_checkpoint, config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config['seed'])
    device = torch.device(config['device'])
    if config.get('initialization') == 'imagenet':
        model = create_model(config.get('architecture', 'resnet18'), warm_checkpoint).to(device)
        warm_epoch = 0
    else:
        warm = torch.load(warm_checkpoint, map_location='cpu', weights_only=False)
        assert set(warm['training_ids']) == set(frame.id) and warm['config']['image_size'] == 384
        model = create_model(config.get('architecture', 'resnet18')).to(device)
        model.load_state_dict(warm['state_dict'])
        warm_epoch = warm['epoch']
    boxes = frame[['crop_left', 'crop_top', 'crop_right', 'crop_bottom']].to_numpy() if config.get('preprocessing') == 'retinal_field' else None
    images = image_tensor(frame.path, config['image_size'], device, boxes)
    sizes = torch.tensor(frame[['width', 'height']].to_numpy(), dtype=torch.float32, device=device)
    labels = torch.tensor(normalize_xy(frame[['Fovea_X', 'Fovea_Y']], frame[['width', 'height']]), dtype=torch.float32, device=device)
    region = training_region(frame)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['backbone_lr']},
        {'params': list(model.lateral.parameters())+list(model.head.parameters()), 'lr': config['head_lr']}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.get('scheduler_epochs', config['epochs']),
                                                         eta_min=config.get('eta_min', 1e-6))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    averaged = copy.deepcopy(model).eval().requires_grad_(False) if 'ema_decay' in config else None
    ema_updates = 0
    snapshots = {}

    def checkpoint(network, epoch, averaging=None):
        value = {'state_dict': network.state_dict(), 'config': config, 'epoch': warm_epoch+epoch,
                 'warm_epoch': warm_epoch, 'fine_tune_epochs': epoch, 'region': region,
                 'training_ids': frame.id.tolist()}
        if averaging is not None:
            value['averaging'] = averaging
        return value

    history = []
    for epoch in range(1, config['epochs']+1):
        started = time.monotonic()
        model.train()
        sum_loss, sum_ce, sum_mse = 0., 0., 0.
        for index in torch.randperm(len(frame), device=device).split(config['batch_size']):
            batch, xy, transform = augment_batch(images[index].float()/255, labels[index], return_transform=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                logits = model(batch)
                coordinates = expectation_xy(logits, batch, region, transform, temperature=config.get('temperature', 1.))
                if config.get('distribution_loss') == 'js':
                    ce = heatmap_js_loss(logits, xy, sigma=config.get('sigma', .025),
                                         temperature=config.get('temperature', 1.))
                else:
                    ce = heatmap_loss(logits, xy, sigma=config.get('sigma', .025))
                metric_transform = transform if config.get('inverse_metric_transform', False) else None
                mse = pixel_coordinate_mse(coordinates, xy, sizes[index], transform=metric_transform)
                loss = ce + config['coordinate_weight'] * mse / config['coordinate_scale']**2
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            if averaged is not None and epoch >= config['ema_start_epoch']:
                if ema_updates == 0:
                    averaged.load_state_dict(model.state_dict())
                else:
                    update_ema(averaged, model, config['ema_decay'])
                ema_updates += 1
            sum_loss += float(loss.detach())*len(index)
            sum_ce += float(ce.detach())*len(index)
            sum_mse += float(mse.detach())*len(index)
        scheduler.step()
        row = {'epoch': epoch, 'loss': sum_loss/len(frame), 'heatmap_ce': sum_ce/len(frame),
               'pixel_mse': sum_mse/len(frame), 'pixel_rmse': np.sqrt(sum_mse/len(frame)),
               'seconds': round(time.monotonic()-started, 2)}
        if config.get('distribution_loss') == 'js':
            row['heatmap_js'] = row.pop('heatmap_ce')
        history.append(row)
        pd.DataFrame(history).to_csv(output/'history.csv', index=False)
        if epoch in config.get('snapshot_epochs', []):
            raw = output/('final.pt' if epoch == config['epochs'] else f'epoch{epoch}.pt')
            torch.save(checkpoint(model, epoch), raw)
            snapshots[str(epoch)] = {'raw': str(raw.resolve())}
            if averaged is not None:
                path = output/f'epoch{epoch}_ema.pt'
                torch.save(checkpoint(averaged, epoch, {'decay': config['ema_decay'],
                           'start_epoch': config['ema_start_epoch'], 'updates': ema_updates}), path)
                snapshots[str(epoch)]['ema'] = str(path.resolve())
        if epoch == 1 or epoch % 5 == 0:
            term = 'heatmap_js' if config.get('distribution_loss') == 'js' else 'heatmap_ce'
            print(f'{output.name}: {epoch}/{config["epochs"]} {term}={row[term]:.4f} pixel_RMSE={row["pixel_rmse"]:.3f} ({row["seconds"]:.1f}s)', flush=True)
    if str(config['epochs']) not in snapshots:
        torch.save(checkpoint(model, config['epochs']), output/'final.pt')
    result = {'checkpoint': str((output/'final.pt').resolve()), 'config': config, 'region': region,
              'warm_checkpoint': str(Path(warm_checkpoint).resolve()), 'warm_epoch': warm_epoch,
              'fine_tune_epochs': config['epochs'], 'training_ids': frame.id.tolist()}
    if snapshots:
        result['snapshots'] = snapshots
    (output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
