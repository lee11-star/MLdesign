import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fovea.data import audit_data, make_split, normalize_xy, restore_xy, pixel_metrics
from fovea.training import fit


def default_weights():
    return str(Path(__file__).resolve().parent.parent / 'task1/outputs/torch_cache/hub/checkpoints/resnet18-f37072fd.pth')


def train_pipeline(args):
    output = Path(args.output)
    frame, audit = audit_data(args.data, output)
    train_indices, val_indices = make_split(frame, args.seed)
    train_frame, val_frame = frame.iloc[train_indices], frame.iloc[val_indices]
    assert set(train_frame.group).isdisjoint(val_frame.group)
    split = {'seed': args.seed, 'train_ids': train_frame.id.tolist(), 'val_ids': val_frame.id.tolist(),
             'grouping': 'train-only 63-bit DCT perceptual hash, Hamming distance <=6, connected components'}
    (output / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    train_sizes = train_frame[['width', 'height']].to_numpy()
    normalized = normalize_xy(train_frame[['Fovea_X', 'Fovea_Y']], train_sizes)
    baseline = np.stack([normalized[train_frame.width.to_numpy() == width].mean(0) for width in val_frame.width])
    baseline_metrics = pixel_metrics(val_frame[['Fovea_X', 'Fovea_Y']].to_numpy(), restore_xy(baseline, val_frame[['width', 'height']]))
    config = {'image_size': args.image_size, 'batch_size': args.batch_size, 'epochs': args.epochs,
              'lr': args.lr, 'seed': args.seed, 'pretrained': str(Path(args.pretrained).resolve()),
              'device': args.device, 'sigma': args.sigma}
    if not Path(config['pretrained']).is_file():
        raise FileNotFoundError(f'Provide ImageNet ResNet18 weights with --pretrained: {config["pretrained"]}')
    print(f'Device: {args.device}; train={len(train_frame)}, val={len(val_frame)}; mean baseline RMSE={baseline_metrics["rmse"]:.2f}px', flush=True)
    validation = fit(train_frame, config, output / 'validation', validation=val_frame)
    summary = {'config': config, 'audit': audit, 'split': split, 'baseline': baseline_metrics, 'validation': validation}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not args.validation_only:
        summary['full'] = fit(frame, config, output / 'full', epochs=validation['best_epoch'])
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def parser():
    result = argparse.ArgumentParser(description='Train and validate a pretrained ResNet18 fovea heatmap network')
    result.add_argument('--data', default='detection')
    result.add_argument('--output', default='outputs')
    result.add_argument('--pretrained', default=default_weights())
    result.add_argument('--image-size', type=int, default=384)
    result.add_argument('--epochs', type=int, default=80)
    result.add_argument('--batch-size', type=int, default=8)
    result.add_argument('--lr', type=float, default=.001)
    result.add_argument('--sigma', type=float, default=.025, help='Gaussian target width in normalized image coordinates')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    result.add_argument('--validation-only', action='store_true')
    return result


if __name__ == '__main__':
    train_pipeline(parser().parse_args())
