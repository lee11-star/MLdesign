"""Train a small U-Net; no downloaded weights or additional training data."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from vessel.data import PatchDataset, image_ids, load_case, split_ids
from vessel.augment import affine_batch
from vessel.model import UNet, predict_probability
from vessel.submission import make_submission_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def dice_score(prediction, target):
    return float((2 * np.logical_and(prediction, target).sum() + 1e-7) /
                 (prediction.sum() + target.sum() + 1e-7))


def score_probabilities(probabilities, native_labels, val_ids):
    labels = [make_submission_mask(label > 0) for label in native_labels]
    candidates = []
    for threshold in np.arange(0.2, 0.751, 0.025):
        scores = [dice_score(make_submission_mask(p > threshold), y) for p, y in zip(probabilities, labels)]
        candidates.append((float(np.mean(scores)), float(threshold), scores))
    mean_dice, threshold, scores = max(candidates, key=lambda item: item[0])
    masks = [make_submission_mask(p > threshold) for p in probabilities]
    pooled = dice_score(np.stack(masks), np.stack(labels))
    native_scores = [dice_score(p > threshold, label > 0) for p, label in zip(probabilities, native_labels)]
    return {'mean_dice': mean_dice, 'pooled_dice': pooled, 'threshold': threshold,
            'per_image_dice': dict(zip(val_ids, scores)), 'native_mean_dice': float(np.mean(native_scores)),
            'native_per_image_dice': dict(zip(val_ids, native_scores)),
            'evaluation_mask_conversion': 'native binary -> PIL bilinear 512x512 -> nonzero, for prediction and reference'}, probabilities, labels


def evaluate(model, val_cases, val_ids, device, tta='flips'):
    probabilities = [predict_probability(model, features, device, tta=tta) * field
                     for features, field, _ in val_cases]
    return score_probabilities(probabilities, [case[2] for case in val_cases], val_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('segmentation'))
    parser.add_argument('--output', type=Path, default=Path('outputs/validation'))
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--steps-per-epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--tta', choices=['flips', 'd4'], default='flips')
    parser.add_argument('--thin-weight', type=float, default=0)
    parser.add_argument('--affine-augment', action='store_true')
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--full-data-init-checkpoint', type=Path,
                        help='Corresponding full-data starting weights when reproducing a fine-tuning selection')
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--all-data', action='store_true')
    parser.add_argument('--selection', type=Path, default=Path('outputs/validation/summary.json'))
    args = parser.parse_args()
    if args.all_data:
        selection = json.loads(args.selection.read_text(encoding='utf-8'))
        # Keep the learning-rate schedule identical to validation; stop at the selected epoch.
        selected_epoch = selection['best_epoch']
        for name in ['epochs', 'steps_per_epoch', 'batch_size', 'patch_size', 'width', 'lr', 'seed']:
            setattr(args, name, selection['config'][name])
        args.split_seed = selection['config'].get('split_seed', selection['config']['seed'])
        args.tta = selection['config'].get('tta', 'flips')
        args.thin_weight = selection['config'].get('thin_weight', 0)
        args.affine_augment = selection['config'].get('affine_augment', False)
        full_data_init = selection['config'].get('full_data_init_checkpoint')
        args.init_checkpoint = Path(full_data_init) if full_data_init else None
        args.full_data_init_checkpoint = args.init_checkpoint
    else:
        selected_epoch = args.epochs
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    ids = image_ids(args.data / 'train/image')
    train_ids, val_ids = (ids, []) if args.all_data else split_ids(ids, args.split_seed)
    train_cases = [load_case(args.data, 'train', case_id) for case_id in train_ids]
    val_cases = [load_case(args.data, 'train', case_id) for case_id in val_ids]
    context_size = int(np.ceil(args.patch_size * 1.7 / 8)) * 8 if args.affine_augment else args.patch_size
    dataset = PatchDataset(train_cases, args.steps_per_epoch * args.batch_size, context_size, args.thin_weight)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda')
    model = UNet(width=args.width).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(initial['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                  parameters=sum(p.numel() for p in model.parameters()),
                  train_ids=train_ids, validation_ids=val_ids, training_epochs=selected_epoch)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2), flush=True)
    start = time.perf_counter()
    best_dice = -1
    summary = None
    with (args.output / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_loss', 'lr', 'val_dice', 'threshold', 'elapsed_seconds'])
        writer.writeheader()
        for epoch in range(1, selected_epoch + 1):
            model.train()
            total_loss = 0
            for features, target, weights in loader:
                features = features.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                if args.affine_augment:
                    features, target, weights = affine_batch(features, target, weights, args.patch_size)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits = model(features)
                    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights if args.thin_weight else None)
                    probability = logits.sigmoid().float()
                    intersection = (probability * target).sum(dim=(1, 2, 3))
                    denominator = (probability + target).sum(dim=(1, 2, 3))
                    soft_dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
                    loss = bce + soft_dice_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            row = {'epoch': epoch, 'train_loss': total_loss / len(loader), 'lr': current_lr,
                   'elapsed_seconds': round(time.perf_counter() - start, 2)}
            if val_cases and (epoch == 1 or epoch % args.eval_every == 0 or epoch == selected_epoch):
                metrics, probabilities, labels = evaluate(model, val_cases, val_ids, device, tta=args.tta)
                row.update(val_dice=metrics['mean_dice'], threshold=metrics['threshold'])
                if metrics['mean_dice'] > best_dice:
                    best_dice = metrics['mean_dice']
                    summary = dict(metrics, best_epoch=epoch, config=config,
                                   note='Validation images select the checkpoint and threshold; not a blind test score.')
                    torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch,
                                'threshold': metrics['threshold']}, args.output / 'best.pt')
                    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
                    np.savez_compressed(args.output / 'validation_predictions.npz',
                                        ids=np.array(val_ids), probabilities=np.stack(probabilities), labels=np.stack(labels),
                                        masks=np.stack([make_submission_mask(p > metrics['threshold']) for p in probabilities]))
            writer.writerow(row)
            handle.flush()
            if epoch == 1 or epoch % 5 == 0 or epoch == selected_epoch:
                print(json.dumps(row), flush=True)
    if args.all_data:
        checkpoint = {'model': model.state_dict(), 'config': config, 'epoch': selected_epoch,
                      'threshold': selection['threshold'], 'selection': str(args.selection)}
        torch.save(checkpoint, args.output / 'model.pt')
        summary = {'epoch': selected_epoch, 'threshold': selection['threshold'], 'config': config,
                   'elapsed_seconds': time.perf_counter() - start, 'validation_metrics_source': str(args.selection)}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Finished:', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
