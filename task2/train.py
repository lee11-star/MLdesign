import argparse
import hashlib
import json
from pathlib import Path

import torch

from emotion_classifier.data import DEFAULT_DATA, read_manifest, load_arrays, check_split
from emotion_classifier.training import train_run

PRETRAINED = str(Path('../task1/outputs/torch_cache/hub/checkpoints/resnet18-f37072fd.pth').resolve())
CONVNEXT_PRETRAINED = str(Path('../task1/outputs/torch_cache/hub/checkpoints/convnext_tiny-983f1562.pth').resolve())
EFFICIENTNET_PRETRAINED = str(Path('../task1/outputs/torch_cache/hub/checkpoints/efficientnet_b2_rwightman-c35c1473.pth').resolve())
DINOV2_PRETRAINED = str(Path('../task1/outputs/torch_cache/hub/checkpoints/dinov2_vits14_pretrain.pth').resolve())
DAN_PRETRAINED = str(Path('outputs/round8/pretrained/dan_affectnet7.pth').resolve())


def run_training(args):
    torch.set_num_threads(args.threads)
    rows = read_manifest(args.manifest)
    check_split(rows)
    if args.resume:
        config = torch.load(args.resume, map_location='cpu', weights_only=False)['config']
        if config['data_signature'] != hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest():
            raise ValueError('Manifest differs from resumed checkpoint')
        final = config.get('final_training', False)
    else:
        final = args.full_data
        config = dict(model=args.model, epochs=args.epochs, scheduler_epochs=args.scheduler_epochs or args.epochs,
                      batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
                      device=args.device, num_workers=args.num_workers,
                      pretrained=(args.pretrained or {'resnet18': PRETRAINED, 'convnext_tiny': CONVNEXT_PRETRAINED,
                                                     'efficientnet_b2': EFFICIENTNET_PRETRAINED,
                                                     'dinov2_vits14': DINOV2_PRETRAINED,
                                                     'dan': DAN_PRETRAINED, 'dan_confidence': DAN_PRETRAINED}.get(args.model))
                      if args.model != 'cnn' else None,
                      label_smoothing=args.label_smoothing, image_size=args.image_size,
                      mixup_alpha=args.mixup_alpha, mixup_probability=args.mixup_probability,
                      head_dropout=args.head_dropout, backbone_lr_scale=args.backbone_lr_scale,
                      final_training=final, data_signature=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest())
        if args.geometry != 'standard':
            config['geometry'] = args.geometry
        if args.model == 'dan_confidence':
            config.update(confidence_fraction=args.confidence_fraction, confidence_margin=args.confidence_margin,
                          confidence_regularization=args.confidence_regularization)
        if args.model == 'dinov2_vits14':
            config.update(trainable_blocks=args.trainable_blocks, dinov2_repo=args.dinov2_repo)
        if args.initial_head:
            config.update(initial_head=str(Path(args.initial_head).resolve()),
                          initial_head_signature=hashlib.sha256(Path(args.initial_head).read_bytes()).hexdigest())
    x, y, _ = load_arrays(rows, args.data_root, 'all' if final else 'train', args.cache_dir)
    vx, vy = (None, None) if final else load_arrays(rows, args.data_root, 'val', args.cache_dir)[:2]
    return train_run(config, x, y, vx, vy, args.output, args.resume, args.stop_after)


def parser():
    p = argparse.ArgumentParser(description='Train or exactly resume a six-class emotion model')
    p.add_argument('--model', choices=['cnn', 'resnet18', 'convnext_tiny', 'efficientnet_b2', 'dinov2_vits14', 'dan', 'dan_confidence'], default='resnet18')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--scheduler-epochs', type=int)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--mixup-alpha', type=float, default=0.0, help='Beta concentration; zero disables MixUp')
    p.add_argument('--mixup-probability', type=float, default=0.5)
    p.add_argument('--head-dropout', type=float, default=0.0)
    p.add_argument('--backbone-lr-scale', type=float, default=1.0)
    p.add_argument('--trainable-blocks', type=int, default=4, help='DINOv2: unfreeze this many final transformer blocks')
    p.add_argument('--dinov2-repo', help='Existing local DINOv2 source directory')
    p.add_argument('--initial-head', help='Local linear-head state fitted on the training features')
    p.add_argument('--image-size', type=int, help='Defaults: CNN 48, DINOv2 224, other transfer models 96')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--geometry', choices=['standard', 'flip_only'], default='standard')
    p.add_argument('--confidence-fraction', type=float, default=0.7)
    p.add_argument('--confidence-margin', type=float, default=0.15)
    p.add_argument('--confidence-regularization', type=float, default=1.0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--pretrained', help='Local pretrained weights; defaults to the architecture-specific cache')
    p.add_argument('--data-root', default=str(DEFAULT_DATA))
    p.add_argument('--manifest', default='outputs/data/manifest.csv')
    p.add_argument('--cache-dir', default='outputs/data')
    p.add_argument('--output', required=True)
    p.add_argument('--full-data', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--stop-after', type=int, help='Stop after this total epoch while retaining full scheduler horizon')
    return p


if __name__ == '__main__':
    run_training(parser().parse_args())
