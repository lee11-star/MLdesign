"""Evaluate a saved checkpoint with a specified inference augmentation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import evaluate, set_seed
from vessel.data import load_case
from vessel.model import UNet
from vessel.submission import make_submission_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tta', choices=['flips', 'd4'], default='d4')
    parser.add_argument('--data', type=Path, default=Path('segmentation'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config'], tta=args.tta)
    ids = config['validation_ids']
    if not ids:
        raise ValueError('Evaluation requires a validation checkpoint, not a full-data model.')
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(width=config['width']).to(device)
    model.load_state_dict(checkpoint['model'])
    cases = [load_case(args.data, 'train', case_id) for case_id in ids]
    metrics, probabilities, labels = evaluate(model, cases, ids, device, tta=args.tta)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(metrics, best_epoch=checkpoint['epoch'], config=config, checkpoint=str(args.checkpoint))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(args.output / 'validation_predictions.npz', ids=np.array(ids),
                        probabilities=np.stack(probabilities), labels=np.stack(labels),
                        masks=np.stack([make_submission_mask(p > metrics['threshold']) for p in probabilities]))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
