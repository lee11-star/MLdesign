"""Generate white-vessel submission masks, inverted display masks and RLE CSV."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from train import set_seed
from vessel.data import image_ids, load_case
from vessel.model import UNet, predict_probability
from vessel.submission import make_submission_mask, write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('segmentation'))
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/final/model.pt'))
    parser.add_argument('--ensemble', type=Path, help='Selection JSON containing members and a validation-selected threshold')
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.ensemble:
        selection = json.loads(args.ensemble.read_text(encoding='utf-8'))
        member_specs = selection['members']
        threshold = selection['threshold']
    else:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        member_specs = [{'checkpoint': str(args.checkpoint), 'tta': checkpoint['config'].get('tta', 'flips')}]
        threshold = checkpoint['threshold']
    models = []
    for member in member_specs:
        checkpoint = torch.load(member['checkpoint'], map_location='cpu', weights_only=True)
        set_seed(checkpoint['config']['seed'])
        model = UNet(width=checkpoint['config']['width']).to(device)
        model.load_state_dict(checkpoint['model'])
        models.append((model, member['tta']))
    for folder in ['masks', 'masks_black_vessels', 'native_masks', 'probabilities']:
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    ids = image_ids(args.data / 'test/image')
    rows = []
    previews = []
    for case_id in ids:
        features, field, _ = load_case(args.data, 'test', case_id)
        probability = np.mean([predict_probability(model, features, device, tta=tta)
                               for model, tta in models], axis=0) * field
        np.save(args.output / 'probabilities' / f'{case_id}.npy', probability)
        native_mask = probability > threshold
        Image.fromarray(native_mask.astype(np.uint8) * 255).save(args.output / 'native_masks' / f'{case_id}.png')
        mask = make_submission_mask(native_mask).astype(np.uint8) * 255
        Image.fromarray(mask).save(args.output / 'masks' / f'{case_id}.png')
        Image.fromarray(255 - mask).save(args.output / 'masks_black_vessels' / f'{case_id}.png')
        rows.append({'Id': case_id, 'vessel_fraction': float((mask > 0).mean())})
        if len(previews) < 4:
            rgb = np.asarray(Image.open(args.data / f'test/image/{case_id}.jpg').convert('RGB').resize((512, 512)))
            overlay = rgb.copy()
            selected = mask > 0
            overlay[selected] = (0.4 * rgb[selected] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            previews.append((case_id, rgb, mask, overlay))
        print(f'Predicted {case_id}: vessel fraction={rows[-1]["vessel_fraction"]:.4f}', flush=True)
    csv_path = args.output / 'submission.csv'
    write_submission(args.output / 'masks', args.data / 'sample_submission.csv', csv_path)
    fig, axes = plt.subplots(len(previews), 3, figsize=(10, 3.5 * len(previews)))
    for row, (case_id, rgb, mask, overlay) in enumerate(previews):
        for col, (array, title) in enumerate([(rgb, 'Test image'), (mask, 'Predicted vessels'), (overlay, 'Overlay')]):
            axes[row, col].imshow(array, cmap='gray')
            axes[row, col].set_title(f'{case_id}: {title}')
            axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig(args.output / 'test_preview.png', dpi=110)
    plt.close(fig)
    manifest = {'members': member_specs, 'selection': str(args.ensemble) if args.ensemble else None,
                'threshold': threshold,
                'mask_size_hw': [512, 512], 'rle_order': 'C (row-major)', 'rle_start': 1,
                'foreground': 'white vessels in masks/; black vessels in masks_black_vessels/',
                'submission_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(), 'images': rows}
    (args.output / 'prediction_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved {csv_path} ({len(ids)} images)', flush=True)


if __name__ == '__main__':
    main()
