import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fovea.data import restore_xy, write_submission, prediction_preview
from fovea.model import FoveaNet
from fovea.training import image_tensor, infer, seed_everything


def predict(checkpoint, data='detection', output='submission.csv', device='cuda', artifacts='outputs'):
    seed_everything(42)
    data, artifacts = Path(data), Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = FoveaNet().to(device)
    model.load_state_dict(saved['state_dict'])
    template = pd.read_csv(data / 'sample_submission.csv')
    ids = list(dict.fromkeys(int(name.split('_')[0]) for name in template.ImageID))
    frame = pd.DataFrame({'id': ids, 'path': [str(data / 'test' / f'{i:04d}.jpg') for i in ids]})
    frame[['width', 'height']] = [Image.open(p).size for p in frame.path]
    images = image_tensor(frame.path, saved['config']['image_size'], torch.device(device))
    normalized = infer(model, images, saved['config']['batch_size'])
    sizes = frame[['width', 'height']].to_numpy()
    xy = np.clip(restore_xy(normalized, sizes), 0, sizes - 1)
    frame[['Fovea_X', 'Fovea_Y']] = xy
    frame.to_csv(artifacts / 'test_predictions.csv', index=False)
    submission = write_submission(data / 'sample_submission.csv', dict(zip(ids, xy)), output)
    # Read the actual on-disk CSV and verify exact template order and pixel bounds.
    written = pd.read_csv(output)
    assert written.columns.tolist() == ['ImageID', 'value']
    assert written.ImageID.tolist() == template.ImageID.tolist()
    assert len(written) == 2 * len(frame) and np.isfinite(written.value).all()
    assert ((xy >= 0) & (xy < sizes)).all()
    prediction_preview(frame, xy, artifacts / 'test_predictions.jpg', truth=False)
    report = {'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_epoch': saved['epoch'],
              'training_images': len(saved['training_ids']), 'test_images': len(frame),
              'rows': len(submission), 'columns': written.columns.tolist(),
              'template_order_matches': True, 'all_finite': True, 'in_image_bounds': True,
              'sha256': hashlib.sha256(Path(output).read_bytes()).hexdigest()}
    (artifacts / 'submission_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reload a fovea model and create Kaggle submission')
    parser.add_argument('--checkpoint', default='outputs/full/final.pt')
    parser.add_argument('--data', default='detection')
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--artifacts', default='outputs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    predict(args.checkpoint, args.data, args.output, args.device, args.artifacts)
