import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fovea.data import read_records, restore_xy, pixel_metrics, prediction_preview
from fovea.model import FoveaNet
from fovea.training import image_tensor, infer, seed_everything


def evaluate(checkpoint, data='detection', output='outputs/evaluation', device='cuda'):
    seed_everything(42)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'validation_ids' not in saved:
        raise ValueError('Full-data refits have no independent validation set. Use validation/best.pt.')
    frame = read_records(data).set_index('id', drop=False).loc[saved['validation_ids']].reset_index(drop=True)
    model = FoveaNet().to(device)
    model.load_state_dict(saved['state_dict'])
    images = image_tensor(frame.path, saved['config']['image_size'], torch.device(device))
    predicted = restore_xy(infer(model, images, saved['config']['batch_size']), frame[['width', 'height']])
    actual = frame[['Fovea_X', 'Fovea_Y']].to_numpy()
    report = pixel_metrics(actual, predicted)
    report['epoch'] = saved['epoch']
    report['images'] = len(frame)
    report['training_validation_ids_disjoint'] = set(saved['training_ids']).isdisjoint(saved['validation_ids'])
    assert report['training_validation_ids_disjoint']
    assert np.isclose(report['rmse'], saved['metrics']['rmse'], atol=.001)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame[['pred_x', 'pred_y']] = predicted
    frame['distance'] = np.linalg.norm(predicted - actual, axis=1)
    frame.to_csv(output / 'predictions.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    prediction_preview(frame, predicted, output / 'predictions.jpg')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Independently reload and evaluate the held-out validation checkpoint')
    parser.add_argument('--checkpoint', default='outputs/validation/best.pt')
    parser.add_argument('--data', default='detection')
    parser.add_argument('--output', default='outputs/evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data, args.output, args.device)
