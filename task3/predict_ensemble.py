import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fovea.data import read_records, write_submission, prediction_preview, pixel_metrics
from fovea.optimized import infer_bundle


def run_prediction(bundle_path, output, artifacts, data='detection', device='cuda'):
    bundle = json.loads(Path(bundle_path).read_text())
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    if bundle['split'] == 'validation':
        frame = read_records(data).set_index('id', drop=False).loc[bundle['validation_ids']].reset_index(drop=True)
    else:
        template = pd.read_csv(Path(data) / 'sample_submission.csv')
        ids = list(dict.fromkeys(int(name.split('_')[0]) for name in template.ImageID))
        frame = pd.DataFrame({'id': ids, 'path': [str(Path(data) / 'test' / f'{i:04d}.jpg') for i in ids]})
        frame[['width', 'height']] = [Image.open(p).size for p in frame.path]
    predicted = infer_bundle(frame, bundle, device)
    sizes = frame[['width', 'height']].to_numpy()
    assert np.isfinite(predicted).all() and ((predicted >= 0) & (predicted < sizes)).all()
    frame[['pred_x', 'pred_y']] = predicted
    frame.to_csv(artifacts / 'predictions.csv', index=False)
    prediction_preview(frame, predicted, artifacts / 'predictions.jpg', truth=bundle['split'] == 'validation')
    if bundle['split'] == 'validation':
        report = pixel_metrics(frame[['Fovea_X', 'Fovea_Y']].to_numpy(), predicted)
        if 'expected_rmse' in bundle:
            assert abs(report['rmse'] - bundle['expected_rmse']) < .002
    else:
        write_submission(Path(data) / 'sample_submission.csv', dict(zip(frame.id, predicted)), output)
        written = pd.read_csv(output)
        assert written.ImageID.tolist() == template.ImageID.tolist()
        assert written.columns.tolist() == ['ImageID', 'value']
        assert len(written) == 2 * len(frame) and np.isfinite(written.value).all()
        report = {'rows': len(written), 'images': len(frame), 'template_order_matches': True,
                  'all_finite': True, 'in_image_bounds': True,
                  'submission_sha256': hashlib.sha256(Path(output).read_bytes()).hexdigest()}
    report['bundle'] = str(Path(bundle_path).resolve())
    (artifacts / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reload the optimized coarse/local model bundle')
    parser.add_argument('--bundle', default='outputs/optimization/bundle_full.json')
    parser.add_argument('--output', default='submission_optimized.csv')
    parser.add_argument('--artifacts', default='outputs/optimization/prediction')
    parser.add_argument('--data', default='detection')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    run_prediction(args.bundle, args.output, args.artifacts, args.data, args.device)
