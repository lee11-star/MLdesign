import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from emotion_classifier.data import DEFAULT_DATA, DEFAULT_TEMPLATE, load_arrays, read_manifest, save_json
from emotion_classifier.inference import load_checkpoint, predict_probabilities, write_submission


def run_prediction(checkpoint, output='submission.csv', template=DEFAULT_TEMPLATE,
                   data_root=DEFAULT_DATA, manifest='outputs/data/manifest.csv', cache_dir='outputs/data',
                   device='cuda', batch_size=128, tta=False, probabilities='outputs/test_probabilities.npz'):
    torch.set_num_threads(4)
    model, config, state = load_checkpoint(checkpoint, device)
    rows = read_manifest(manifest)
    signature = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if config['data_signature'] != signature:
        raise ValueError('Prediction manifest differs from training')
    x, _, ids = load_arrays(rows, data_root, 'test', cache_dir)
    probs = predict_probabilities(model, x, config['model'], device, batch_size, tta,
                                  image_size=config.get('preprocessing', {}).get('size'))
    Path(probabilities).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(probabilities, ids=np.asarray(ids), probabilities=probs,
                        class_names=np.asarray(list(config['class_to_label'])))
    result = write_submission(template, dict(zip(ids, probs.argmax(1))), output)
    result.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  checkpoint_epoch=state['epoch'], tta=tta, probabilities=str(probabilities),
                  submission_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest())
    save_json(result, 'outputs/submission_validation.json')
    print(result, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='submission.csv')
    p.add_argument('--template', default=str(DEFAULT_TEMPLATE))
    p.add_argument('--data-root', default=str(DEFAULT_DATA))
    p.add_argument('--manifest', default='outputs/data/manifest.csv')
    p.add_argument('--cache-dir', default='outputs/data')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--probabilities', default='outputs/test_probabilities.npz')
    run_prediction(**vars(p.parse_args()))
