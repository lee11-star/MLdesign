"""One geometric-augmentation ablation, replicated on a fixed extra group fold."""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from analysis import OUT, paired_metrics
from emotion_classifier.data import (CLASSES, DEFAULT_DATA, check_split, load_arrays,
                                     read_manifest, save_json, save_manifest)
from emotion_classifier.ensemble import align_probabilities, fuse_probabilities
from emotion_classifier.inference import load_checkpoint, predict_probabilities
from emotion_classifier.training import train_run
from evaluate import run_evaluation
from feature_transfer import old_validation
from baseline import ROOT, digest, read_json

BASE = ROOT / 'outputs/optimization/convnext_tiny'
ORIGINAL_MANIFEST = ROOT / 'outputs/data/manifest.csv'


def extra_manifest():
    rows = read_manifest(ORIGINAL_MANIFEST)
    eligible = [i for i, r in enumerate(rows) if r['split'] in ('train', 'val')]
    labels = np.asarray([rows[i]['label'] for i in eligible])
    groups = np.asarray([rows[i]['sha256'] for i in eligible])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
        np.zeros(len(labels)), labels, groups))
    original_val = {i for i, r in enumerate(rows) if r['split'] == 'val'}
    assert {eligible[i] for i in folds[0][1]} == original_val
    for i in eligible:
        rows[i]['split'] = 'train'
    for i in folds[1][1]:
        rows[eligible[i]]['split'] = 'val'
    check_split(rows)
    assert not original_val.intersection(i for i, r in enumerate(rows) if r['split'] == 'val')
    path = OUT / 'fold1_manifest.csv'
    save_manifest(rows, path)
    save_json(dict(splitter='StratifiedGroupKFold', n_splits=5, seed=42, fold_index=1,
                   manifest_sha256=digest(path), validation_disjoint_from_original=True,
                   train=sum(r['split'] == 'train' for r in rows),
                   val=sum(r['split'] == 'val' for r in rows),
                   train_val_hash_overlap=0), OUT / 'fold1_split.json')
    return path


def train_and_evaluate(name, manifest, geometry):
    folder = OUT / name
    config = read_json(BASE / 'config.json')
    config.pop('preprocessing')
    config.pop('class_to_label')
    config.update(geometry=geometry, data_signature=digest(manifest))
    last = folder / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        assert all(saved['config'].get(k) == v for k, v in config.items()), name
        complete = saved['epoch'] == config['epochs']
        del saved
    else:
        complete = False
    if not complete:
        rows = read_manifest(manifest)
        x, y, _ = load_arrays(rows, DEFAULT_DATA, 'train', ROOT / 'outputs/data')
        vx, vy, _ = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
        train_run(config, x, y, vx, vy, folder, resume=last if last.exists() else None)
        del x, y, vx, vy
        gc.collect()
        torch.cuda.empty_cache()
    metrics = run_evaluation(folder / 'best.pt', manifest=manifest, batch_size=128)
    # Primary endpoint is fixed at epoch 30 and flip TTA on BOTH folds.
    rows = read_manifest(manifest)
    vx, vy, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
    model, config, state = load_checkpoint(last, 'cuda')
    assert state['epoch'] == 30
    probs = predict_probabilities(model, vx, config['model'], 'cuda', 128, tta=True, image_size=96)
    np.savez_compressed(folder / 'epoch30_flip.npz', ids=np.asarray(ids), targets=vy,
                        probabilities=probs, class_names=np.asarray(CLASSES))
    del model, state, vx
    gc.collect()
    torch.cuda.empty_cache()
    return folder


def epoch30_baseline():
    path = OUT / 'fold0_standard_epoch30_flip.npz'
    if path.exists():
        return path
    rows = read_manifest(ORIGINAL_MANIFEST)
    vx, vy, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
    model, config, state = load_checkpoint(BASE / 'last.pt', 'cuda')
    assert state['epoch'] == 30 and config['data_signature'] == digest(ORIGINAL_MANIFEST)
    probs = predict_probabilities(model, vx, config['model'], 'cuda', 128, tta=True, image_size=96)
    np.savez_compressed(path, ids=np.asarray(ids), targets=vy, probabilities=probs, class_names=np.asarray(CLASSES))
    del model, state, vx
    gc.collect()
    torch.cuda.empty_cache()
    return path


def fold_comparison(manifest, baseline_path, candidate_path):
    rows = [r for r in read_manifest(manifest) if r['split'] == 'val']
    ids = [r['id'] for r in rows]
    labels = np.asarray([r['label'] for r in rows])
    arrays = []
    for path in (baseline_path, candidate_path):
        with np.load(path) as data:
            assert list(data['class_names']) == CLASSES
            arrays.append(align_probabilities(ids, data['ids'], data['probabilities'], labels, data['targets']))
    from model_selection import probability_metrics
    return dict(baseline=probability_metrics(arrays[0], labels),
                candidate=paired_metrics(*arrays, labels, [r['sha256'] for r in rows]),
                sources=[dict(path=str(p), sha256=digest(p)) for p in (baseline_path, candidate_path)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-round6', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    manifest = extra_manifest()
    if args.wait_round6:
        while read_json(ROOT / 'outputs/round6/status.json')['stage'] not in (
                'candidate_ready', 'complete', 'validation_complete', 'no_accuracy_improvement'):
            save_json(dict(stage='waiting_for_existing_round6_gpu_training'), OUT / 'status.json')
            time.sleep(20)
    originals = {p.name: digest(p) for p in ROOT.glob('submission*.csv')}
    save_json(originals, OUT / 'original_submissions.json')
    for name, fold_manifest, geometry in (
            ('fold0_flip_only', ORIGINAL_MANIFEST, 'flip_only'),
            ('fold1_standard', manifest, 'standard'),
            ('fold1_flip_only', manifest, 'flip_only')):
        save_json(dict(stage='training', run=name), OUT / 'status.json')
        train_and_evaluate(name, fold_manifest, geometry)
    baseline = epoch30_baseline()
    results = dict(primary_endpoint='epoch30_flip_tta',
                   fold0=fold_comparison(ORIGINAL_MANIFEST, baseline, OUT / 'fold0_flip_only/epoch30_flip.npz'),
                   fold1=fold_comparison(manifest, OUT / 'fold1_standard/epoch30_flip.npz', OUT / 'fold1_flip_only/epoch30_flip.npz'))
    # Fixed supplemental blend on the original fold; never tune this weight here.
    val = [r for r in read_manifest(ORIGINAL_MANIFEST) if r['split'] == 'val']
    labels, ids = np.asarray([r['label'] for r in val]), [r['id'] for r in val]
    old, _ = old_validation(ids, labels)
    with np.load(OUT / 'fold0_flip_only/epoch30_flip.npz') as data:
        new = align_probabilities(ids, data['ids'], data['probabilities'], labels, data['targets'])
    results['fixed_quarter_blend'] = paired_metrics(old, fuse_probabilities([old, new], [0.75, 0.25]), labels,
                                                   [r['sha256'] for r in val])
    results['consistent_positive'] = all(results[f]['candidate']['net_correct'] > 0 for f in ('fold0', 'fold1'))
    results['old_submissions_preserved'] = all(digest(ROOT / n) == sha for n, sha in originals.items())
    results['new_submission'] = False
    results['note'] = 'Additional group fold, same source dataset; no external validation. No full-data retraining or Kaggle upload.'
    save_json(results, OUT / 'geometry_comparison.json')
    save_json(dict(stage='experiments_complete', consistent_positive=results['consistent_positive']), OUT / 'status.json')
    print(results, flush=True)


if __name__ == '__main__':
    main()
