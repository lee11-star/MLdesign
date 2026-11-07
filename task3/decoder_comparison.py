"""Improve the original 384 model, export a new submission, preserve old files."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fovea.data import pixel_metrics, restore_xy
from fovea.optimized import load_model
from fovea.supported_decoder import training_region, infer_supported
from fovea.training import fit, image_tensor
from predict_ensemble import run_prediction


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/improved384'
SEEDS = [42, 2026, 7]
CANDIDATES = {
    'support_legacy': {'decoder': 'legacy', 'weights': {42: 1.}},
    'support_quadratic': {'decoder': 'quadratic', 'weights': {42: 1.}},
    'three_seed_mean': {'decoder': 'quadratic', 'weights': {42: 1/3, 2026: 1/3, 7: 1/3}},
    'original_half': {'decoder': 'quadratic', 'weights': {42: .5, 2026: .25, 7: .25}}}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def checkpoint_for(frame, config, folder, original=None):
    if original is not None:
        return str(original)
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['config'] == config and result['training_ids'] == frame.id.tolist()
        assert result['epochs'] == 61
    else:
        result = fit(frame, config, folder, epochs=61)
        result.update(config=config, training_ids=frame.id.tolist())
        dump(result_path, result)
    return result['checkpoint']


def evaluate_checkpoint(checkpoint, training_frame, val, folder, legacy=False):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'predictions.csv'
    region = training_region(training_frame)
    if target.exists():
        frame = pd.read_csv(target)
        manifest = json.loads((folder / 'evaluation.json').read_text())
        assert manifest['checkpoint'] == checkpoint and manifest['region'] == region
        assert manifest['training_ids'] == training_frame.id.tolist()
        assert frame.id.tolist() == val.id.tolist()
        return frame
    model, saved = load_model(checkpoint, 'cuda')
    assert set(saved['training_ids']) == set(training_frame.id)
    assert set(saved['training_ids']).isdisjoint(val.id)
    assert saved['epoch'] == 61 and saved['config']['image_size'] == 384
    images = image_tensor(val.path, 384, torch.device('cuda'))
    frame = val[['id', 'Fovea_X', 'Fovea_Y', 'width', 'height']].copy()
    quad = restore_xy(infer_supported(model, images, region), val[['width', 'height']])
    frame[['quadratic_x', 'quadratic_y']] = quad
    if legacy:
        xy = restore_xy(infer_supported(model, images, region, method='legacy'), val[['width', 'height']])
        frame[['legacy_x', 'legacy_y']] = xy
    frame.to_csv(target, index=False)
    dump(folder / 'evaluation.json', {'checkpoint': checkpoint, 'region': region,
                                     'training_ids': training_frame.id.tolist(), 'validation_ids': val.id.tolist()})
    return frame


def improve():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_hash = hashlib.sha256((ROOT / 'submission.csv').read_bytes()).hexdigest()
    original_summary = json.loads((ROOT / 'outputs/summary.json').read_text())
    config = original_summary['config']
    assert config['image_size'] == 384 and config.get('sigma', .025) == .025 and config['epochs'] == 80
    records = pd.read_csv(ROOT / 'outputs/records.csv')
    by_id = records.set_index('id', drop=False)
    splits = json.loads((ROOT / 'outputs/generalization_audit/splits.json').read_text())
    oof = records[['id', 'Fovea_X', 'Fovea_Y', 'width', 'height']].copy()
    for name in CANDIDATES:
        oof[[f'{name}_x', f'{name}_y']] = np.nan
    fold_metrics = []
    for split in splits:
        fold = split['fold']
        training_frame = by_id.loc[split['training_ids']].reset_index(drop=True)
        val = by_id.loc[split['validation_ids']].reset_index(drop=True)
        assert set(training_frame.group).isdisjoint(val.group)
        predictions = {}
        for seed in SEEDS:
            folder = OUTPUT / f'fold{fold}/seed{seed}'
            original = ROOT / f'outputs/generalization_audit/fold{fold}/original384/final.pt' if seed == 42 else None
            checkpoint = checkpoint_for(training_frame, {**config, 'seed': seed}, folder, original)
            frame = evaluate_checkpoint(checkpoint, training_frame, val, folder, legacy=seed == 42)
            predictions[seed] = frame[['quadratic_x', 'quadratic_y']].to_numpy()
            if seed == 42:
                legacy_xy = frame[['legacy_x', 'legacy_y']].to_numpy()
        ids = oof.index[oof.id.isin(val.id)]
        assert oof.loc[ids].id.tolist() == val.id.tolist()
        actual = val[['Fovea_X', 'Fovea_Y']].to_numpy()
        metrics = {}
        for name, choice in CANDIDATES.items():
            xy = legacy_xy if name == 'support_legacy' else sum(predictions[seed] * weight for seed, weight in choice['weights'].items())
            oof.loc[ids, [f'{name}_x', f'{name}_y']] = xy
            metrics[name] = pixel_metrics(actual, xy)
        fold_metrics.append({'fold': fold, 'metrics': metrics})
        dump(OUTPUT / 'fold_results.json', fold_metrics)
        oof.to_csv(OUTPUT / 'oof.csv', index=False)
        print(f'FOLD {fold+1}: '+', '.join(f'{name}={value["rmse"]:.3f}' for name, value in metrics.items()), flush=True)
    assert np.isfinite(oof.filter(regex='_x$|_y$').to_numpy()).all()
    actual = oof[['Fovea_X', 'Fovea_Y']].to_numpy()
    candidates = {name: pixel_metrics(actual, oof[[f'{name}_x', f'{name}_y']].to_numpy()) for name in CANDIDATES}
    selected_name = min(candidates, key=lambda name: candidates[name]['rmse'])
    selected = CANDIDATES[selected_name]
    old_oof = json.loads((ROOT / 'outputs/generalization_audit/summary.json').read_text())['original_oof']
    assert candidates[selected_name]['rmse'] < old_oof['rmse']
    # Record selection before any evaluation on the previously reused 12-image split.
    selection = {'name': selected_name, 'candidate_metrics': candidates, 'fold_metrics': fold_metrics,
                 'old_original_oof': old_oof, 'selected': selected}
    dump(OUTPUT / 'selection.json', selection)
    print(f'SELECTED {selected_name}: OOF RMSE={candidates[selected_name]["rmse"]:.4f}', flush=True)
    original_split = json.loads((ROOT / 'outputs/split.json').read_text())
    original_train = by_id.loc[original_split['train_ids']].reset_index(drop=True)
    val_bundle = {'split': 'validation', 'validation_ids': original_split['val_ids'],
                  'decoder': selected['decoder'], 'coarse': [], 'refiners': []}
    full_bundle = {'split': 'test', 'decoder': selected['decoder'], 'coarse': [], 'refiners': []}
    for seed, weight in selected['weights'].items():
        seed_config = {**config, 'seed': seed}
        val_checkpoint = checkpoint_for(original_train, seed_config, OUTPUT / f'validation/seed{seed}',
                                        ROOT / 'outputs/validation/best.pt' if seed == 42 else None)
        full_checkpoint = checkpoint_for(records, seed_config, OUTPUT / f'full/seed{seed}',
                                         ROOT / 'outputs/full/final.pt' if seed == 42 else None)
        val_bundle['coarse'].append({'checkpoint': val_checkpoint, 'weight': weight, 'region': training_region(original_train)})
        full_bundle['coarse'].append({'checkpoint': full_checkpoint, 'weight': weight, 'region': training_region(records)})
    dump(OUTPUT / 'bundle_validation.json', val_bundle)
    dump(OUTPUT / 'bundle_full.json', full_bundle)
    validation = run_prediction(OUTPUT / 'bundle_validation.json', None, OUTPUT / 'validation_prediction')
    test = run_prediction(OUTPUT / 'bundle_full.json', ROOT / 'submission_improved.csv', OUTPUT / 'prediction')
    old_test = pd.read_csv(ROOT / 'outputs/test_predictions.csv')
    new_test = pd.read_csv(OUTPUT / 'prediction/predictions.csv')
    assert old_test.id.tolist() == new_test.id.tolist()
    delta = new_test[['pred_x', 'pred_y']].to_numpy() - old_test[['Fovea_X', 'Fovea_Y']].to_numpy()
    old_raw = pd.read_csv(ROOT / 'outputs/generalization_audit/oof.csv')
    per_fold = []
    for row in fold_metrics:
        index = row['fold']
        mask = old_raw.fold == index
        baseline = pixel_metrics(old_raw.loc[mask, ['Fovea_X', 'Fovea_Y']].to_numpy(), old_raw.loc[mask, ['original_x', 'original_y']].to_numpy())
        per_fold.append({'fold': index, 'original': baseline, 'improved': row['metrics'][selected_name]})
    assert hashlib.sha256((ROOT / 'submission.csv').read_bytes()).hexdigest() == original_hash
    result = {**selection, 'per_fold_comparison': per_fold, 'validation12_original': original_summary['validation']['metrics'],
              'validation12_improved': validation, 'test_verification': test, 'old_submission_sha256': original_hash,
              'old_submission_unchanged': True, 'test_movement_mean': float(np.linalg.norm(delta, axis=1).mean()),
              'test_movement_max': float(np.linalg.norm(delta, axis=1).max()),
              'old_public_score': 12.66810, 'new_public_score': None,
              'validation_protocol': 'same 3 grouped folds, fixed 61 epochs/80-epoch schedule; support fitted on training labels only; selection uses pooled OOF before the 12-image check'}
    dump(OUTPUT / 'summary.json', result)
    print(json.dumps({'selection': selected_name, 'oof': candidates[selected_name], 'validation12': validation,
                      'per_fold_comparison': per_fold, 'test': test}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improve and ensemble the original 384 fovea model')
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    if args.predict_only:
        run_prediction(OUTPUT / 'bundle_full.json', ROOT / 'submission_improved.csv', OUTPUT / 'prediction')
    else:
        improve()
