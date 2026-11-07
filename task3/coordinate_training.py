"""Controlled old-model fine-tuning with a pixel-coordinate MSE objective."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fovea.data import pixel_metrics, restore_xy
from fovea.metric_objective import infer_expectation
from fovea.metric_training import fit_metric
from fovea.optimized import load_model
from fovea.supported_decoder import training_region
from fovea.training import image_tensor
from predict_ensemble import run_prediction


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/metric_mse_joint'
CONFIG = {'image_size': 384, 'batch_size': 8, 'epochs': 61, 'scheduler_epochs': 80, 'seed': 42, 'device': 'cuda',
          'backbone_lr': .00015, 'head_lr': .001, 'eta_min': .000005,
          'coordinate_scale': 50., 'sigma': .025, 'temperature': .25, 'initialization': 'imagenet',
          'pretrained': str(ROOT.parent/'task1/outputs/torch_cache/hub/checkpoints/resnet18-f37072fd.pth')}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def fitted(frame, warm, config, folder):
    if config.get('initialization') == 'imagenet':
        warm = config['pretrained']
    path = folder/'result.json'
    if path.exists():
        result = json.loads(path.read_text())
        assert result['config'] == config and result['training_ids'] == frame.id.tolist()
        assert Path(result['warm_checkpoint']).resolve() == Path(warm).resolve()
        return result
    return fit_metric(frame, warm, config, folder)


def evaluate(frame, train, checkpoint, folder):
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'predictions.csv'
    if result_path.exists():
        saved = pd.read_csv(result_path)
        assert saved.id.tolist() == frame.id.tolist()
        return saved[['pred_x', 'pred_y']].to_numpy()
    model, metadata = load_model(checkpoint, 'cuda')
    assert set(metadata['training_ids']) == set(train.id)
    assert set(frame.id).isdisjoint(metadata['training_ids'])
    images = image_tensor(frame.path, metadata['config']['image_size'], torch.device('cuda'))
    predicted = restore_xy(infer_expectation(model, images, training_region(train),
                                            temperature=CONFIG.get('temperature', 1.)), frame[['width', 'height']])
    saved = frame[['id', 'Fovea_X', 'Fovea_Y', 'width', 'height']].copy()
    saved[['pred_x', 'pred_y']] = predicted
    saved.to_csv(result_path, index=False)
    dump(folder/'evaluation.json', {'checkpoint': str(checkpoint), 'training_ids': train.id.tolist(),
                                   'validation_ids': frame.id.tolist(), 'region': training_region(train)})
    return predicted


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in ['submission.csv', 'submission_optimized.csv', 'submission_improved.csv']}
    records = pd.read_csv(ROOT/'outputs/records.csv')
    by_id = records.set_index('id', drop=False)
    splits = json.loads((ROOT/'outputs/generalization_audit/splits.json').read_text())
    previous = pd.read_csv(ROOT/'outputs/improved384/oof.csv').set_index('id').loc[records.id]
    actual = records[['Fovea_X', 'Fovea_Y']].to_numpy()
    baseline = previous[['support_quadratic_x', 'support_quadratic_y']].to_numpy()
    oof = records[['id', 'Fovea_X', 'Fovea_Y']].copy()
    candidate_names = ['warm_expectation', 'control_ce', 'pixel_mse', 'pixel_mse_half_old']
    for name in candidate_names:
        oof[[f'{name}_x', f'{name}_y']] = np.nan
    folds = []
    for split in splits:
        fold = split['fold']
        train = by_id.loc[split['training_ids']].reset_index(drop=True)
        val = by_id.loc[split['validation_ids']].reset_index(drop=True)
        assert set(train.group).isdisjoint(val.group)
        indices = oof.index[oof.id.isin(val.id)]
        assert oof.loc[indices].id.tolist() == val.id.tolist()
        warm = ROOT/f'outputs/generalization_audit/fold{fold}/original384/final.pt'
        predictions = {'warm_expectation': evaluate(val, train, warm, OUTPUT/f'fold{fold}/warm_expectation')}
        for name, weight in [('control_ce', 0.), ('pixel_mse', 1.)]:
            config = {**CONFIG, 'coordinate_weight': weight}
            folder = OUTPUT/f'fold{fold}/{name}'
            print(f'FOLD {fold+1}/3 {name}', flush=True)
            if name == 'control_ce' and config.get('initialization') == 'imagenet':
                folder.mkdir(parents=True, exist_ok=True)
                result = {'checkpoint': str(warm), 'config': config, 'reused_exact_ce_control': True,
                          'training_ids': train.id.tolist(), 'epochs': 61, 'scheduler_epochs': 80}
                dump(folder/'result.json', result)
            else:
                result = fitted(train, warm, config, folder)
            predictions[name] = evaluate(val, train, result['checkpoint'], folder)
        predictions['pixel_mse_half_old'] = .5*predictions['pixel_mse']+.5*baseline[indices]
        metrics = {'old_quadratic': pixel_metrics(actual[indices], baseline[indices])}
        for name, xy in predictions.items():
            oof.loc[indices, [f'{name}_x', f'{name}_y']] = xy
            metrics[name] = pixel_metrics(actual[indices], xy)
        folds.append({'fold': fold, 'metrics': metrics})
        oof.to_csv(OUTPUT/'oof.csv', index=False)
        dump(OUTPUT/'fold_results.json', folds)
        print('FOLD RESULT '+str(fold+1)+': '+', '.join(f'{name}={value["rmse"]:.4f}' for name, value in metrics.items()), flush=True)
    assert np.isfinite(oof.filter(regex='_x$|_y$').to_numpy()).all()
    metrics = {'old_quadratic': pixel_metrics(actual, baseline)}
    for name in candidate_names:
        metrics[name] = pixel_metrics(actual, oof[[f'{name}_x', f'{name}_y']].to_numpy())
    selected_name = min(['pixel_mse', 'pixel_mse_half_old'], key=lambda name: metrics[name]['rmse'])
    selection = {'config': CONFIG, 'folds': folds, 'candidate_metrics': metrics, 'selected': selected_name}
    dump(OUTPUT/'selection.json', selection)
    print('CANDIDATES '+json.dumps(metrics, indent=2), flush=True)
    # An inferior coordinate-MSE experiment is evidence, not a reason to overwrite a submission.
    if metrics[selected_name]['rmse'] >= metrics['old_quadratic']['rmse']:
        selection['needs_revision'] = True
        dump(OUTPUT/'selection.json', selection)
        print('MSE candidate did not beat the current original-model decoder; no final export.', flush=True)
        return selection
    print(f'SELECTED {selected_name}', flush=True)
    old_split = json.loads((ROOT/'outputs/split.json').read_text())
    old_train = by_id.loc[old_split['train_ids']].reset_index(drop=True)
    config = {**CONFIG, 'coordinate_weight': 1.}
    val_fit = fitted(old_train, ROOT/'outputs/validation/best.pt', config, OUTPUT/'validation/pixel_mse')
    full_fit = fitted(records, ROOT/'outputs/full/final.pt', config, OUTPUT/'full/pixel_mse')
    mse_weight = 1. if selected_name == 'pixel_mse' else .5
    val_bundle = {'split': 'validation', 'validation_ids': old_split['val_ids'],
                  'decoder': 'expectation', 'coarse': [], 'refiners': [], 'expected_rmse': None}
    full_bundle = {'split': 'test', 'decoder': 'expectation', 'coarse': [], 'refiners': []}
    for bundle, fitted_result, train, old in [
        (val_bundle, val_fit, old_train, ROOT/'outputs/validation/best.pt'),
        (full_bundle, full_fit, records, ROOT/'outputs/full/final.pt')]:
        bundle['coarse'].append({'checkpoint': fitted_result['checkpoint'], 'weight': mse_weight,
                                 'decoder': 'expectation', 'region': training_region(train),
                                 'temperature': CONFIG.get('temperature', 1.)})
        if mse_weight < 1:
            bundle['coarse'].append({'checkpoint': str(old), 'weight': 1-mse_weight,
                                     'decoder': 'quadratic', 'region': training_region(train)})
    del val_bundle['expected_rmse']
    dump(OUTPUT/'bundle_validation.json', val_bundle)
    dump(OUTPUT/'bundle_full.json', full_bundle)
    validation = run_prediction(OUTPUT/'bundle_validation.json', None, OUTPUT/'validation_prediction')
    test = run_prediction(OUTPUT/'bundle_full.json', ROOT/'submission_metric.csv', OUTPUT/'prediction')
    assert {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in hashes} == hashes
    summary = {**selection, 'validation12': validation, 'test_verification': test,
               'validation_model': val_fit, 'full_model': full_fit, 'old_submissions_unchanged': hashes,
               'new_public_score': None,
               'protocol': 'Original384 architecture; same initialization, augmentation and schedule in CE vs CE+pixelMSE/2500; joint phase61/80 epochs; grouped80 OOF selection precedes12-image check; training/inference masked expectation match.'}
    dump(OUTPUT/'summary.json', summary)
    print(json.dumps({'selected': selected_name, 'oof_rmse': metrics[selected_name]['rmse'],
                      'old_quadratic_oof_rmse': metrics['old_quadratic']['rmse'],
                      'validation12': validation, 'test': test}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixel-MSE fine-tuning of the original384 fovea model')
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    if args.predict_only:
        run_prediction(OUTPUT/'bundle_full.json', ROOT/'submission_metric.csv', OUTPUT/'prediction')
    else:
        run()
