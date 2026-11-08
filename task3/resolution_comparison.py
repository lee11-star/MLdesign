"""A single resolution change to the successful pixel-MSE objective."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fovea.data import pixel_metrics
from fovea.optimized import infer_bundle
from predict_ensemble import run_prediction
from coordinate_training import CONFIG, fitted, evaluate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/target115_resolution'
RESOLUTION_CONFIG = {**CONFIG, 'image_size': 512, 'coordinate_weight': 1.}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def bundle_for(split, checkpoint, region, weight):
    bundle = copy.deepcopy(json.loads((ROOT/f'outputs/metric_mse_joint/bundle_{split}.json').read_text()))
    old = bundle['coarse'][0]
    new = {'checkpoint': checkpoint, 'weight': weight, 'decoder': 'expectation', 'temperature': .25, 'region': region}
    bundle['coarse'] = [new]
    if weight < 1:
        old['weight'] = 1-weight
        bundle['coarse'].append(old)
    return bundle


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = ['submission.csv', 'submission_optimized.csv', 'submission_improved.csv', 'submission_metric.csv']
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    records = pd.read_csv(ROOT/'outputs/records.csv')
    by_id = records.set_index('id', drop=False)
    baseline_file = pd.read_csv(ROOT/'outputs/metric_mse_joint/oof.csv').set_index('id').loc[records.id]
    actual = records[['Fovea_X', 'Fovea_Y']].to_numpy()
    baseline = baseline_file[['pixel_mse_x', 'pixel_mse_y']].to_numpy()
    predicted = np.full_like(baseline, np.nan)
    splits = json.loads((ROOT/'outputs/generalization_audit/splits.json').read_text())
    for split in splits:
        train = by_id.loc[split['training_ids']].reset_index(drop=True)
        val = by_id.loc[split['validation_ids']].reset_index(drop=True)
        assert set(train.group).isdisjoint(val.group)
        folder = OUTPUT/f'fold{split["fold"]}/metric512'
        print(f'Resolution512 fold{split["fold"]+1}/3', flush=True)
        result = fitted(train, RESOLUTION_CONFIG['pretrained'], RESOLUTION_CONFIG, folder)
        xy = evaluate(val, train, result['checkpoint'], folder)
        index = records.index[records.id.isin(val.id)]
        assert records.loc[index].id.tolist() == val.id.tolist()
        predicted[index] = xy
    assert np.isfinite(predicted).all()
    choices = {'metric512': {'weight': 1., 'xy': predicted},
               'metric512_half384': {'weight': .5, 'xy': (predicted+baseline)/2}}
    oof = records[['id', 'Fovea_X', 'Fovea_Y']].copy()
    baseline_metrics = pixel_metrics(actual, baseline)
    for name, item in choices.items():
        item['metrics'] = pixel_metrics(actual, item['xy'])
        item['folds'] = []
        for split in splits:
            mask = records.id.isin(split['validation_ids']).to_numpy()
            item['folds'].append({'fold': split['fold'], 'baseline': pixel_metrics(actual[mask], baseline[mask]),
                                  'candidate': pixel_metrics(actual[mask], item['xy'][mask])})
        item['improved_folds'] = sum(row['candidate']['rmse'] < row['baseline']['rmse'] for row in item['folds'])
        oof[[name+'_x', name+'_y']] = item['xy']
    oof.to_csv(OUTPUT/'oof.csv', index=False)
    screen = {'config': RESOLUTION_CONFIG, 'baseline': baseline_metrics,
              'candidates': {name: {key: value for key, value in item.items() if key != 'xy'} for name, item in choices.items()}}
    dump(OUTPUT/'screening.json', screen)
    print('SCREEN '+json.dumps({name: {'rmse': item['metrics']['rmse'], 'improved_folds': item['improved_folds']} for name, item in choices.items()}, indent=2), flush=True)
    eligible = [name for name, item in choices.items() if item['metrics']['rmse'] < baseline_metrics['rmse'] and item['improved_folds'] >= 2]
    if not eligible:
        print('Resolution candidates did not pass OOF comparison; no export.', flush=True)
        return screen
    original_split = json.loads((ROOT/'outputs/split.json').read_text())
    train = by_id.loc[original_split['train_ids']].reset_index(drop=True)
    val = by_id.loc[original_split['val_ids']].reset_index(drop=True)
    val_fit = fitted(train, RESOLUTION_CONFIG['pretrained'], RESOLUTION_CONFIG, OUTPUT/'validation/metric512')
    original_val = pd.read_csv(ROOT/'outputs/metric_mse_joint/validation_prediction/predictions.csv').set_index('id').loc[val.id]
    old_xy = original_val[['pred_x', 'pred_y']].to_numpy()
    val_bundle = bundle_for('validation', val_fit['checkpoint'], val_fit['region'], 1.)
    new_xy = infer_bundle(val, val_bundle)
    val_actual = val[['Fovea_X', 'Fovea_Y']].to_numpy()
    reference = pixel_metrics(val_actual, old_xy)
    for name in eligible:
        weight = choices[name]['weight']
        choices[name]['validation12'] = pixel_metrics(val_actual, weight*new_xy+(1-weight)*old_xy)
    guarded = [name for name in eligible if choices[name]['validation12']['rmse'] <= reference['rmse']]
    screen['validation12_baseline'] = reference
    screen['validation12_candidates'] = {name: choices[name]['validation12'] for name in eligible}
    dump(OUTPUT/'screening.json', screen)
    if not guarded:
        print('Resolution OOF gains failed auxiliary12 comparison; no export.', flush=True)
        return screen
    selected_name = min(guarded, key=lambda name: choices[name]['metrics']['rmse'])
    selected = choices[selected_name]
    print(f'SELECTED {selected_name}: OOF={selected["metrics"]["rmse"]:.4f}, val12={selected["validation12"]["rmse"]:.4f}', flush=True)
    dump(OUTPUT/'selection.json', {'name': selected_name, **{key: value for key, value in selected.items() if key != 'xy'}})
    full_fit = fitted(records, RESOLUTION_CONFIG['pretrained'], RESOLUTION_CONFIG, OUTPUT/'full/metric512')
    validation_bundle = bundle_for('validation', val_fit['checkpoint'], val_fit['region'], selected['weight'])
    validation_bundle['expected_rmse'] = selected['validation12']['rmse']
    full_bundle = bundle_for('full', full_fit['checkpoint'], full_fit['region'], selected['weight'])
    dump(OUTPUT/'bundle_validation.json', validation_bundle)
    dump(OUTPUT/'bundle_full.json', full_bundle)
    verification_val = run_prediction(OUTPUT/'bundle_validation.json', None, OUTPUT/'validation_prediction')
    verification_test = run_prediction(OUTPUT/'bundle_full.json', ROOT/'submission_target115.csv', OUTPUT/'prediction')
    assert {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in hashes} == hashes
    result = {**screen, 'selected': selected_name, 'selection': {key: value for key, value in selected.items() if key != 'xy'},
              'validation_reloaded': verification_val, 'test_verification': verification_test,
              'full_model': full_fit, 'old_files_unchanged': hashes, 'target_public_rmse': 11.5,
              'target_verified': False, 'new_public_score': None}
    dump(OUTPUT/'summary.json', result)
    print(json.dumps({'selected': selected_name, 'oof': selected['metrics'], 'validation12': verification_val,
                      'test': verification_test, 'target_verified': False}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='512 pixel-MSE resolution control toward public RMSE11.5')
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    if args.predict_only:
        run_prediction(OUTPUT/'bundle_full.json', ROOT/'submission_target115.csv', OUTPUT/'prediction')
    else:
        run()
