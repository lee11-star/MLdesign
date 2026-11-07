"""One new backbone, fixed folds and budgets, and a gated 50/50 ensemble."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fovea.data import pixel_metrics
from fovea.optimized import infer_bundle
from fovea.supported_decoder import training_region
from predict_ensemble import run_prediction
from coordinate_training import CONFIG, fitted, evaluate


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/backbone_20261027'
SOURCE = ROOT/'outputs/target115_resolution'
NEW_CONFIG = {**CONFIG, 'architecture': 'convnext_tiny', 'backbone_lr': .00003, 'coordinate_weight': 1.,
              'pretrained': str(ROOT.parent/'task1/outputs/torch_cache/hub/checkpoints/convnext_tiny-983f1562.pth')}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def comparison(frame, predicted, baseline, splits):
    actual = frame[['Fovea_X', 'Fovea_Y']].to_numpy()
    result = {'metrics': pixel_metrics(actual, predicted), 'folds': []}
    for split in splits:
        mask = frame.id.isin(split['validation_ids']).to_numpy()
        result['folds'].append({'fold': split['fold'], 'images': int(mask.sum()),
                               'baseline': pixel_metrics(actual[mask], baseline[mask]),
                               'candidate': pixel_metrics(actual[mask], predicted[mask])})
    result['improved_folds'] = sum(row['candidate']['rmse'] < row['baseline']['rmse'] for row in result['folds'])
    for name, mask in [('without49', frame.id.ne(49).to_numpy()),
                       ('without_previous5', ~frame.id.isin([49, 72, 24, 62, 16]).to_numpy())]:
        result[name] = {'baseline': pixel_metrics(actual[mask], baseline[mask]),
                        'candidate': pixel_metrics(actual[mask], predicted[mask])}
    result['improved_images'] = int((np.linalg.norm(predicted-actual, axis=1) < np.linalg.norm(baseline-actual, axis=1)).sum())
    return result


def new_bundle(checkpoint, train, split, weight, old_bundle):
    result = copy.deepcopy(old_bundle)
    result.pop('expected_rmse', None)
    for item in result['coarse']:
        item['weight'] *= 1-weight
    if weight == 1:
        result['coarse'] = []
    result['coarse'].append({'checkpoint': str(checkpoint), 'weight': weight, 'decoder': 'expectation',
                             'temperature': .25, 'region': training_region(train)})
    result['split'] = split
    return result


def run(candidate_name='convnext', baseline_columns=('metric512_half384_x', 'metric512_half384_y'),
        submission_name='submission_backbone.csv'):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('submission*.csv')}
    dump(OUTPUT/'history_hashes.json', history_hashes)
    dump(OUTPUT/'protocol.json', {'config': NEW_CONFIG, 'candidate_weights': [1., .5],
        'gate': 'OOF <= 97% current; >=2 folds improved; excluding49 improved; auxiliary12 <= current',
        'user_feedback': 'Multiple follow-up submissions brought only weak gains or regressed.',
        'public_score': None, 'labels_modified': False})
    records = pd.read_csv(ROOT/'outputs/records.csv')
    by_id = records.set_index('id', drop=False)
    splits = json.loads((ROOT/'outputs/generalization_audit/splits.json').read_text())
    old = pd.read_csv(SOURCE/'oof.csv').set_index('id').loc[records.id]
    actual = records[['Fovea_X', 'Fovea_Y']].to_numpy()
    np.testing.assert_allclose(actual, old[['Fovea_X', 'Fovea_Y']])
    baseline = old[list(baseline_columns)].to_numpy()
    predicted = np.full_like(baseline, np.nan)
    oof = records[['id', 'Fovea_X', 'Fovea_Y']].copy()
    oof[['baseline_x', 'baseline_y']] = baseline
    for split in splits:
        fold = split['fold']
        train = by_id.loc[split['training_ids']].reset_index(drop=True)
        val = by_id.loc[split['validation_ids']].reset_index(drop=True)
        assert set(train.id).isdisjoint(val.id) and set(train.group).isdisjoint(val.group)
        assert set(train.id) | set(val.id) == set(records.id)
        folder = OUTPUT/f'fold{fold}'
        print(f'{candidate_name} fold{fold+1}/3: {len(train)} train, {len(val)} held out', flush=True)
        result = fitted(train, NEW_CONFIG['pretrained'], NEW_CONFIG, folder)
        xy = evaluate(val, train, result['checkpoint'], folder)
        indices = records.index[records.id.isin(val.id)]
        assert records.loc[indices].id.tolist() == val.id.tolist()
        predicted[indices] = xy
        oof.loc[indices, [candidate_name+'_x', candidate_name+'_y']] = xy
        oof.loc[indices, 'fold'] = fold
        oof.to_csv(OUTPUT/'oof.csv', index=False)
        print(f'FOLD{fold+1}: baseline={pixel_metrics(actual[indices], baseline[indices])["rmse"]:.4f}, '
              f'{candidate_name}={pixel_metrics(actual[indices], xy)["rmse"]:.4f}, '
              f'half={pixel_metrics(actual[indices], (xy+baseline[indices])/2)["rmse"]:.4f}', flush=True)
    assert np.isfinite(predicted).all() and oof.id.is_unique and len(oof) == 80
    choices = {candidate_name: (1., predicted), 'half': (.5, (predicted+baseline)/2)}
    baseline_metrics = pixel_metrics(actual, baseline)
    candidates = {name: {'weight': weight, **comparison(records, xy, baseline, splits)} for name, (weight, xy) in choices.items()}
    for name, (_, xy) in choices.items():
        oof[[name+'_x', name+'_y']] = xy
        oof[name+'_sse_change'] = ((xy-actual)**2).sum(1)-((baseline-actual)**2).sum(1)
    oof.to_csv(OUTPUT/'oof.csv', index=False)
    eligible = [name for name, item in candidates.items() if item['metrics']['rmse'] <= .97*baseline_metrics['rmse']
                and item['improved_folds'] >= 2 and item['without49']['candidate']['rmse'] < item['without49']['baseline']['rmse']]
    summary = {'config': NEW_CONFIG, 'baseline': baseline_metrics, 'candidates': candidates,
               'selected': None, 'recommended': False, 'public_score': None}
    dump(OUTPUT/'summary.json', summary)
    print('SCREEN '+json.dumps({name: {'rmse': item['metrics']['rmse'], 'improved_folds': item['improved_folds']} for name, item in candidates.items()}), flush=True)
    if eligible:
        original_split = json.loads((ROOT/'outputs/split.json').read_text())
        train = by_id.loc[original_split['train_ids']].reset_index(drop=True)
        val = by_id.loc[original_split['val_ids']].reset_index(drop=True)
        val_fit = fitted(train, NEW_CONFIG['pretrained'], NEW_CONFIG, OUTPUT/'validation')
        old_bundle = json.loads((SOURCE/'bundle_validation.json').read_text())
        current = infer_bundle(val, old_bundle)
        val_actual = val[['Fovea_X', 'Fovea_Y']].to_numpy()
        reference = pixel_metrics(val_actual, current)
        summary['validation12_baseline'] = reference
        for name in eligible:
            bundle = new_bundle(val_fit['checkpoint'], train, 'validation', choices[name][0], old_bundle)
            xy = infer_bundle(val, bundle)
            candidates[name]['validation12'] = pixel_metrics(val_actual, xy)
        eligible = [name for name in eligible if candidates[name]['validation12']['rmse'] <= reference['rmse']]
        if eligible:
            selected = min(eligible, key=lambda name: candidates[name]['metrics']['rmse'])
            weight = choices[selected][0]
            full_fit = fitted(records, NEW_CONFIG['pretrained'], NEW_CONFIG, OUTPUT/'full')
            val_bundle = new_bundle(val_fit['checkpoint'], train, 'validation', weight, old_bundle)
            val_bundle['expected_rmse'] = candidates[selected]['validation12']['rmse']
            full_bundle = new_bundle(full_fit['checkpoint'], records, 'test', weight, json.loads((SOURCE/'bundle_full.json').read_text()))
            dump(OUTPUT/'bundle_validation.json', val_bundle)
            dump(OUTPUT/'bundle_full.json', full_bundle)
            summary.update(selected=selected, recommended=True)
            summary['validation_reloaded'] = run_prediction(OUTPUT/'bundle_validation.json', None, OUTPUT/'validation_prediction')
            summary['test_verification'] = run_prediction(OUTPUT/'bundle_full.json', ROOT/submission_name, OUTPUT/'prediction')
    assert {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in history_hashes} == history_hashes
    summary['old_submissions_unchanged'] = history_hashes
    dump(OUTPUT/'summary.json', summary)
    print('FINAL '+json.dumps({'selected': summary['selected'], 'recommended': summary['recommended'],
                              'metrics': {name: item['metrics'] for name, item in candidates.items()}}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fixed ConvNeXt backbone experiment scored by pixel RMSE')
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    if args.predict_only:
        run_prediction(OUTPUT/'bundle_full.json', ROOT/'submission_backbone.csv', OUTPUT/'prediction')
    else:
        run()
