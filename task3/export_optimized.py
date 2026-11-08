"""Select a measured improvement, check a second seed, refit and export a bundle."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from fovea.data import pixel_metrics
from fovea.refinement import fit_refiner
from fovea.training import fit
from predict_ensemble import run_prediction
from resolution_screen import ROOT, OUTPUT, load_split


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def finalize():
    original_sha = hashlib.sha256((ROOT / 'submission.csv').read_bytes()).hexdigest()
    records, train, val = load_split()
    old = pd.read_csv(ROOT / 'outputs/validation/predictions.csv')
    actual = old[['Fovea_X', 'Fovea_Y']].to_numpy()
    coarse_original = old[['pred_x', 'pred_y']].to_numpy()
    screening = json.loads((OUTPUT / 'screening.json').read_text())
    refinement = json.loads((OUTPUT / 'refinement/result.json').read_text())
    # The resolution comparison selected the best individual new model. All preset
    # resolution ensembles remain reported, including candidates we did not select.
    coarse_name = min(screening['experiments'], key=lambda name: screening['experiments'][name]['metrics']['rmse'])
    coarse_result = screening['experiments'][coarse_name]
    local_wins = refinement['metrics']['rmse'] < coarse_result['metrics']['rmse']
    name = 'refinement' if local_wins else coarse_name
    first = refinement if local_wins else coarse_result
    fit_function = fit_refiner if local_wins else fit
    config = {**first['config'], 'seed': 2026}
    second_folder = OUTPUT / f'{name}_seed2026'
    if (second_folder / 'result.json').exists():
        second = json.loads((second_folder / 'result.json').read_text())
        assert second['config'] == config
    else:
        kwargs = {'validation': val}
        if local_wins:
            kwargs['coarse_xy'] = coarse_original
        print(f'Checking second seed for {name}', flush=True)
        second = fit_function(train, config, second_folder, **kwargs)
        second['config'] = config
        dump(second_folder / 'result.json', second)
    first_predictions = pd.read_csv(OUTPUT / name / 'predictions.csv')
    second_predictions = pd.read_csv(second_folder / 'predictions.csv')
    assert first_predictions.id.tolist() == second_predictions.id.tolist() == old.id.tolist()
    first_xy = first_predictions[['pred_x', 'pred_y']].to_numpy()
    second_xy = second_predictions[['pred_x', 'pred_y']].to_numpy()
    candidates = {
        'seed42': {'members': [first], 'xy': first_xy},
        'seed2026': {'members': [second], 'xy': second_xy},
        'equal_seed_mean': {'members': [first, second], 'xy': (first_xy + second_xy) / 2}}
    for candidate in candidates.values():
        candidate['metrics'] = pixel_metrics(actual, candidate['xy'])
    winner = min(candidates, key=lambda key: candidates[key]['metrics']['rmse'])
    selected = candidates[winner]
    assert selected['metrics']['rmse'] < screening['baseline']['metrics']['rmse']
    count = len(selected['members'])
    print(f'Selected {name}/{winner}: RMSE={selected["metrics"]["rmse"]:.4f}px', flush=True)
    val_bundle = {'split': 'validation', 'validation_ids': val.id.tolist(),
                  'coarse': [], 'refiners': [], 'refinement_weight': .5,
                  'expected_rmse': selected['metrics']['rmse']}
    full_bundle = {'split': 'test', 'coarse': [], 'refiners': [], 'refinement_weight': .5}
    if local_wins:
        val_bundle['coarse'] = [{'checkpoint': str(ROOT / 'outputs/validation/best.pt'), 'weight': 1.}]
        full_bundle['coarse'] = [{'checkpoint': str(ROOT / 'outputs/full/final.pt'), 'weight': 1.}]
    for member in selected['members']:
        stage = 'refiners' if local_wins else 'coarse'
        val_bundle[stage].append({'checkpoint': member['checkpoint'], 'weight': 1 / count})
        full_folder = OUTPUT / f'full_{name}_seed{member["config"]["seed"]}'
        if (full_folder / 'result.json').exists():
            full = json.loads((full_folder / 'result.json').read_text())
            assert full['config'] == member['config'] and full['epochs'] == member['best_epoch']
        else:
            full = fit_function(records, member['config'], full_folder, epochs=member['best_epoch'])
            full['config'] = member['config']
            dump(full_folder / 'result.json', full)
        full_bundle[stage].append({'checkpoint': full['checkpoint'], 'weight': 1 / count})
    dump(OUTPUT / 'bundle_validation.json', val_bundle)
    dump(OUTPUT / 'bundle_full.json', full_bundle)
    verification_val = run_prediction(OUTPUT / 'bundle_validation.json', None, OUTPUT / 'verified_validation')
    verification_test = run_prediction(OUTPUT / 'bundle_full.json', ROOT / 'submission_optimized.csv', OUTPUT / 'prediction')
    assert hashlib.sha256((ROOT / 'submission.csv').read_bytes()).hexdigest() == original_sha
    comparison = old[['id', 'Fovea_X', 'Fovea_Y', 'distance']].rename(columns={'distance': 'old_distance'})
    comparison[['pred_x', 'pred_y']] = selected['xy']
    comparison['new_distance'] = np.linalg.norm(selected['xy'] - actual, axis=1)
    comparison.to_csv(OUTPUT / 'validation_comparison.csv', index=False)
    report = {'original_submission_sha256': original_sha,
              'old_validation': screening['baseline']['metrics'],
              'resolution_candidates': screening['candidates'], 'refinement': refinement,
              'second_seed_candidates': {key: value['metrics'] for key, value in candidates.items()},
              'selection': {'stage': name, 'variant': winner, 'metrics': selected['metrics'],
                            'members': selected['members']},
              'relative_rmse_reduction': 1 - selected['metrics']['rmse'] / screening['baseline']['metrics']['rmse'],
              'validation_reloaded': verification_val, 'test_verification': verification_test,
              'old_submission_unchanged': True}
    dump(OUTPUT / 'optimization_summary.json', report)
    print(json.dumps({'selection': report['selection'], 'relative_rmse_reduction': report['relative_rmse_reduction'],
                      'test_verification': verification_test}, indent=2), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    finalize()
