"""Fixed-budget grouped OOF audit after the public leaderboard regression."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from fovea.data import pixel_metrics, restore_xy


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/generalization_audit'


def build_folds(frame):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1701)
    return list(splitter.split(frame, y=frame.width, groups=frame.group))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def audit():
    import torch
    from fovea.optimized import load_model
    from fovea.training import fit, image_tensor, infer

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / 'outputs/records.csv')
    original = json.loads((ROOT / 'outputs/summary.json').read_text())
    optimized = json.loads((ROOT / 'outputs/optimization/optimization_summary.json').read_text())
    variants = [('original384', original['config'], 61)]
    for member in optimized['selection']['members']:
        variants.append((f'optimized640_seed{member["config"]["seed"]}', member['config'], member['best_epoch']))
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ['submission.csv', 'submission_optimized.csv']}
    folds = build_folds(frame)
    schedule = []
    oof = frame[['id', 'Fovea_X', 'Fovea_Y', 'width', 'height', 'group']].copy()
    oof['fold'] = -1
    for column in ['original_x', 'original_y', 'optimized_x', 'optimized_y']:
        oof[column] = np.nan
    fold_results = []
    for fold, (train_indices, val_indices) in enumerate(folds):
        train, val = frame.iloc[train_indices].copy(), frame.iloc[val_indices].copy()
        assert set(train.group).isdisjoint(val.group)
        assert set(train.id).isdisjoint(val.id)
        oof.loc[val_indices, 'fold'] = fold
        schedule.append({'fold': fold, 'training_ids': train.id.tolist(), 'validation_ids': val.id.tolist()})
        dump(OUTPUT / 'splits.json', schedule)
        predictions = {}
        for name, config, epochs in variants:
            folder = OUTPUT / f'fold{fold}' / name
            result_path = folder / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                assert result['config'] == config and result['epochs'] == epochs
                assert result['training_ids'] == train.id.tolist() and result['validation_ids'] == val.id.tolist()
            else:
                print(f'Fold {fold+1}/3 {name}: train={len(train)}, held-out={len(val)}, fixed_epochs={epochs}', flush=True)
                fitted = fit(train, config, folder, epochs=epochs)
                model, saved = load_model(fitted['checkpoint'], 'cuda')
                assert saved['epoch'] == epochs and set(saved['training_ids']) == set(train.id)
                assert set(saved['training_ids']).isdisjoint(val.id)
                images = image_tensor(val.path, config['image_size'], torch.device('cuda'))
                xy = restore_xy(infer(model, images, config['batch_size']), val[['width', 'height']])
                output = val[['id', 'Fovea_X', 'Fovea_Y', 'width', 'height']].copy()
                output[['pred_x', 'pred_y']] = xy
                output.to_csv(folder / 'held_out.csv', index=False)
                result = {'config': config, 'epochs': epochs, 'checkpoint': fitted['checkpoint'],
                          'training_ids': train.id.tolist(), 'validation_ids': val.id.tolist(),
                          'metrics': pixel_metrics(val[['Fovea_X', 'Fovea_Y']].to_numpy(), xy)}
                dump(result_path, result)
                del model, images
            output = pd.read_csv(folder / 'held_out.csv')
            assert output.id.tolist() == val.id.tolist()
            predictions[name] = output[['pred_x', 'pred_y']].to_numpy()
        original_xy = predictions['original384']
        optimized_xy = np.mean([value for name, value in predictions.items() if name != 'original384'], axis=0)
        oof.loc[val_indices, ['original_x', 'original_y']] = original_xy
        oof.loc[val_indices, ['optimized_x', 'optimized_y']] = optimized_xy
        actual = val[['Fovea_X', 'Fovea_Y']].to_numpy()
        record = {'fold': fold, 'train_count': len(train), 'validation_count': len(val),
                  'original': pixel_metrics(actual, original_xy), 'optimized': pixel_metrics(actual, optimized_xy)}
        fold_results.append(record)
        dump(OUTPUT / 'fold_results.json', fold_results)
        oof.to_csv(OUTPUT / 'oof.csv', index=False)
        print(f'FOLD {fold+1}: original={record["original"]["rmse"]:.4f}, optimized={record["optimized"]["rmse"]:.4f}', flush=True)
    assert not oof[['original_x', 'original_y', 'optimized_x', 'optimized_y']].isna().any().any()
    assert oof.id.is_unique and len(oof) == 80 and (oof.fold >= 0).all()
    actual = oof[['Fovea_X', 'Fovea_Y']].to_numpy()
    old_xy = oof[['original_x', 'original_y']].to_numpy()
    new_xy = oof[['optimized_x', 'optimized_y']].to_numpy()
    oof['original_distance'] = np.linalg.norm(old_xy - actual, axis=1)
    oof['optimized_distance'] = np.linalg.norm(new_xy - actual, axis=1)
    oof.to_csv(OUTPUT / 'oof.csv', index=False)
    original_split = json.loads((ROOT / 'outputs/split.json').read_text())
    untouched = oof.id.isin(original_split['train_ids']).to_numpy()
    result = {'protocol': '3 folds, StratifiedGroupKFold, camera-size strata, seed1701, fixed schedules/epochs, no OOF early stopping',
              'images': len(oof), 'folds': fold_results,
              'original_oof': pixel_metrics(actual, old_xy), 'optimized_oof': pixel_metrics(actual, new_xy),
              'original_train68_oof': pixel_metrics(actual[untouched], old_xy[untouched]),
              'optimized_train68_oof': pixel_metrics(actual[untouched], new_xy[untouched]),
              'improved_folds': sum(r['optimized']['rmse'] < r['original']['rmse'] for r in fold_results),
              'public_scores_user_feedback': {'original': 12.66810, 'optimized': 13.43397},
              'recommended_submission': 'submission.csv',
              'note': 'Configurations and epoch budgets were selected earlier using the original 12-image split; OOF is a sensitivity audit, not a new untouched test set.'}
    assert {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in hashes} == hashes
    result['submission_hashes_unchanged'] = hashes
    dump(OUTPUT / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit()
