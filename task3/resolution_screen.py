"""Fixed-split resolution and supervision-width experiments, preserving round one."""
import argparse
import json
from pathlib import Path

import pandas as pd

from fovea.data import pixel_metrics
from fovea.training import fit


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/optimization'


def load_split():
    records = pd.read_csv(ROOT / 'outputs/records.csv').set_index('id', drop=False)
    split = json.loads((ROOT / 'outputs/split.json').read_text())
    train = records.loc[split['train_ids']].reset_index(drop=True)
    val = records.loc[split['val_ids']].reset_index(drop=True)
    assert set(train.group).isdisjoint(val.group)
    return records.reset_index(drop=True), train, val


def screen():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / 'outputs/summary.json').read_text())
    _, train, val = load_split()
    experiments = [('res512_sigma015', 512, .015), ('res640_sigma012', 640, .012)]
    results = {'baseline': summary['validation'], 'experiments': {}}
    for name, size, sigma in experiments:
        folder = OUTPUT / name
        config = {**summary['config'], 'image_size': size, 'sigma': sigma, 'epochs': 100}
        if (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text())
            assert result['config'] == config
            print(f'Reusing {name}: RMSE={result["metrics"]["rmse"]:.4f}', flush=True)
        else:
            print(f'Starting {name}', flush=True)
            result = fit(train, config, folder, validation=val)
            result['config'] = config
            (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results['experiments'][name] = result
        (OUTPUT / 'screening.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    # Equal weighting is specified in advance; compare against each constituent.
    old = pd.read_csv(ROOT / 'outputs/validation/predictions.csv')
    arrays = {'original': old[['pred_x', 'pred_y']].to_numpy()}
    actual = old[['Fovea_X', 'Fovea_Y']].to_numpy()
    for name, _, _ in experiments:
        frame = pd.read_csv(OUTPUT / name / 'predictions.csv')
        assert frame.id.tolist() == old.id.tolist()
        arrays[name] = frame[['pred_x', 'pred_y']].to_numpy()
    import numpy as np
    results['candidates'] = {name: pixel_metrics(actual, prediction) for name, prediction in arrays.items()}
    results['candidates']['new_equal_mean'] = pixel_metrics(actual, np.mean([arrays[n] for n, _, _ in experiments], axis=0))
    results['candidates']['all_equal_mean'] = pixel_metrics(actual, np.mean(list(arrays.values()), axis=0))
    (OUTPUT / 'screening.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results['candidates'], indent=2), flush=True)
    return results


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    screen()
