import argparse
import json

import pandas as pd

from fovea.refinement import fit_refiner
from resolution_screen import ROOT, OUTPUT, load_split


if __name__ == '__main__':
    argparse.ArgumentParser(description='Train the original local-refinement control').parse_args()
    original = json.loads((ROOT / 'outputs/summary.json').read_text())
    _, train, val = load_split()
    # Start with the original model as coarse localization to isolate the refinement effect.
    coarse = pd.read_csv(ROOT / 'outputs/validation/predictions.csv')
    assert coarse.id.tolist() == val.id.tolist()
    config = {**original['config'], 'stage': 'refine', 'image_size': 320, 'source_size': 1536,
              'crop_fraction': .28, 'sigma': .035, 'epochs': 100, 'lr': .0005,
              'coarse_validation': str(ROOT / 'outputs/validation/predictions.csv')}
    result = fit_refiner(train, config, OUTPUT / 'refinement', validation=val,
                         coarse_xy=coarse[['pred_x', 'pred_y']].to_numpy())
    (OUTPUT / 'refinement/result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
