"""Compare the baseline, a new candidate and their equal probability average."""
import argparse
import json
from pathlib import Path

import numpy as np

from train import score_probabilities
from vessel.data import load_case
from vessel.submission import make_submission_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, default=Path('outputs/improved/baseline_d4'))
    parser.add_argument('--candidate-run', type=Path, default=Path('outputs/improved/validation_context'))
    parser.add_argument('--baseline-final', type=Path, default=Path('outputs/final/model.pt'))
    parser.add_argument('--candidate-final', type=Path, default=Path('outputs/improved/final_context/model.pt'))
    parser.add_argument('--data', type=Path, default=Path('segmentation'))
    parser.add_argument('--output', type=Path, default=Path('outputs/improved'))
    args = parser.parse_args()
    runs = [Path('outputs/validation'), args.baseline_run, args.candidate_run]
    summaries = [json.loads((run / 'summary.json').read_text(encoding='utf-8')) for run in runs]
    saved = [np.load(run / 'validation_predictions.npz') for run in runs]
    ids = saved[0]['ids'].tolist()
    for summary, data in zip(summaries, saved):
        if data['ids'].tolist() != ids or summary['config']['validation_ids'] != ids:
            raise ValueError('All runs must use the same validation images in the same order.')
        if set(summary['config']['train_ids']) & set(ids):
            raise ValueError('Validation images were used for training.')
        np.testing.assert_array_equal(data['labels'], saved[0]['labels'])
    labels = [load_case(args.data, 'train', case_id)[2] for case_id in ids]
    probabilities = [data['probabilities'] for data in saved]
    probabilities.append((probabilities[1] + probabilities[2]) / 2)
    names = ['baseline_flips', 'baseline_d4', 'context_d4', 'mean_baseline_context_d4']
    base_member = {'checkpoint': str(args.baseline_final), 'validation_checkpoint': 'outputs/validation/best.pt', 'tta': 'd4'}
    candidate_member = {'checkpoint': str(args.candidate_final), 'validation_checkpoint': str(args.candidate_run / 'best.pt'),
                        'tta': summaries[2]['config']['tta']}
    member_sets = [[dict(base_member, tta='flips')], [base_member], [candidate_member], [base_member, candidate_member]]
    scores = [score_probabilities(p, labels, ids)[0] for p in probabilities]
    selected = max(range(len(scores)), key=lambda i: scores[i]['mean_dice'])
    selection = dict(scores[selected], strategy=names[selected], members=member_sets[selected],
                     validation_ids=ids, baseline_mean_dice=scores[0]['mean_dice'],
                     improvement=scores[selected]['mean_dice'] - scores[0]['mean_dice'],
                     comparisons=[dict(score, strategy=name) for name, score in zip(names, scores)],
                     note='Same four validation images select checkpoint, threshold and strategy; not a blind test score.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'ensemble_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    np.savez_compressed(args.output / 'validation_predictions.npz', ids=np.array(ids),
                        probabilities=probabilities[selected], labels=saved[0]['labels'],
                        masks=np.stack([make_submission_mask(p > selection['threshold']) for p in probabilities[selected]]))
    for name, score in zip(names, scores):
        print(f'{name}: converted={score["mean_dice"]:.6f} native={score["native_mean_dice"]:.6f} threshold={score["threshold"]:.3f}')
    print(f'Selected {names[selected]}, improvement={selection["improvement"]:.6f}')


if __name__ == '__main__':
    main()
