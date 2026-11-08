"""Compare this round's saved holdout probabilities, then retrain selected members.

Run from task2 with the existing PyTorch interpreter. --comparison-only skips
full-data training. This experiment script does not upload to Kaggle.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import numpy as np
import torch

from emotion_classifier.data import (CLASSES, CLASS_TO_LABEL, DEFAULT_DATA,
                                     DEFAULT_TEMPLATE, check_split, load_arrays,
                                     read_manifest, save_json)
from emotion_classifier.ensemble import align_probabilities, fuse_probabilities
from emotion_classifier.inference import load_checkpoint, predict_probabilities, write_submission
from model_selection import probability_metrics, train_stage, check_requested_config
from baseline_training import completed_run_is_consistent
from evaluate import run_evaluation

OUT = ROOT / 'outputs/kaggle80'
OLD = ROOT / 'outputs/optimization'
NEW_NAMES = ('convnext_regularized96', 'convnext_regularized128')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare_models():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = read_json(OLD / 'selection.json')
    baseline = OUT / 'baseline.json'
    if not baseline.exists():
        save_json(dict(validation_accuracy=selection['accuracy'], data_signature=selection['data_signature'],
                       user_reported_kaggle_accuracy=None,
                       original_submissions={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                             for name in ('submission.csv', 'submission_optimized.csv')}), baseline)
    base = read_json(OLD / 'convnext_tiny/config.json')
    for name, size in zip(NEW_NAMES, (96, 128)):
        config = dict(base, model='convnext_tiny', epochs=40, scheduler_epochs=40, image_size=size,
                      lr=0.0001, weight_decay=0.01, label_smoothing=0.1, batch_size=128,
                      seed=42, final_training=False)
        folder = OUT / name
        train_stage(folder, config, 'cuda')
        run_evaluation(folder / 'best.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-only', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from task2')
    torch.set_num_threads(4)
    prepare_models()
    baseline = read_json(OUT / 'baseline.json')
    old_selection = read_json(OLD / 'selection.json')
    signature = hashlib.sha256((ROOT / 'outputs/data/manifest.csv').read_bytes()).hexdigest()
    assert signature == baseline['data_signature'] == old_selection['data_signature']
    rows = read_manifest(ROOT / 'outputs/data/manifest.csv')
    check_split(rows)
    _, labels, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
    ids = np.asarray(ids)
    models = {}
    for name in list(old_selection['members']) + list(NEW_NAMES):
        folder = (OUT if name in NEW_NAMES else OLD) / name
        config = read_json(folder / 'config.json')
        metrics = read_json(folder / 'evaluation/metrics.json')
        summary = read_json(folder / 'summary.json')
        assert config['data_signature'] == signature
        assert config['class_to_label'] == CLASS_TO_LABEL and not config['final_training']
        assert summary['completed_epochs'] == config['epochs']
        assert summary['best_epoch'] == metrics['best_epoch']
        if name in NEW_NAMES:
            expected = dict(config, model='convnext_tiny', epochs=40, scheduler_epochs=40,
                            image_size=96 if name.endswith('96') else 128,
                            lr=0.0001, weight_decay=0.01, label_smoothing=0.1,
                            batch_size=128, seed=42, final_training=False)
            check_requested_config(config, expected)
        assert completed_run_is_consistent(folder, config['epochs'], False)
        checkpoint_config = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)['config']
        assert checkpoint_config == config
        if name in NEW_NAMES:
            use_tta = metrics['selected_tta']
        else:
            use_tta = old_selection['tta'][old_selection['members'].index(name)]
        key = 'flip_tta_probabilities' if use_tta else 'probabilities'
        with np.load(folder / 'evaluation/validation_probabilities.npz') as data:
            assert np.array_equal(data['class_names'], np.asarray(CLASSES))
            probs = align_probabilities(ids, data['ids'], data[key], labels, data['targets'])
        expected = metrics['flip_tta_accuracy'] if use_tta else metrics['accuracy']
        assert float((probs.argmax(1) == labels).mean()) == expected
        models[name] = dict(folder=folder, config=config, metrics=metrics,
                            tta=bool(use_tta), probabilities=probs)

    comparison, probability_arrays = [], {}

    def add(name, members, weights, temperatures=None):
        temperatures = temperatures or [1.0] * len(members)
        probs = fuse_probabilities([models[item]['probabilities'] for item in members], weights, temperatures)
        entry = dict(name=name, members=list(members), weights=list(weights),
                     temperatures=list(temperatures), tta=[models[item]['tta'] for item in members],
                     **probability_metrics(probs, labels))
        comparison.append(entry)
        probability_arrays[name] = probs
        print(f"{name}: accuracy={entry['accuracy']:.6f} cross_entropy={entry['cross_entropy']:.4f}", flush=True)

    old_members = old_selection['members']
    add('previous_ensemble', old_members, old_selection['weights'], old_selection['temperatures'])
    assert comparison[0]['accuracy'] == baseline['validation_accuracy']
    for name in NEW_NAMES:
        add('single:' + name, [name], [1.0])
        add('previous_plus_' + name + '_equal', old_members + [name],
            [value * 0.5 for value in old_selection['weights']] + [0.5],
            old_selection['temperatures'] + [1.0])
    add('new_pair_equal', NEW_NAMES, [0.5, 0.5])
    add('previous_plus_new_pair_equal', old_members + list(NEW_NAMES),
        [value / 3 for value in old_selection['weights']] + [1 / 3, 1 / 3],
        old_selection['temperatures'] + [1.0, 1.0])
    ranked = sorted(models, key=lambda name: probability_metrics(models[name]['probabilities'], labels)['accuracy'], reverse=True)
    add('top3_single_equal', ranked[:3], [1 / 3] * 3)
    selected = dict(max(comparison, key=lambda item: (item['accuracy'], -item['cross_entropy'])),
                    samples=len(labels), data_signature=signature,
                    previous_holdout_accuracy=baseline['validation_accuracy'],
                    holdout_gain=max(item['accuracy'] for item in comparison) - baseline['validation_accuracy'],
                    user_reported_previous_kaggle_accuracy=baseline['user_reported_kaggle_accuracy'],
                    target_kaggle_accuracy=0.8,
                    note='Exploratory selection on the reused holdout; candidate Kaggle score is unknown.')
    save_json(comparison, OUT / 'comparison.json')
    save_json(selected, OUT / 'selection.json')
    np.savez_compressed(OUT / 'validation_ensemble.npz', ids=ids, targets=labels,
                        probabilities=probability_arrays[selected['name']], class_names=np.asarray(CLASSES))
    print('SELECTED', selected['name'], selected['accuracy'], flush=True)
    if args.comparison_only:
        save_json(dict(stage='validation_complete', selection=selected), OUT / 'status.json')
        return
    if selected['accuracy'] <= baseline['validation_accuracy']:
        save_json(dict(stage='no_holdout_improvement', selection=selected), OUT / 'status.json')
        return

    test_x, _, test_ids = load_arrays(rows, DEFAULT_DATA, 'test', ROOT / 'outputs/data')
    test_ids = np.asarray(test_ids)
    member_probabilities, checkpoints = [], []
    old_artifacts = read_json(OLD / 'submission_validation.json')
    old_hashes = {item['run']: item['sha256'] for item in old_artifacts['checkpoints']}
    for name, use_tta in zip(selected['members'], selected['tta']):
        source = models[name]
        config = dict(source['config'], epochs=source['metrics']['best_epoch'], final_training=True)
        folder = (OUT if name in NEW_NAMES else OLD) / ('final_' + name)
        save_json(dict(stage='full_training', current_run=name, selection=selected), OUT / 'status.json')
        if name in NEW_NAMES:
            train_stage(folder, config, 'cuda')
        else:
            check_requested_config(read_json(folder / 'config.json'), config)
            assert completed_run_is_consistent(folder, config['epochs'], True)
        checkpoint = folder / 'final.pt'
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if name not in NEW_NAMES:
            assert checkpoint_hash == old_hashes[name]
        model, restored, state = load_checkpoint(checkpoint, 'cuda')
        assert state['epoch'] == config['epochs']
        probs = predict_probabilities(model, test_x, restored['model'], 'cuda', restored['batch_size'],
                                      use_tta, image_size=restored['preprocessing']['size'])
        probabilities_path = (folder / 'test_probabilities.npz' if name in NEW_NAMES
                              else OUT / ('reused_' + name + '_test_probabilities.npz'))
        np.savez_compressed(probabilities_path, ids=test_ids, probabilities=probs,
                            class_names=np.asarray(CLASSES))
        member_probabilities.append(probs)
        checkpoints.append(dict(run=name, checkpoint=str(checkpoint), epoch=state['epoch'],
                                sha256=checkpoint_hash, probabilities=str(probabilities_path),
                                tta=use_tta))
        del model, state
        gc.collect()
        torch.cuda.empty_cache()
    fused = fuse_probabilities(member_probabilities, selected['weights'], selected['temperatures'])
    np.savez_compressed(OUT / 'test_probabilities.npz', ids=test_ids, probabilities=fused,
                        class_names=np.asarray(CLASSES))
    output = ROOT / 'submission_kaggle80_candidate.csv'
    result = write_submission(DEFAULT_TEMPLATE, dict(zip(test_ids, fused.argmax(1))), output)
    assert result['rows'] == 7065
    result.update(selection=selected, checkpoints=checkpoints, submission=str(output),
                  submission_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), kaggle_score=None)
    for path, expected in baseline['original_submissions'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    save_json(result, OUT / 'submission_validation.json')
    save_json(dict(stage='candidate_ready', selection=selected, submission=result,
                   kaggle_score=None, target_achieved=None), OUT / 'status.json')
    print('CANDIDATE READY', output, flush=True)


if __name__ == '__main__':
    main()
