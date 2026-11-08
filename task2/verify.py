"""Verify actual DAN experiments and immutable real-data provenance."""
import csv
import gc
from pathlib import Path

import numpy as np
import torch

from emotion_classifier.data import CLASS_TO_LABEL, CLASSES, DEFAULT_DATA, DEFAULT_TEMPLATE, check_split, load_arrays, read_manifest, save_json
from emotion_classifier.inference import load_checkpoint, predict_probabilities, validate_submission
from pipeline import OUT, MANIFEST, WEIGHTS, EXTRA
from baseline import ROOT, digest, read_json
from verify_generalization import verify_interval
from verify_transfer import aligned, blend, scores


def check_metrics(probabilities, labels, recorded):
    actual = scores(probabilities, labels)
    assert actual['accuracy'] == recorded['accuracy']
    np.testing.assert_allclose(actual['cross_entropy'], recorded['cross_entropy'], rtol=0, atol=2e-6)
    correct = probabilities.argmax(1) == labels
    for i, name in enumerate(CLASSES):
        assert recorded['per_class'][name] == correct[labels == i].mean()


def check_pair(old, new, labels, groups, recorded):
    check_metrics(new, labels, recorded)
    a, b = old.argmax(1) == labels, new.argmax(1) == labels
    assert recorded['corrected'] == int((~a & b).sum())
    assert recorded['broken'] == int((a & ~b).sum())
    assert recorded['net_correct'] == int(b.sum() - a.sum())
    verify_interval(old, new, labels, groups, recorded['group_bootstrap_95pct_delta_accuracy'])


def verify_model(entry, expected_model, manifest):
    assert entry['model'] == expected_model and entry['data_signature'] == digest(manifest)
    assert digest(entry['checkpoint']) == entry['checkpoint_sha256']
    assert digest(entry['probabilities']) == entry['probability_sha256']
    rows = read_manifest(manifest)
    check_split(rows)
    x, labels, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
    model, config, state = load_checkpoint(entry['checkpoint'], 'cuda')
    assert state['epoch'] == config['epochs'] == config['scheduler_epochs'] == 30
    assert len(state['history']) == 30 and not config['final_training']
    assert config['data_signature'] == digest(manifest)
    assert config['pretrained_sha256'] == digest(WEIGHTS)
    assert config['image_size'] == 224 and config['batch_size'] == 64
    assert config['seed'] == 42 and config['lr'] == 3e-4 and config['backbone_lr_scale'] == 0.1
    assert config['mixup_alpha'] == config['label_smoothing'] == config['head_dropout'] == 0
    assert config['geometry'] == 'standard'
    parameter_groups = state['optimizer']['param_groups']
    assert [g['name'] for g in parameter_groups] == ['head', 'backbone']
    if expected_model == 'dan_confidence':
        assert config['confidence_fraction'] == 0.7 and config['confidence_margin'] == 0.15
        assert config['confidence_regularization'] == 1.0
        assert len(parameter_groups[0]['params']) == len(list(model.classifier.parameters())) + len(list(model.confidence.parameters()))
        for row in state['history']:
            assert np.isfinite(row['train_confidence_rank_loss']) and row['train_confidence_rank_loss'] >= 0
            np.testing.assert_allclose(row['train_loss'], row['train_classification_loss'] + row['train_confidence_rank_loss'], rtol=1e-5, atol=1e-6)
    for key, flip, metrics in (('probabilities', False, 'ordinary'), ('flip_tta_probabilities', True, 'flip')):
        expected = aligned(ids, entry['probabilities'], key, labels)
        check_metrics(expected, labels, entry[metrics])
        actual = predict_probabilities(model, x[:64], expected_model, 'cuda', 64, tta=flip, image_size=224)
        np.testing.assert_allclose(actual, expected[:64], rtol=1e-5, atol=1e-6)
    expected_best = max(state['history'], key=lambda r: r['val_accuracy'])['epoch']
    assert expected_best == entry['descriptive_best']['best_epoch']
    del model, state, x
    gc.collect()
    torch.cuda.empty_cache()
    return dict(checkpoint=entry['checkpoint'], sha256=entry['checkpoint_sha256'], gpu_prefix=64)


def main():
    assert Path.cwd().resolve() == ROOT
    torch.set_num_threads(4)
    contract = read_json(OUT / 'data_contract.json')
    assert contract['raw_images_checked'] == 28165
    assert contract['all_real_images_match_cache'] and contract['original_labels_and_folder_names_match']
    assert digest(MANIFEST) == contract['manifest_sha256']
    assert digest(contract['cache']) == contract['cache_sha256']
    assert digest(WEIGHTS) == contract['source_parameters']['sha256']
    for name, sha in contract['source_files'].items():
        assert digest(ROOT / name) == sha
    rows = read_manifest(MANIFEST)
    check_split(rows)
    for row in rows:
        if row['split'] in ('train', 'val'):
            assert row['kind'] == 'train' and row['label'] == CLASS_TO_LABEL[Path(row['path']).parent.name]
    val = [r for r in rows if r['split'] == 'val']
    ids, labels, groups = [r['id'] for r in val], np.asarray([r['label'] for r in val]), [r['sha256'] for r in val]
    entries = read_json(OUT / 'models.json')
    assert set(entries) == {'dan_plain', 'dan_confidence'}
    checked = [verify_model(entries['dan_plain'], 'dan', MANIFEST),
               verify_model(entries['dan_confidence'], 'dan_confidence', MANIFEST)]
    configs = [read_json(Path(entries[n]['folder']) / 'config.json') for n in ('dan_plain', 'dan_confidence')]
    ignored = {'model', 'confidence_fraction', 'confidence_margin', 'confidence_regularization'}
    assert {k: v for k, v in configs[0].items() if k not in ignored} == {
        k: v for k, v in configs[1].items() if k not in ignored}
    previous = read_json(ROOT / 'outputs/round4/selection.json')
    old_arrays = []
    for source in read_json(OUT / 'baseline_sources.json'):
        assert digest(source['path']) == source['sha256']
        old_arrays.append(aligned(ids, source['path'], source['key'], labels))
    old = blend(old_arrays, previous['weights'], previous['temperatures'])
    comparison = read_json(OUT / 'comparison.json')
    assert len(comparison) == 5
    assert {(r['source'], r['fraction']) for r in comparison} == {
        (None, 0.0), ('dan_plain', 0.25), ('dan_plain', 1.0),
        ('dan_confidence', 0.25), ('dan_confidence', 1.0)}
    probabilities = {}
    for candidate in comparison:
        if candidate['source'] is None:
            p = old
            check_metrics(p, labels, candidate)
        else:
            assert candidate['fraction'] in (0.25, 1.0)
            new = aligned(ids, entries[candidate['source']]['probabilities'], 'flip_tta_probabilities', labels)
            p = blend([old, new], [1 - candidate['fraction'], candidate['fraction']])
            check_pair(old, p, labels, groups, candidate)
        probabilities[candidate['name']] = p
    winner = max(comparison, key=lambda r: (r['accuracy'], -r['cross_entropy']))
    selection = read_json(OUT / 'selection.json')
    assert all(selection[k] == v for k, v in winner.items())
    np.testing.assert_allclose(aligned(ids, OUT / 'validation_ensemble.npz', targets=labels), probabilities[winner['name']], rtol=0, atol=1e-10)
    plain = aligned(ids, entries['dan_plain']['probabilities'], 'flip_tta_probabilities', labels)
    confidence = aligned(ids, entries['dan_confidence']['probabilities'], 'flip_tta_probabilities', labels)
    check_pair(plain, confidence, labels, groups, read_json(OUT / 'confidence_ablation.json'))
    replicated, final_prefix = False, None
    if winner['accuracy'] >= previous['accuracy'] + 0.005:
        extra_rows = read_manifest(EXTRA)
        assert [{k: v for k, v in r.items() if k != 'split'} for r in rows] == [
            {k: v for k, v in r.items() if k != 'split'} for r in extra_rows]
        assert set(groups).isdisjoint(r['sha256'] for r in extra_rows if r['split'] == 'val')
        for original, extra_row in zip(rows, extra_rows):
            if original['split'] in ('test', 'excluded'):
                assert extra_row['split'] == original['split']
        repetition = read_json(OUT / 'replication.json')
        assert repetition['fraction'] == winner['fraction']
        source = entries[winner['source']]
        checked.append(verify_model(repetition['extra_model'], source['model'], EXTRA))
        for fold, manifest, entry in (('original', MANIFEST, source), ('extra', EXTRA, repetition['extra_model'])):
            vr = [r for r in read_manifest(manifest) if r['split'] == 'val']
            vi, vy, vg = [r['id'] for r in vr], np.asarray([r['label'] for r in vr]), [r['sha256'] for r in vr]
            pair = repetition[fold]
            assert digest(pair['reference_path']) == pair['reference_sha256']
            a = aligned(vi, pair['reference_path'], 'probabilities', vy)
            b = aligned(vi, entry['probabilities'], 'flip_tta_probabilities', vy)
            check_metrics(b, vy, pair['new_single'])
            mixed = blend([a, b], [1 - winner['fraction'], winner['fraction']])
            check_pair(a, mixed, vy, vg, pair['candidate'])
        assert repetition['positive_lower_bounds_both_folds'] == all(
            repetition[k]['candidate']['group_bootstrap_95pct_delta_accuracy'][0] > 0 for k in ('original', 'extra'))
        replicated = True
    status = read_json(OUT / 'status.json')
    if status['new_submission']:
        assert replicated and repetition['positive_lower_bounds_both_folds']
        result = read_json(OUT / 'submission_validation.json')
        assert result['selection'] == winner
        assert digest(result['checkpoint']) == result['checkpoint_sha256']
        assert digest(result['submission']) == result['submission_sha256']
        assert digest(result['new_probabilities']) == result['new_probability_sha256']
        model, config, state = load_checkpoint(result['checkpoint'], 'cuda')
        assert config['final_training'] and state['epoch'] == 30
        assert config['model'] == entries[winner['source']]['model']
        assert config['data_signature'] == digest(MANIFEST) and config['pretrained_sha256'] == digest(WEIGHTS)
        tx, _, ti = load_arrays(rows, DEFAULT_DATA, 'test', ROOT / 'outputs/data')
        new = aligned(ti, result['new_probabilities'])
        actual = predict_probabilities(model, tx[:64], config['model'], 'cuda', 64, tta=True, image_size=224)
        np.testing.assert_allclose(actual, new[:64], rtol=1e-5, atol=1e-6)
        old_test_arrays = []
        for source in result['previous_probability_sources']:
            assert digest(source['path']) == source['sha256']
            old_test_arrays.append(aligned(ti, source['path']))
        old_test = blend(old_test_arrays, previous['weights'], previous['temperatures'])
        fused = blend([old_test, new], [1 - winner['fraction'], winner['fraction']])
        np.testing.assert_allclose(aligned(ti, OUT / 'test_probabilities.npz'), fused, rtol=0, atol=1e-10)
        validate_submission(result['submission'], DEFAULT_TEMPLATE, expected_rows=7065)
        mapping = dict(zip(ti, fused.argmax(1)))
        with open(result['submission'], newline='', encoding='utf-8') as handle:
            assert all(int(r['Emotion']) == mapping[r['ID']] for r in csv.DictReader(handle))
        final_prefix = 64
    else:
        assert not (ROOT / 'submission_round8.csv').exists()
    for name, sha in read_json(OUT / 'original_submissions.json').items():
        assert digest(ROOT / name) == sha
    result = dict(valid=True, real_training_validation_images=28165, original_manifest_labels_cache_preserved=True,
                  additional_training_datasets=0, generated_training_images=0, pseudo_labels=0,
                  transferred_parameter_source=contract['source_parameters'],
                  models_checked=checked, fixed_candidates_checked=5, extra_fold_verified=replicated,
                  new_submission=status['new_submission'], final_gpu_prefix=final_prefix,
                  original_submissions_preserved=True, selected_accuracy=winner['accuracy'], kaggle_score=None)
    save_json(result, OUT / 'artifact_verification.json')
    save_json(dict(status, stage='complete', verification=result), OUT / 'status.json')
    print({k: v for k, v in result.items() if k not in ('models_checked', 'transferred_parameter_source')}, flush=True)


if __name__ == '__main__':
    main()
