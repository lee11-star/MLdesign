"""Verify actual fold, checkpoint, prediction and error-audit artifacts."""
import csv
import gc

import numpy as np
import torch

from analysis import OUT
from emotion_classifier.data import CLASSES, DEFAULT_DATA, check_split, load_arrays, read_manifest, save_json
from emotion_classifier.inference import load_checkpoint, predict_probabilities
from generalization import BASE, ORIGINAL_MANIFEST
from baseline import ROOT, digest, read_json


def verify_interval(baseline, candidate, labels, groups, recorded):
    """Scalar resampling reference for the vectorized analysis implementation."""
    old_ok, new_ok = baseline.argmax(1) == labels, candidate.argmax(1) == labels
    grouped = {}
    for group, a, b in zip(groups, old_ok, new_ok):
        delta, size = grouped.get(group, (0, 0))
        grouped[group] = (delta + int(b) - int(a), size + 1)
    values = np.asarray([grouped[key] for key in sorted(grouped)])
    rng, estimates = np.random.default_rng(42), []
    for _ in range(2000):
        sampled = values[rng.integers(len(values), size=len(values))].sum(0)
        estimates.append(sampled[0] / sampled[1])
    np.testing.assert_allclose(np.quantile(estimates, [0.025, 0.975]), recorded, rtol=0, atol=1e-12)


def verify_dino_and_audit():
    from verify_transfer import aligned, blend, scores
    rows = [r for r in read_manifest(ORIGINAL_MANIFEST) if r['split'] == 'val']
    ids, labels, groups = [r['id'] for r in rows], np.asarray([r['label'] for r in rows]), [r['sha256'] for r in rows]
    old_selection = read_json(ROOT / 'outputs/round4/selection.json')
    sources = read_json(ROOT / 'outputs/round6/baseline_sources.json')
    old_arrays = []
    for source in sources:
        assert digest(source['path']) == source['sha256']
        old_arrays.append(aligned(ids, source['path'], 'flip_tta_probabilities' if source['tta'] else 'probabilities', labels))
    baseline = blend(old_arrays, old_selection['weights'], old_selection['temperatures'])
    report = read_json(OUT / 'dino_comparison.json')
    models = read_json(ROOT / 'outputs/round6/models.json')
    assert len(report['comparisons']) == 40
    assert len({(r['model'], r['view'], r['fraction']) for r in report['comparisons']}) == 40
    for source in report['sources']:
        assert digest(source['path']) == source['sha256']
    for entry in report['comparisons']:
        new = aligned(ids, models[entry['model']]['probabilities'], entry['view'], labels)
        probs = blend([baseline, new], [1 - entry['fraction'], entry['fraction']])
        actual = scores(probs, labels)
        assert actual['accuracy'] == entry['accuracy']
        np.testing.assert_allclose(actual['cross_entropy'], entry['cross_entropy'], rtol=0, atol=1e-10)
        a, b = baseline.argmax(1) == labels, probs.argmax(1) == labels
        assert entry['corrected'] == int((~a & b).sum()) and entry['broken'] == int((a & ~b).sum())
    selected = report['best_exploratory']
    new = aligned(ids, models[selected['model']]['probabilities'], selected['view'], labels)
    mixed = blend([baseline, new], [1 - selected['fraction'], selected['fraction']])
    verify_interval(baseline, mixed, labels, groups, selected['paired']['group_bootstrap_95pct_delta_accuracy'])
    dino = aligned(ids, models['dinov2_last4']['probabilities'], 'flip_tta_probabilities', labels)
    verify_interval(baseline, dino, labels, groups, report['finetune_flip']['group_bootstrap_95pct_delta_accuracy'])
    with (OUT / 'error_audit_reviewed.csv').open(newline='', encoding='utf-8-sig') as handle:
        audit = list(csv.DictReader(handle))
    rng, expected = np.random.default_rng(42), []
    for label, count in ((1, 34), (3, 33), (0, 33)):
        unique = {}
        for i, row in enumerate(rows):
            if labels[i] == label and baseline[i].argmax() != label:
                unique.setdefault(row['sha256'], i)
        expected.extend(sorted(rng.choice(list(unique.values()), count, replace=False).tolist()))
    assert [r['id'] for r in audit] == [ids[i] for i in expected]
    assert [r['path'] for r in audit] == [rows[i]['path'] for i in expected]
    assert [r['sha256'] for r in audit] == [groups[i] for i in expected]
    assert [r['baseline_prediction'] for r in audit] == [CLASSES[baseline[i].argmax()] for i in expected]
    return 40


def main():
    torch.set_num_threads(4)
    original, extra = read_manifest(ORIGINAL_MANIFEST), read_manifest(OUT / 'fold1_manifest.csv')
    check_split(original)
    check_split(extra)
    assert [{k: v for k, v in r.items() if k != 'split'} for r in original] == [
        {k: v for k, v in r.items() if k != 'split'} for r in extra]
    assert {r['sha256'] for r in original if r['split'] == 'val'}.isdisjoint(
        {r['sha256'] for r in extra if r['split'] == 'val'})
    for a, b in zip(original, extra):
        if a['split'] in ('test', 'excluded'):
            assert a['split'] == b['split']
    result = read_json(OUT / 'geometry_comparison.json')
    configs, checked_models = {}, []
    for name, folder, manifest, probability_file in (
            ('fold0_standard', BASE, ORIGINAL_MANIFEST, OUT / 'fold0_standard_epoch30_flip.npz'),
            ('fold0_flip_only', OUT / 'fold0_flip_only', ORIGINAL_MANIFEST, OUT / 'fold0_flip_only/epoch30_flip.npz'),
            ('fold1_standard', OUT / 'fold1_standard', OUT / 'fold1_manifest.csv', OUT / 'fold1_standard/epoch30_flip.npz'),
            ('fold1_flip_only', OUT / 'fold1_flip_only', OUT / 'fold1_manifest.csv', OUT / 'fold1_flip_only/epoch30_flip.npz')):
        rows = read_manifest(manifest)
        vx, labels, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
        model, config, state = load_checkpoint(folder / 'last.pt', 'cuda')
        assert config['data_signature'] == digest(manifest)
        assert state['epoch'] == config['epochs'] == config['scheduler_epochs'] == 30
        assert len(state['history']) == 30
        assert not config['final_training']
        configs[name] = config
        with np.load(probability_file) as data:
            assert list(data['class_names']) == CLASSES
            np.testing.assert_array_equal(data['ids'], ids)
            np.testing.assert_array_equal(data['targets'], labels)
            probs = data['probabilities']
        assert probs.shape == (len(labels), 6) and np.isfinite(probs).all() and (probs >= 0).all()
        np.testing.assert_allclose(probs.sum(1), 1, rtol=0, atol=2e-6)
        actual = predict_probabilities(model, vx[:128], config['model'], 'cuda', 128, tta=True, image_size=96)
        np.testing.assert_allclose(actual, probs[:128], rtol=1e-5, atol=1e-6)
        checked_models.append(dict(name=name, checkpoint=str(folder / 'last.pt'),
                                   sha256=digest(folder / 'last.pt'),
                                   probability_file=str(probability_file), probability_sha256=digest(probability_file),
                                   data_signature=config['data_signature']))
        del model, state, vx
        gc.collect()
        torch.cuda.empty_cache()
    ignored = {'geometry', 'preprocessing', 'data_signature'}
    reference = {k: v for k, v in configs['fold0_standard'].items() if k not in ignored}
    for name, config in configs.items():
        assert {k: v for k, v in config.items() if k not in ignored} == reference, name
        aug = config['preprocessing']['training_augmentation']
        assert aug['horizontal_flip'] == 0.5
        if name.endswith('flip_only'):
            assert config['geometry'] == 'flip_only'
            assert aug['rotation_degrees'] == aug['translation_fraction'] == 0
            assert aug['scale'] == [1.0, 1.0]
        else:
            assert config.get('geometry', 'standard') == 'standard'
            assert aug['rotation_degrees'] == 10 and aug['translation_fraction'] == 0.08
            assert aug['scale'] == [0.9, 1.1]
    for fold in ('fold0', 'fold1'):
        section = result[fold]
        arrays = []
        for source in section['sources']:
            assert digest(source['path']) == source['sha256']
            with np.load(source['path']) as data:
                arrays.append(data['probabilities'])
                labels = data['targets']
        a, b = arrays[0].argmax(1) == labels, arrays[1].argmax(1) == labels
        expected = section['candidate']
        assert expected['corrected'] == int((~a & b).sum())
        assert expected['broken'] == int((a & ~b).sum())
        assert expected['net_correct'] == int(b.sum() - a.sum())
        assert expected['accuracy'] == b.mean() and section['baseline']['accuracy'] == a.mean()
        for i, label in enumerate(CLASSES):
            assert expected['per_class'][label] == b[labels == i].mean()
        val_rows = [r for r in (original if fold == 'fold0' else extra) if r['split'] == 'val']
        verify_interval(arrays[0], arrays[1], labels, [r['sha256'] for r in val_rows],
                        expected['group_bootstrap_95pct_delta_accuracy'])
    for name, sha in read_json(OUT / 'original_submissions.json').items():
        assert digest(ROOT / name) == sha, name
    with (OUT / 'error_audit_reviewed.csv').open(newline='', encoding='utf-8-sig') as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == len({r['sha256'] for r in audit}) == 100
    assert all(r['observation'] and r['review_tags'] for r in audit)
    assert [int(r['number']) for r in audit] == list(range(1, 101))
    from collections import Counter
    assert Counter(r['true_class'] for r in audit) == dict(Fear=34, Sad=33, Angry=33)
    dino_comparisons = verify_dino_and_audit()
    verification = dict(valid=True, group_leakage=False, extra_fold_validation_disjoint=True,
                        checkpoints_verified=4, gpu_prefix_per_checkpoint=128, epochs_per_run=30,
                        only_changed_training_factor='geometric_augmentation',
                        dino_both_view_candidates_verified=dino_comparisons,
                        hash_group_intervals_recomputed=4,
                        checkpoints=checked_models,
                        reviewed_unique_error_images=100, original_submissions_preserved=True,
                        new_submission=False)
    save_json(verification, OUT / 'artifact_verification.json')
    save_json(dict(stage='complete', verification=verification), OUT / 'status.json')
    print(verification, flush=True)


if __name__ == '__main__':
    main()
