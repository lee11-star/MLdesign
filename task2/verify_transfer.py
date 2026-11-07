"""Independently recompute DINO probe/fusion artifacts and reload selected models."""
import csv
import gc
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from emotion_classifier.data import (CLASS_TO_LABEL, CLASSES, DEFAULT_DATA, DEFAULT_TEMPLATE,
                                     check_split, load_arrays, read_manifest, save_json)
from emotion_classifier.inference import load_checkpoint, predict_probabilities, validate_submission
from emotion_classifier.models import build_model, preprocess
from baseline import ROOT, digest, read_json

OUT = ROOT / 'outputs/round6'


def aligned(reference, path, key='probabilities', targets=None):
    with np.load(path) as data:
        assert list(data['class_names']) == CLASSES
        ids = list(data['ids'])
        assert len(set(ids)) == len(ids) and set(ids) == set(reference)
        position = {value: i for i, value in enumerate(ids)}
        order = [position[value] for value in reference]
        probabilities = data[key][order]
        if targets is not None:
            np.testing.assert_array_equal(data['targets'][order], targets)
    assert probabilities.shape == (len(reference), 6)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=2e-6, rtol=0)
    return probabilities


def blend(arrays, weights, temperatures=None):
    result = np.zeros_like(arrays[0], dtype=np.float64)
    if temperatures is None:
        temperatures = np.ones(len(arrays))
    for array, weight, temperature in zip(arrays, np.asarray(weights)/sum(weights), temperatures):
        adjusted = np.clip(np.asarray(array, dtype=np.float64), 1e-12, 1) ** (1/temperature)
        result += weight * adjusted / adjusted.sum(1, keepdims=True)
    return result


def scores(probabilities, labels):
    return dict(accuracy=float((probabilities.argmax(1) == labels).mean()),
                cross_entropy=float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()))


def main():
    assert Path.cwd().resolve() == ROOT
    torch.set_num_threads(4)
    rows = read_manifest(ROOT / 'outputs/data/manifest.csv')
    signature = digest(ROOT / 'outputs/data/manifest.csv')
    check_split(rows)
    for name, sha in read_json(OUT / 'original_submissions.json').items():
        assert digest(ROOT / name) == sha, name
    metadata = read_json(OUT / 'feature_metadata.json')
    assert metadata['data_signature'] == signature
    assert digest(OUT / 'frozen_features.npy') == metadata['sha256']
    assert metadata['feature_layout'] == 'last4_cls+last_patch_mean'
    eligible = [r for r in rows if r['split'] in ('train', 'val')]
    with np.load(OUT / 'feature_index.npz') as index:
        np.testing.assert_array_equal(index['ids'], [r['id'] for r in eligible])
        np.testing.assert_array_equal(index['targets'], [r['label'] for r in eligible])
        np.testing.assert_array_equal(index['splits'], [r['split'] for r in eligible])
    features = np.load(OUT / 'frozen_features.npy', mmap_mode='r', allow_pickle=False)
    assert list(features.shape) == metadata['shape']
    from emotion_classifier.dinov2 import DEFAULT_REPO, DEFAULT_WEIGHTS
    assert digest(DEFAULT_WEIGHTS) == metadata['pretrained_sha256']
    assert digest(DEFAULT_REPO / 'dinov2/models/vision_transformer.py') == metadata['backbone_source_sha256']
    # One real GPU prefix checks the cached feature extractor and both views.
    model = build_model('dinov2_vits14', DEFAULT_WEIGHTS, trainable_blocks=0).cuda().eval()
    all_x, _, _ = load_arrays(rows, DEFAULT_DATA, 'all', ROOT / 'outputs/data')
    batch = torch.from_numpy(all_x[:64]).cuda()
    with torch.inference_mode():
        for view in range(2):
            with torch.autocast('cuda'):
                actual = model.features(preprocess(batch, 'dinov2_vits14', flip=bool(view), image_size=224))
            np.testing.assert_allclose(actual.float().cpu().numpy(), features[:64, view], rtol=1e-4, atol=1e-4)
    del model, all_x, batch
    gc.collect()
    torch.cuda.empty_cache()
    val_mask = np.asarray([r['split'] == 'val' for r in eligible])
    vx, labels, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
    models = read_json(OUT / 'models.json')
    probes_checked = 0
    probe_numerics = {}
    for item in models.values():
        if item['kind'] != 'frozen':
            continue
        assert item['data_signature'] == signature and item['train_images'] == 22532
        assert item['scaler_fit_split'] == item['classifier_fit_split'] == 'train_only'
        assert digest(item['head']) == item['head_sha256']
        state = torch.load(item['head'], map_location='cpu', weights_only=True)
        weights, bias = state['1.weight'].numpy().astype(np.float64), state['1.bias'].numpy().astype(np.float64)
        with threadpool_limits(limits=4):
            logits = np.asarray(features[val_mask], dtype=np.float64) @ weights.T + bias
        probabilities = softmax(logits, axis=-1)
        for key, expected in [('probabilities', probabilities[:, 0]), ('flip_tta_probabilities', probabilities.mean(1))]:
            actual = aligned(ids, item['probabilities'], key, labels)
            # The stored head uses float32; offset cancellation differs from the
            # independent float64 reference by up to 2.86e-5 on these probes.
            np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-5)
            assert np.array_equal(actual.argmax(1), expected.argmax(1))
            probe_numerics[f"{item['name']}:{key}"] = dict(
                max_absolute_difference=float(np.abs(actual-expected).max()), argmax_differences=0)
        probes_checked += 1
    assert probes_checked == 3
    initial = read_json(OUT / 'finetune_initialization.json')
    assert initial['fit_split'] == 'train_only' and digest(initial['head']) == initial['sha256']
    run = OUT / 'dinov2_last4_nodropout'
    last = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    model, config, best = load_checkpoint(run / 'best.pt', 'cuda')
    assert last['epoch'] == config['epochs'] == config['scheduler_epochs'] == 30
    assert config['data_signature'] == signature and config['trainable_blocks'] == 4
    assert config['head_dropout'] == 0.0
    assert config['initial_head_signature'] == initial['sha256']
    assert last['best_epoch'] == best['epoch'] == best['best_epoch'] == models['dinov2_last4']['best_epoch']
    expected_best = max(last['history'], key=lambda row: row['val_accuracy'])['epoch']
    assert best['epoch'] == expected_best
    pretrained = torch.load(DEFAULT_WEIGHTS, map_location='cpu', weights_only=True)
    for name, value in best['model'].items():
        if name.startswith('backbone.'):
            original_name = name[len('backbone.'):]
            frozen = not original_name.startswith(('blocks.8.', 'blocks.9.', 'blocks.10.', 'blocks.11.', 'norm.'))
            if frozen:
                assert torch.equal(value, pretrained[original_name]), name
    for key, tta in [('probabilities', False), ('flip_tta_probabilities', True)]:
        expected = aligned(ids, models['dinov2_last4']['probabilities'], key, labels)
        actual = predict_probabilities(model, vx[:64], config['model'], 'cuda', 64, tta, 224)
        np.testing.assert_allclose(actual, expected[:64], rtol=1e-5, atol=1e-6)
    del model, last, best, pretrained
    gc.collect()
    torch.cuda.empty_cache()
    old = read_json(ROOT / 'outputs/round4/selection.json')
    sources = read_json(OUT / 'baseline_sources.json')
    assert [s['name'] for s in sources] == old['members']
    old_arrays = []
    for source in sources:
        assert digest(source['path']) == source['sha256']
        old_arrays.append(aligned(ids, source['path'], 'flip_tta_probabilities' if source['tta'] else 'probabilities', labels))
    baseline = blend(old_arrays, old['weights'], old['temperatures'])
    assert scores(baseline, labels)['accuracy'] == old['accuracy']
    comparison = read_json(OUT / 'comparison.json')
    assert len(comparison) == 21
    probabilities = {}
    for candidate in comparison:
        if candidate['source'] is None:
            computed = baseline
        else:
            item = models[candidate['source']]
            assert candidate['tta'] == item['selected_tta']
            key = 'flip_tta_probabilities' if candidate['tta'] else 'probabilities'
            new = aligned(ids, item['probabilities'], key, labels)
            computed = blend([baseline, new], [1-candidate['new_fraction'], candidate['new_fraction']])
        measured = scores(computed, labels)
        assert measured['accuracy'] == candidate['accuracy']
        np.testing.assert_allclose(measured['cross_entropy'], candidate['cross_entropy'], rtol=0, atol=1e-10)
        probabilities[candidate['name']] = computed
    winner = max(comparison, key=lambda item: (item['accuracy'], -item['cross_entropy']))
    selection = read_json(OUT / 'selection.json')
    assert all(selection[k] == v for k, v in winner.items())
    np.testing.assert_allclose(aligned(ids, OUT / 'validation_ensemble.npz', targets=labels), probabilities[winner['name']], rtol=0, atol=1e-10)
    improved = selection['accuracy'] > selection['baseline_accuracy']
    result = None
    if improved:
        result = read_json(OUT / 'submission_validation.json')
        assert result['selection'] == selection
        assert digest(result['checkpoint']) == result['checkpoint_sha256']
        assert digest(result['new_probabilities']) == result['new_probability_sha256']
        assert digest(result['submission']) == result['submission_sha256']
        model, full_config, full = load_checkpoint(result['checkpoint'], 'cuda')
        assert full_config['final_training'] and full_config['data_signature'] == signature
        head_fit = read_json(Path(result['checkpoint']).parent / 'head_fit.json')
        assert head_fit['fit_split'] == 'all_clean_training' and head_fit['images'] == 28165
        assert digest(full_config['initial_head']) == full_config['initial_head_signature'] == head_fit['head_sha256']
        selected = models[selection['source']]
        if selected['kind'] == 'finetuned':
            assert full['epoch'] == full_config['epochs'] == selected['best_epoch']
            assert full_config['scheduler_epochs'] == 30
        x, _, test_ids = load_arrays(rows, DEFAULT_DATA, 'test', ROOT / 'outputs/data')
        expected = aligned(test_ids, result['new_probabilities'])
        actual = predict_probabilities(model, x[:64], 'dinov2_vits14', 'cuda', 64, selected['selected_tta'], 224)
        np.testing.assert_allclose(actual, expected[:64], rtol=1e-5, atol=1e-6)
        old_arrays = []
        assert [s['name'] for s in result['old_probability_sources']] == old['members']
        for source in result['old_probability_sources']:
            assert digest(source['path']) == source['sha256']
            old_arrays.append(aligned(test_ids, source['path']))
        old_test = blend(old_arrays, old['weights'], old['temperatures'])
        fused = blend([old_test, expected], [1-selection['new_fraction'], selection['new_fraction']])
        np.testing.assert_allclose(aligned(test_ids, OUT / 'test_probabilities.npz'), fused, rtol=0, atol=1e-10)
        validate_submission(result['submission'], DEFAULT_TEMPLATE, expected_rows=7065)
        expected_labels = dict(zip(test_ids, fused.argmax(1)))
        with open(result['submission'], newline='', encoding='utf-8') as handle:
            assert all(int(row['Emotion']) == expected_labels[row['ID']] for row in csv.DictReader(handle))
    else:
        assert not (ROOT / 'submission_round6.csv').exists()
    status = dict(valid=True, probes_checked=probes_checked, fusion_candidates_checked=len(comparison),
                  completed_finetune_epochs=30, new_submission=improved,
                  baseline_accuracy=selection['baseline_accuracy'], selected_accuracy=selection['accuracy'],
                  original_submissions_preserved=True, frozen_feature_gpu_prefix=64,
                  validation_model_gpu_prefix=64, final_model_gpu_prefix=64 if improved else None,
                  submission_sha256=result['submission_sha256'] if result else None,
                  probe_float32_float64_comparison=probe_numerics)
    save_json(status, OUT / 'artifact_verification.json')
    save_json(dict(stage='complete', selection=selection, verification=status), OUT / 'status.json')
    print(f'Verified 3 frozen probes, 21 fusion candidates, 30 fine-tuning epochs; new submission={improved}', flush=True)


if __name__ == '__main__':
    main()
