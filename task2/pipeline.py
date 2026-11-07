"""Real-image-only DAN transfer, one confidence ablation, and gated replication."""
import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from analysis import paired_metrics
from emotion_classifier.data import (CLASS_TO_LABEL, CLASSES, DEFAULT_DATA, DEFAULT_TEMPLATE,
                                     check_split, load_arrays, read_manifest, save_json)
from emotion_classifier.ensemble import align_probabilities, fuse_probabilities
from emotion_classifier.inference import load_checkpoint, predict_probabilities, write_submission
from emotion_classifier.training import train_run, recover_selected_artifact, save_run_metadata
from evaluate import run_evaluation
from model_selection import probability_metrics
from baseline import ROOT, digest, read_json

OUT = ROOT / 'outputs/round8'
MANIFEST = ROOT / 'outputs/data/manifest.csv'
WEIGHTS = OUT / 'pretrained/dan_affectnet7.pth'
EXTRA = ROOT / 'outputs/round7/fold1_manifest.csv'


def audit_real_data():
    rows = read_manifest(MANIFEST)
    check_split(rows)
    previous = OUT / 'data_contract.json'
    if previous.exists():
        recorded = read_json(previous)
        assert digest(MANIFEST) == recorded['manifest_sha256']
        assert digest(ROOT / 'outputs/data/pixels.npy') == recorded['cache_sha256']
        assert digest(WEIGHTS) == recorded['source_parameters']['sha256']
        for name, sha in recorded['source_files'].items():
            assert digest(OUT / 'sources' / name) == sha
        return rows
    cache = np.load(ROOT / 'outputs/data/pixels.npy', mmap_mode='r', allow_pickle=False)
    checked = 0
    for row in rows:
        if row['split'] not in ('train', 'val'):
            continue
        path = ROOT / DEFAULT_DATA / row['path']
        assert row['kind'] == 'train' and row['id'] == path.name
        assert CLASS_TO_LABEL[path.parent.name] == row['label']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], str(path)
        with Image.open(path) as image:
            assert image.size == (48, 48) and image.mode == 'L'
            np.testing.assert_array_equal(np.asarray(image), cache[row['cache_index'], 0])
        checked += 1
    result = dict(manifest=str(MANIFEST), manifest_sha256=digest(MANIFEST),
                  cache=str(ROOT / 'outputs/data/pixels.npy'), cache_sha256=digest(ROOT / 'outputs/data/pixels.npy'),
                  raw_images_checked=checked, split_counts=dict(Counter(r['split'] for r in rows)),
                  original_labels_and_folder_names_match=True, all_real_images_match_cache=True,
                  training_data='Existing task2 images only; no external, synthetic or pseudo-labeled images.',
                  source_parameters=dict(path=str(WEIGHTS), sha256=digest(WEIGHTS),
                      source='https://github.com/yaoing/DAN',
                      download='https://drive.google.com/file/d/1_Z-U7rT5NJ3Vc73aN2ZBmuvCkzUQG4jT/view',
                      task='Official AffectNet7 parameter checkpoint; source images not downloaded.',
                      source_head='Discarded; fresh six-class head fitted only on real task training split.'),
                  source_files={p.name: digest(p) for p in (OUT / 'sources').iterdir() if p.is_file()})
    assert checked == 28165 and result['split_counts']['train'] == 22532 and result['split_counts']['val'] == 5633
    save_json(result, previous)
    return rows


def configuration(model, manifest, final=False):
    config = dict(model=model, epochs=30, scheduler_epochs=30, batch_size=64, image_size=224,
                  lr=3e-4, backbone_lr_scale=0.1, weight_decay=1e-4, seed=42, device='cuda', num_workers=0,
                  pretrained=str(WEIGHTS), pretrained_sha256=digest(WEIGHTS),
                  label_smoothing=0.0, mixup_alpha=0.0, mixup_probability=0.5, head_dropout=0.0,
                  geometry='standard', final_training=final, data_signature=digest(manifest))
    if model == 'dan_confidence':
        config.update(confidence_fraction=0.7, confidence_margin=0.15, confidence_regularization=1.0)
    return config


def fit(name, model, manifest=MANIFEST, final=False):
    folder = OUT / name
    config, last = configuration(model, manifest, final), folder / 'last.pt'
    complete = False
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        assert all(state['config'].get(k) == v for k, v in config.items()), name
        complete = state['epoch'] == config['epochs']
        if complete:
            recover_selected_artifact(state, folder, not final)
            save_run_metadata(state, folder, not final)
        del state
    if not complete:
        rows = read_manifest(manifest)
        check_split(rows)
        x, y, _ = load_arrays(rows, DEFAULT_DATA, 'all' if final else 'train', ROOT / 'outputs/data')
        vx, vy = (None, None) if final else load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')[:2]
        train_run(config, x, y, vx, vy, folder, resume=last if last.exists() else None)
        del x, y, vx, vy
        gc.collect()
        torch.cuda.empty_cache()
    return folder


def evaluate_primary(folder, manifest=MANIFEST):
    # Ordinary-best checkpoint is descriptive; selection uses fixed epoch30 flip TTA.
    descriptive = run_evaluation(folder / 'best.pt', manifest=manifest, batch_size=64)
    rows = read_manifest(manifest)
    x, labels, ids = load_arrays(rows, DEFAULT_DATA, 'val', ROOT / 'outputs/data')
    model, config, state = load_checkpoint(folder / 'last.pt', 'cuda')
    assert state['epoch'] == 30
    base = predict_probabilities(model, x, config['model'], 'cuda', 64, image_size=224)
    flip = predict_probabilities(model, x, config['model'], 'cuda', 64, tta=True, image_size=224)
    path = folder / 'primary_probabilities.npz'
    np.savez_compressed(path, ids=np.asarray(ids), targets=labels, probabilities=base,
                        flip_tta_probabilities=flip, class_names=np.asarray(CLASSES))
    result = dict(model=config['model'], folder=str(folder), checkpoint=str(folder / 'last.pt'),
                  checkpoint_sha256=digest(folder / 'last.pt'), probabilities=str(path), probability_sha256=digest(path),
                  manifest=str(manifest), data_signature=digest(manifest), epoch=30,
                  ordinary=probability_metrics(base, labels), flip=probability_metrics(flip, labels),
                  descriptive_best=descriptive)
    save_json(result, folder / 'primary_metrics.json')
    del model, state, x
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aligned(path, ids, labels=None, key='flip_tta_probabilities'):
    with np.load(path) as data:
        assert list(data['class_names']) == CLASSES
        return align_probabilities(ids, data['ids'], data[key], labels,
                                   data['targets'] if labels is not None else None)


def baseline(ids, labels):
    selected = read_json(ROOT / 'outputs/round4/selection.json')
    models = read_json(ROOT / 'outputs/round4/models.json')
    arrays, sources = [], []
    for name, flip in zip(selected['members'], selected['tta']):
        path = Path(models[name]['folder']) / 'evaluation/validation_probabilities.npz'
        key = 'flip_tta_probabilities' if flip else 'probabilities'
        arrays.append(aligned(path, ids, labels, key))
        sources.append(dict(name=name, path=str(path), sha256=digest(path), key=key))
    result = fuse_probabilities(arrays, selected['weights'], selected['temperatures'])
    assert probability_metrics(result, labels)['accuracy'] == selected['accuracy']
    save_json(sources, OUT / 'baseline_sources.json')
    return result


def compare(entries, rows):
    val = [r for r in rows if r['split'] == 'val']
    ids, labels, groups = [r['id'] for r in val], np.asarray([r['label'] for r in val]), [r['sha256'] for r in val]
    old = baseline(ids, labels)
    candidates = [dict(name='round4', source=None, fraction=0.0, **probability_metrics(old, labels))]
    arrays = {'round4': old}
    for name, entry in entries.items():
        new = aligned(entry['probabilities'], ids, labels)
        for fraction in (0.25, 1.0):
            p = fuse_probabilities([old, new], [1 - fraction, fraction])
            candidate_name = f'{name}_{fraction:g}'
            candidates.append(dict(name=candidate_name, source=name, fraction=fraction,
                                   **paired_metrics(old, p, labels, groups)))
            arrays[candidate_name] = p
    winner = max(candidates, key=lambda r: (r['accuracy'], -r['cross_entropy']))
    np.savez_compressed(OUT / 'validation_ensemble.npz', ids=np.asarray(ids), targets=labels,
                        probabilities=arrays[winner['name']], class_names=np.asarray(CLASSES))
    save_json(candidates, OUT / 'comparison.json')
    save_json(dict(winner, baseline_accuracy=candidates[0]['accuracy'], primary_endpoint='epoch30_flip_tta',
                   exploratory_reused_holdout=True, kaggle_score=None), OUT / 'selection.json')
    a = aligned(entries['dan_plain']['probabilities'], ids, labels)
    b = aligned(entries['dan_confidence']['probabilities'], ids, labels)
    save_json(paired_metrics(a, b, labels, groups), OUT / 'confidence_ablation.json')
    return winner, candidates[0]['accuracy']


def replicate(model, primary, fraction):
    folder = fit('extra_' + model, model, EXTRA)
    extra = evaluate_primary(folder, EXTRA)
    results = {}
    for name, manifest, new_entry, old_path in (
            ('original', MANIFEST, primary, ROOT / 'outputs/round7/fold0_standard_epoch30_flip.npz'),
            ('extra', EXTRA, extra, ROOT / 'outputs/round7/fold1_standard/epoch30_flip.npz')):
        val = [r for r in read_manifest(manifest) if r['split'] == 'val']
        ids, labels = [r['id'] for r in val], np.asarray([r['label'] for r in val])
        old = aligned(old_path, ids, labels, key='probabilities')
        new = aligned(new_entry['probabilities'], ids, labels)
        mixed = fuse_probabilities([old, new], [1 - fraction, fraction])
        results[name] = dict(reference=probability_metrics(old, labels),
                             candidate=paired_metrics(old, mixed, labels, [r['sha256'] for r in val]),
                             new_single=probability_metrics(new, labels),
                             reference_path=str(old_path), reference_sha256=digest(old_path))
    results['extra_model'] = extra
    results['positive_lower_bounds_both_folds'] = all(
        results[k]['candidate']['group_bootstrap_95pct_delta_accuracy'][0] > 0 for k in ('original', 'extra'))
    results['fraction'] = fraction
    results['scope'] = 'Fixed-fraction new model plus ConvNeXt reference versus ConvNeXt; not a seven-model ensemble replication.'
    save_json(results, OUT / 'replication.json')
    return results


def publish_local(selection, entry):
    folder = fit('final_' + entry['model'], entry['model'], final=True)
    rows = read_manifest(MANIFEST)
    x, _, ids = load_arrays(rows, DEFAULT_DATA, 'test', ROOT / 'outputs/data')
    model, config, state = load_checkpoint(folder / 'final.pt', 'cuda')
    assert config['final_training'] and config['model'] == entry['model']
    assert state['epoch'] == config['epochs'] == 30 and config['data_signature'] == digest(MANIFEST)
    new = predict_probabilities(model, x, config['model'], 'cuda', 64, tta=True, image_size=224)
    np.savez_compressed(folder / 'test_probabilities.npz', ids=np.asarray(ids), probabilities=new,
                        class_names=np.asarray(CLASSES))
    previous = read_json(ROOT / 'outputs/round4/selection.json')
    old_artifacts = read_json(ROOT / 'outputs/round4/submission_validation.json')
    by_name = {r['run']: r for r in old_artifacts['checkpoints']}
    old_arrays, sources = [], []
    for name in previous['members']:
        source = by_name[name]
        assert digest(source['checkpoint']) == source['sha256']
        old_arrays.append(aligned(source['probabilities'], ids, key='probabilities'))
        sources.append(dict(path=source['probabilities'], sha256=digest(source['probabilities'])))
    old = fuse_probabilities(old_arrays, previous['weights'], previous['temperatures'])
    final = fuse_probabilities([old, new], [1 - selection['fraction'], selection['fraction']])
    np.savez_compressed(OUT / 'test_probabilities.npz', ids=np.asarray(ids), probabilities=final,
                        class_names=np.asarray(CLASSES))
    submission = ROOT / 'submission_round8.csv'
    result = write_submission(DEFAULT_TEMPLATE, dict(zip(ids, final.argmax(1))), submission)
    result.update(submission=str(submission), submission_sha256=digest(submission), checkpoint=str(folder / 'final.pt'),
                  checkpoint_sha256=digest(folder / 'final.pt'), epoch=state['epoch'],
                  new_probabilities=str(folder / 'test_probabilities.npz'),
                  new_probability_sha256=digest(folder / 'test_probabilities.npz'),
                  previous_probability_sources=sources, selection=selection, kaggle_score=None)
    save_json(result, OUT / 'submission_validation.json')
    del model, state, x
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = audit_real_data()
    if args.audit_only:
        print('Audited all28165 real training/validation images, original labels and cache.', flush=True)
        return
    original_path = OUT / 'original_submissions.json'
    if not original_path.exists():
        save_json({p.name: digest(p) for p in ROOT.glob('submission*.csv') if p.name != 'submission_round8.csv'}, original_path)
    for name, sha in read_json(original_path).items():
        assert digest(ROOT / name) == sha, name
    entries = {}
    for name, model in (('dan_plain', 'dan'), ('dan_confidence', 'dan_confidence')):
        save_json(dict(stage='training', run=name), OUT / 'status.json')
        folder = fit(name, model)
        entries[name] = evaluate_primary(folder)
        save_json(entries, OUT / 'models.json')
    selection, old_accuracy = compare(entries, rows)
    replication, submission = None, None
    if selection['accuracy'] >= old_accuracy + 0.005:
        chosen = entries[selection['source']]
        save_json(dict(stage='extra_fold', model=chosen['model']), OUT / 'status.json')
        replication = replicate(chosen['model'], chosen, selection['fraction'])
        if replication['positive_lower_bounds_both_folds'] and not args.validation_only:
            save_json(dict(stage='full_training', model=chosen['model']), OUT / 'status.json')
            submission = publish_local(selection, chosen)
    for name, sha in read_json(original_path).items():
        assert digest(ROOT / name) == sha, name
    assert digest(MANIFEST) == read_json(OUT / 'data_contract.json')['manifest_sha256']
    save_json(dict(stage='candidate_ready' if submission else 'validation_complete',
                   selection=selection, extra_fold_completed=replication is not None,
                   new_submission=submission is not None, original_submissions_preserved=True), OUT / 'status.json')
    print(dict(selection=selection, extra_fold=replication is not None, new_submission=submission is not None), flush=True)


if __name__ == '__main__':
    main()
