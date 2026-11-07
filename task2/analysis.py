"""Paired DINO evaluation and reproducible difficult-class visual sample."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from emotion_classifier.data import CLASSES, DEFAULT_DATA, read_manifest, save_json
from emotion_classifier.ensemble import align_probabilities, fuse_probabilities
from feature_transfer import old_validation
from model_selection import probability_metrics
from baseline import ROOT, digest, read_json

OUT = ROOT / 'outputs/round7'


def paired_metrics(baseline, candidate, labels, groups=None):
    old_ok, new_ok = baseline.argmax(1) == labels, candidate.argmax(1) == labels
    result = dict(**probability_metrics(candidate, labels),
                  corrected=int((~old_ok & new_ok).sum()),
                  broken=int((old_ok & ~new_ok).sum()),
                  net_correct=int(new_ok.sum() - old_ok.sum()),
                  old_errors=int((~old_ok).sum()))
    if groups is not None:
        _, index = np.unique(groups, return_inverse=True)
        delta = np.bincount(index, weights=new_ok.astype(int) - old_ok.astype(int))
        size = np.bincount(index)
        rng, estimates = np.random.default_rng(42), []
        for _ in range(20):
            picks = rng.integers(len(size), size=(100, len(size)))
            estimates.extend((delta[picks].sum(1) / size[picks].sum(1)).tolist())
        result['group_bootstrap_95pct_delta_accuracy'] = np.quantile(estimates, [0.025, 0.975]).tolist()
        result['interval_note'] = 'Exploratory paired hash-group bootstrap, 2000 resamples; reused validation set.'
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(ROOT / 'outputs/data/manifest.csv')
    val = [r for r in rows if r['split'] == 'val']
    ids, labels = np.asarray([r['id'] for r in val]), np.asarray([r['label'] for r in val])
    baseline, selection = old_validation(ids, labels)
    entries = read_json(ROOT / 'outputs/round6/models.json')
    if 'dinov2_last4' not in entries:
        raise ValueError('DINO fine-tune evaluation must finish before this analysis')
    comparisons, sources = [], []
    dinop = None
    for name, entry in entries.items():
        path = entry['probabilities']
        sources.append(dict(name=name, path=path, sha256=digest(path)))
        with np.load(path) as data:
            for key in ('probabilities', 'flip_tta_probabilities'):
                new = align_probabilities(ids, data['ids'], data[key], labels, data['targets'])
                if name == 'dinov2_last4' and key == 'flip_tta_probabilities':
                    dinop = new
                for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
                    mixed = fuse_probabilities([baseline, new], [1 - fraction, fraction])
                    comparisons.append(dict(model=name, view=key, fraction=fraction,
                                            **paired_metrics(baseline, mixed, labels)))
    winner = max(comparisons, key=lambda x: (x['accuracy'], -x['cross_entropy']))
    with np.load(entries[winner['model']]['probabilities']) as data:
        new = align_probabilities(ids, data['ids'], data[winner['view']], labels, data['targets'])
    mixed = fuse_probabilities([baseline, new], [1 - winner['fraction'], winner['fraction']])
    groups = np.asarray([r['sha256'] for r in val])
    report = dict(baseline=probability_metrics(baseline, labels), comparisons=comparisons,
                  best_exploratory=dict(winner, paired=paired_metrics(baseline, mixed, labels, groups)),
                  finetune_flip=paired_metrics(baseline, dinop, labels, groups),
                  sources=sources, manifest_sha256=digest(ROOT / 'outputs/data/manifest.csv'),
                  fixed_fractions=[0.1, 0.25, 0.5, 0.75, 1.0],
                  note='Both views reported; no continuous weight search. Kaggle score unknown.')
    save_json(report, OUT / 'dino_comparison.json')
    rng, sample = np.random.default_rng(42), []
    for label, count in ((1, 34), (3, 33), (0, 33)):
        eligible = np.flatnonzero((labels == label) & (baseline.argmax(1) != labels))
        unique = {}
        for i in eligible:
            unique.setdefault(groups[i], int(i))
        chosen = rng.choice(list(unique.values()), count, replace=False)
        sample.extend(sorted(chosen.tolist()))
    records = []
    for number, i in enumerate(sample, 1):
        records.append(dict(number=number, id=ids[i], path=val[i]['path'], sha256=groups[i],
                            true_class=CLASSES[labels[i]], baseline_prediction=CLASSES[baseline[i].argmax()],
                            dinov2_prediction=CLASSES[dinop[i].argmax()],
                            review_tags='', observation=''))
    with (OUT / 'error_audit_sample.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for page in range(5):
        canvas = Image.new('RGB', (1000, 960), 'white')
        draw = ImageDraw.Draw(canvas)
        for cell, record in enumerate(records[page * 20:(page + 1) * 20]):
            left, top = cell % 5 * 200, cell // 5 * 240
            with Image.open(DEFAULT_DATA / record['path']) as original:
                canvas.paste(original.resize((168, 168), Image.Resampling.NEAREST).convert('RGB'), (left + 16, top))
            draw.text((left + 8, top + 172), f"#{record['number']:03d} T:{record['true_class']}", fill='black', font=font)
            draw.text((left + 8, top + 192), 'Old:' + record['baseline_prediction'], fill='black', font=font)
            draw.text((left + 8, top + 212), 'DINO:' + record['dinov2_prediction'], fill='black', font=font)
        canvas.save(OUT / f'error_audit_{page + 1}.png')
    save_json(dict(sample_count=100, seed=42, unique_hash_groups=100,
                   classes=dict(Fear=34, Sad=33, Angry=33), sampling='Random within each true class among round4 errors; unique byte hashes.',
                   reviewed=False), OUT / 'error_audit_summary.json')
    print('DINO comparison:', winner, flush=True)
    print('Exported 100 unique-group errors for visual review.', flush=True)


if __name__ == '__main__':
    main()
