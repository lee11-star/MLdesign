import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from emotion_classifier.data import (CLASSES, DEFAULT_DATA, DEFAULT_TEMPLATE, audit_dataset,
                                     split_records, save_manifest, save_json)


def run_audit(root=DEFAULT_DATA, template=DEFAULT_TEMPLATE, output='outputs/data', seed=42, zip_path='2026task2.zip'):
    output, root = Path(output), Path(root)
    records, summary = audit_dataset(root, template, output)
    records = split_records(records, seed=seed)
    save_manifest(records, output / 'manifest.csv')
    save_manifest([r for r in records if r['split'] == 'excluded'], output / 'excluded_conflicts.csv')
    groups = defaultdict(list)
    for row in records:
        groups[row['sha256']].append(row)
    save_manifest([row for group in groups.values() if len(group) > 1 for row in group], output / 'duplicates.csv')
    summary['seed'] = seed
    summary['split_counts'] = dict(Counter(r['split'] for r in records))
    summary['split_class_counts'] = {split: dict(Counter(CLASSES[r['label']] for r in records if r['split'] == split))
                                   for split in ['train', 'val', 'excluded']}
    summary['validation_fraction'] = summary['split_counts']['val'] / (summary['split_counts']['train'] + summary['split_counts']['val'])
    summary['train_val_hash_intersection'] = 0
    summary['manifest_sha256'] = hashlib.sha256((output / 'manifest.csv').read_bytes()).hexdigest()
    if zip_path and Path(zip_path).is_file():
        with ZipFile(zip_path) as archive:
            zip_images = {name.split('fer_data/fer_data/', 1)[1]: name for name in archive.namelist()
                          if 'fer_data/fer_data/' in name and name.lower().endswith('.jpg')}
            # Archives may contain one fer_data prefix instead of two; resolve using /train/ or /test/.
            if not zip_images:
                zip_images = {}
                for name in archive.namelist():
                    for kind in ['train', 'test']:
                        token = f'/{kind}/'
                        if token in name and name.lower().endswith('.jpg'):
                            zip_images[kind + '/' + name.split(token, 1)[1]] = name
            if set(zip_images) != {r['path'] for r in records}:
                raise ValueError('Extracted image paths differ from ZIP')
            for row in records:
                if hashlib.sha256(archive.read(zip_images[row['path']])).hexdigest() != row['sha256']:
                    raise ValueError(f"Extracted bytes differ from ZIP: {row['path']}")
            templates = [name for name in archive.namelist() if name.endswith('/submission.csv') or name == 'submission.csv']
            if len(templates) != 1 or archive.read(templates[0]) != Path(template).read_bytes():
                raise ValueError('Extracted template differs from ZIP')
        summary['zip_verified_images'] = len(records)
        summary['zip_template_identical'] = True
    save_json(summary, output / 'audit.json')
    cache = np.load(output / 'pixels.npy', mmap_mode='r')
    fig, axes = plt.subplots(6, 8, figsize=(12, 9))
    for label, name in enumerate(CLASSES):
        selected = [r for r in records if r['kind'] == 'train' and r['label'] == label][:8]
        for column, row in enumerate(selected):
            axes[label, column].imshow(cache[row['cache_index'], 0], cmap='gray', vmin=0, vmax=255)
            axes[label, column].set_axis_off()
            if column == 0:
                axes[label, column].set_title(f'{name} ({label})')
    fig.tight_layout()
    fig.savefig(output / 'samples.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(CLASSES, [summary['class_counts'][name] for name in CLASSES])
    ax.set_ylabel('Training images')
    fig.tight_layout()
    fig.savefig(output / 'class_distribution.png', dpi=140)
    plt.close(fig)
    print(summary, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=str(DEFAULT_DATA))
    parser.add_argument('--template', default=str(DEFAULT_TEMPLATE))
    parser.add_argument('--output', default='outputs/data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--zip', default='2026task2.zip')
    args = parser.parse_args()
    run_audit(args.data_root, args.template, args.output, args.seed, args.zip)
