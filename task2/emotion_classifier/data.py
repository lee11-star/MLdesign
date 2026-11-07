import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASS_TO_LABEL = {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5}
CLASSES = list(CLASS_TO_LABEL)
DEFAULT_DATA = Path('2026task2/fer_data/fer_data')
DEFAULT_TEMPLATE = Path('2026task2/submission.csv')


def read_template(path):
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['ID', 'Emotion']:
            raise ValueError('Expected columns ID,Emotion')
        ids = [row['ID'] for row in reader]
    if not ids or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError('Template contains empty or duplicate IDs')
    return ids


def audit_dataset(root, template, cache_dir=None):
    root = Path(root)
    records, pixels = [], []
    dimensions, modes, groups = Counter(), Counter(), defaultdict(list)
    for kind in ['train', 'test']:
        if kind == 'train':
            actual = {p.name for p in (root / kind).iterdir() if p.is_dir()}
            if actual != set(CLASS_TO_LABEL):
                raise ValueError(f'Unexpected class folders: {actual}')
            paths = sorted((root / kind).glob('*/*.jpg'))
        else:
            paths = sorted((root / kind).glob('*.jpg'))
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                image.load()
                dimensions[f'{image.width}x{image.height}'] += 1
                modes[image.mode] += 1
                if image.size != (48, 48) or image.mode != 'L':
                    raise ValueError(f'Unexpected image format: {path}')
                if cache_dir is not None:
                    pixels.append(np.asarray(image, dtype=np.uint8).copy()[None])
            row = dict(path=path.relative_to(root).as_posix(), id=path.name,
                       label=CLASS_TO_LABEL[path.parent.name] if kind == 'train' else -1,
                       kind=kind, sha256=digest, cache_index=len(records))
            records.append(row)
            groups[digest].append(row)
    template_ids = read_template(template)
    test_ids = [r['id'] for r in records if r['kind'] == 'test']
    if len(test_ids) != len(set(test_ids)) or set(test_ids) != set(template_ids):
        raise ValueError('Test image IDs do not match the template')
    train_groups = {key: [r for r in group if r['kind'] == 'train'] for key, group in groups.items()}
    conflict = {key: group for key, group in train_groups.items() if len({r['label'] for r in group}) > 1}
    overlap = {key: group for key, group in groups.items() if {r['kind'] for r in group} == {'train', 'test'}}
    summary = dict(train_count=sum(r['kind'] == 'train' for r in records), test_count=len(test_ids),
                   class_counts=dict(Counter(CLASSES[r['label']] for r in records if r['kind'] == 'train')),
                   dimensions=dict(dimensions), modes=dict(modes),
                   duplicate_groups=sum(len(group) > 1 for group in groups.values()),
                   train_duplicate_groups=sum(len(group) > 1 for group in train_groups.values()),
                   train_extra_duplicates=sum(max(0, len(group) - 1) for group in train_groups.values()),
                   conflict_groups=len(conflict), conflict_samples=sum(map(len, conflict.values())),
                   train_test_overlap_groups=len(overlap),
                   overlapping_test_samples=sum(r['kind'] == 'test' for group in overlap.values() for r in group),
                   class_to_label=CLASS_TO_LABEL)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / 'pixels.npy', np.stack(pixels), allow_pickle=False)
    return records, summary


def split_records(records, seed=42, val_fraction=0.2):
    if val_fraction != 0.2:
        raise ValueError('This experiment uses one fold of a five-fold stratified group split (20%)')
    result = [dict(row) for row in records]
    labels_by_hash = defaultdict(set)
    for row in result:
        if row['kind'] == 'train':
            labels_by_hash[row['sha256']].add(int(row['label']))
    conflicts = {key for key, labels in labels_by_hash.items() if len(labels) > 1}
    eligible = []
    for index, row in enumerate(result):
        row['split'] = 'test' if row['kind'] == 'test' else 'excluded' if row['sha256'] in conflicts else 'train'
        if row['split'] == 'train':
            eligible.append(index)
    labels = np.array([result[i]['label'] for i in eligible], dtype=np.int64)
    groups = np.array([result[i]['sha256'] for i in eligible])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, validation = next(splitter.split(np.zeros(len(labels)), labels, groups))
    for i in validation:
        result[eligible[i]]['split'] = 'val'
    check_split(result)
    return result


def check_split(records):
    hashes = {split: {r['sha256'] for r in records if r['split'] == split} for split in ['train', 'val']}
    if hashes['train'] & hashes['val']:
        raise ValueError('Duplicate hash leakage across training and validation')
    if {int(r['label']) for r in records if r['split'] == 'val'} != set(range(6)):
        raise ValueError('Validation must contain all six classes')


def save_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_manifest(path):
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['label'] = int(row['label'])
        row['cache_index'] = int(row['cache_index'])
    return rows


def load_arrays(records, root, split, cache_dir=None):
    selected = [r for r in records if (r.get('split', r['kind']) == split or
                                     (split == 'all' and r.get('split') in ['train', 'val']))]
    if not selected:
        raise ValueError(f'No records for split {split}')
    if cache_dir is not None:
        cached = np.load(Path(cache_dir) / 'pixels.npy', mmap_mode='r', allow_pickle=False)
        x = np.array(cached[[int(r['cache_index']) for r in selected]], copy=True)
    else:
        images = []
        for row in selected:
            with Image.open(Path(root) / row['path']) as image:
                images.append(np.asarray(image.convert('L'), dtype=np.uint8).copy()[None])
        x = np.stack(images)
    return x, np.array([int(r['label']) for r in selected], dtype=np.int64), [r['id'] for r in selected]


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
