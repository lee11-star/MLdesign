import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vessel.data import image_ids, split_ids


def main():
    root = Path('segmentation')
    output = Path('outputs')
    output.mkdir(exist_ok=True)
    train_ids = image_ids(root / 'train/image')
    label_ids = image_ids(root / 'train/label')
    test_ids = image_ids(root / 'test/image')
    assert train_ids == label_ids
    train_split, val_split = split_ids(train_ids)
    records, hashes = [], {}
    duplicates = []
    for part in ['train/image', 'train/label', 'test/image']:
        for path in sorted((root / part).iterdir(), key=lambda p: int(p.stem)):
            with Image.open(path) as im:
                array = np.asarray(im)
                record = {'path': str(path), 'size_wh': list(im.size), 'mode': im.mode, 'actual_format': im.format}
                if part == 'train/label':
                    record.update(values=np.unique(array).tolist(), vessel_fraction=float((array > 128).mean()))
                else:
                    digest = hashlib.sha256(array.tobytes()).hexdigest()
                    record['pixel_sha256'] = digest
                    if digest in hashes:
                        duplicates.append([hashes[digest], str(path)])
                    hashes[digest] = str(path)
                records.append(record)
    report = {'train_count': len(train_ids), 'test_count': len(test_ids), 'seed': 42,
              'train_ids': train_split, 'validation_ids': val_split, 'duplicate_images': duplicates,
              'foreground': 'white vessels (>128); provided encoder selects nonzero pixels', 'files': records}
    (output / 'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(4, 3, figsize=(10, 14))
    for row, case_id in enumerate(val_split):
        rgb = np.asarray(Image.open(root / f'train/image/{case_id}.jpg').convert('RGB'))
        mask = np.asarray(Image.open(root / f'train/label/{case_id}.jpg')) > 128
        overlay = rgb.copy()
        overlay[mask] = (0.4 * rgb[mask] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        for col, (a, title) in enumerate([(rgb, 'Image'), (mask, 'White = vessel'), (overlay, 'Overlay')]):
            axes[row, col].imshow(a, cmap='gray')
            axes[row, col].set_title(f'{case_id}: {title}')
            axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig(output / 'data_preview.png', dpi=110)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
