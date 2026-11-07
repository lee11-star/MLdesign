import csv
from pathlib import Path

import numpy as np
from PIL import Image


def make_submission_mask(mask, size=512):
    """Match organizer: resize binary image bilinearly, then select every nonzero pixel."""
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)) > 0


def encode_rle(mask):
    pixels = np.asarray(mask, dtype=np.uint8).ravel(order='C')
    padded = np.concatenate(([0], pixels, [0]))
    boundaries = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    boundaries[1::2] -= boundaries[::2]
    return ' '.join(map(str, boundaries))


def write_submission(mask_dir, sample_path, output_path):
    with open(sample_path, newline='', encoding='utf-8') as handle:
        ids = [row['Id'] for row in csv.DictReader(handle)]
    with open(output_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Id', 'Predicted'])
        for case_id in ids:
            mask = np.asarray(Image.open(Path(mask_dir) / f'{case_id}.png').convert('L')) > 0
            writer.writerow([case_id, encode_rle(mask)])
