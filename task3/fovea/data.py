import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.fft import dctn
from sklearn.model_selection import GroupShuffleSplit


def normalize_xy(xy, sizes):
    return np.asarray(xy, dtype=np.float64) / np.asarray(sizes, dtype=np.float64)


def restore_xy(xy, sizes):
    return np.asarray(xy, dtype=np.float64) * np.asarray(sizes, dtype=np.float64)


def pixel_metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    distances = np.linalg.norm(error, axis=1)
    return {'rmse': float(np.sqrt(np.mean(error ** 2))),
            'rmse_x': float(np.sqrt(np.mean(error[:, 0] ** 2))),
            'rmse_y': float(np.sqrt(np.mean(error[:, 1] ** 2))),
            'mean_distance': float(distances.mean()),
            'median_distance': float(np.median(distances)),
            'max_distance': float(distances.max())}


def write_submission(template, predictions, output):
    frame = pd.read_csv(template)
    values = []
    for name in frame.ImageID:
        image_id, _, axis = name.rsplit('_', 2)
        if int(image_id) not in predictions:
            raise ValueError(f'Missing prediction: {image_id}')
        values.append(float(predictions[int(image_id)][{'X': 0, 'Y': 1}[axis]]))
    if not np.isfinite(values).all():
        raise ValueError('Predictions must be finite')
    frame['value'] = values
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format='%.6f')
    return frame


def read_records(root):
    root = Path(root)
    frame = pd.read_csv(root / 'fovea_localization_train_GT.csv').rename(columns={'data': 'id'})
    frame['path'] = [str(root / 'train' / f'{int(i):04d}.jpg') for i in frame.id]
    sizes = [Image.open(p).size for p in frame.path]
    frame[['width', 'height']] = sizes
    xy = frame[['Fovea_X', 'Fovea_Y']].to_numpy()
    if not np.isfinite(xy).all() or (xy < 0).any() or (xy >= np.asarray(sizes)).any():
        raise ValueError('Invalid training coordinates')
    if frame.id.duplicated().any():
        raise ValueError('Duplicate training IDs')
    return frame


def audit_data(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = read_records(root)
    paths = list(frame.path) + [str(p) for p in sorted((root / 'test').glob('*.jpg'))]
    hashes, exact = [], {}
    for path in paths:
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            exact.setdefault(digest, []).append(str(path))
            small = np.asarray(rgb.convert('L').resize((32, 32)), dtype=float)
            freq = dctn(small, norm='ortho')[:8, :8].ravel()[1:]
            hashes.append(freq > np.median(freq))
    hashes = np.stack(hashes)
    parents = list(range(len(paths)))

    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i

    similar = []
    for i in range(len(paths)):
        for j in range(i):
            distance = int(np.count_nonzero(hashes[i] != hashes[j]))
            if distance <= 6:
                if i < len(frame):
                    parents[find(i)] = find(j)
                similar.append({'a': paths[j], 'b': paths[i], 'phash_distance': distance})
    frame['group'] = [find(i) for i in range(len(frame))]
    box_errors = []
    for row in frame.itertuples():
        annotation = ET.parse(root / 'train_location' / f'{row.id:04d}.xml').getroot()
        box = annotation.find('.//bndbox')
        center = [(float(box.findtext('xmin')) + float(box.findtext('xmax'))) / 2,
                  (float(box.findtext('ymin')) + float(box.findtext('ymax'))) / 2]
        box_errors.append(np.abs(np.asarray(center) - [row.Fovea_X, row.Fovea_Y]))
        if (int(annotation.findtext('size/width')), int(annotation.findtext('size/height'))) != (row.width, row.height):
            raise ValueError(f'XML/image size mismatch: {row.id}')
    xy = frame[['Fovea_X', 'Fovea_Y']].to_numpy()
    report = {'train_count': len(frame), 'test_count': len(paths) - len(frame),
              'train_image_sizes': frame.groupby(['width', 'height']).size().to_dict(),
              'invisible_labels': int(np.all(xy == 0, axis=1).sum()),
              'exact_duplicate_sets': [v for v in exact.values() if len(v) > 1],
              'similar_pairs': similar, 'training_groups': int(frame.group.nunique()),
              'box_center_mean_absolute_difference': np.mean(box_errors, axis=0).tolist()}
    report['train_image_sizes'] = {str(k): int(v) for k, v in report['train_image_sizes'].items()}
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    frame.to_csv(output / 'records.csv', index=False)
    return frame, report


def make_split(frame, seed=42):
    # Keep perceptually similar pictures together and represent both camera sizes.
    splitter = GroupShuffleSplit(n_splits=100, test_size=.2, random_state=seed)
    for train, val in splitter.split(frame, groups=frame.group):
        if set(frame.iloc[train].width) == set(frame.width) == set(frame.iloc[val].width) and 12 <= len(val) <= 24:
            return train, val
    raise ValueError('Could not construct a size-balanced group split')


def load_images(paths, size, boxes=None):
    images = []
    for index, path in enumerate(paths):
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            if boxes is not None:
                rgb = rgb.crop(tuple(int(value) for value in boxes[index]))
            images.append(np.asarray(rgb.resize((size, size), Image.Resampling.BILINEAR)).copy())
    return np.stack(images)


def prediction_preview(frame, predicted, output, truth=True):
    cell_w, cell_h, columns = 300, 235, 4
    canvas = Image.new('RGB', (columns * cell_w, ((len(frame) + columns - 1) // columns) * cell_h), 'white')
    for index, (row, xy) in enumerate(zip(frame.itertuples(), predicted)):
        with Image.open(row.path) as original:
            w, h = original.size
            im = original.convert('RGB').resize((cell_w, cell_h - 25))
        draw = ImageDraw.Draw(im)
        x, y = xy * np.array([cell_w / w, (cell_h - 25) / h])
        draw.line((x-7, y, x+7, y), fill='cyan', width=2)
        draw.line((x, y-7, x, y+7), fill='cyan', width=2)
        if truth:
            gx, gy = np.array([row.Fovea_X, row.Fovea_Y]) * [cell_w / w, (cell_h - 25) / h]
            draw.ellipse((gx-4, gy-4, gx+4, gy+4), outline='lime', width=2)
        left, top = (index % columns) * cell_w, (index // columns) * cell_h
        canvas.paste(im, (left, top))
        text = f'{row.id:04d}  ({xy[0]:.1f}, {xy[1]:.1f})'
        if truth:
            text += f'  err={np.linalg.norm(xy-[row.Fovea_X, row.Fovea_Y]):.1f}px'
        ImageDraw.Draw(canvas).text((left+4, top+cell_h-21), text, fill='black')
    canvas.save(output)
