import csv
import re
from pathlib import Path

import numpy as np
import torch

from .data import CLASS_TO_LABEL, read_template
from .models import build_model, preprocess, preprocessing_config


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['class_to_label'] != CLASS_TO_LABEL:
        raise ValueError('Checkpoint class mapping differs from the competition')
    config = checkpoint['config']
    if 'preprocessing' in config and config['preprocessing'] != preprocessing_config(
            config['model'], config.get('image_size'), config.get('geometry', 'standard')):
        raise ValueError('Checkpoint preprocessing differs from the implementation')
    model = build_model(config['model'], head_dropout=config.get('head_dropout', 0.0),
                        trainable_blocks=config.get('trainable_blocks', 4),
                        dinov2_repo=config.get('dinov2_repo')).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, config, checkpoint


@torch.inference_mode()
def predict_probabilities(model, images, architecture, device, batch_size=128, tta=False, image_size=None):
    device = torch.device(device)
    model.eval()
    result = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start:start + batch_size]).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            probs = model(preprocess(batch, architecture, image_size=image_size)).float().softmax(1)
            if tta:
                probs = (probs + model(preprocess(batch, architecture, flip=True, image_size=image_size)).float().softmax(1)) / 2
        result.append(probs.cpu().numpy())
    return np.concatenate(result)


@torch.inference_mode()
def predict_shift_tta(model, images, architecture, device, batch_size=128, image_size=None):
    """Center and four 2-pixel translations; average each with its horizontal flip."""
    device = torch.device(device)
    model.eval()
    centers, shifted = [], []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(images[start:start + batch_size]).to(device)
        padded = torch.nn.functional.pad(batch.float(), (2, 2, 2, 2), mode='replicate')
        pairs = []
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            for dy, dx in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
                view = batch if dy == dx == 0 else padded[:, :, 2 + dy:2 + dy + 48, 2 + dx:2 + dx + 48]
                p = model(preprocess(view, architecture, image_size=image_size)).float().softmax(1)
                flipped = model(preprocess(view, architecture, flip=True, image_size=image_size)).float().softmax(1)
                pairs.append((p + flipped) / 2)
        centers.append(pairs[0].cpu().numpy())
        shifted.append(torch.stack(pairs).mean(0).cpu().numpy())
    return dict(center=np.concatenate(centers), shift=np.concatenate(shifted))


def write_submission(template, predictions, output):
    ids = read_template(template)
    if set(ids) != set(predictions):
        raise ValueError('Prediction IDs must match template IDs exactly')
    for value in predictions.values():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 5:
            raise ValueError('Emotion must be an integer from 0 to 5')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'Emotion'])
        writer.writerows((image_id, int(predictions[image_id])) for image_id in ids)
    return validate_submission(output, template)


def validate_submission(submission, template, expected_rows=None):
    ids = read_template(template)
    with open(submission, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['ID', 'Emotion']:
            raise ValueError('Submission columns must be ID,Emotion')
        rows = list(reader)
    if [row['ID'] for row in rows] != ids:
        raise ValueError('Submission IDs differ from template row order')
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows, found {len(rows)}')
    if any(None in row or not re.fullmatch('[0-5]', row['Emotion'] or '') for row in rows):
        raise ValueError('Invalid, empty or non-integer Emotion label')
    counts = {str(label): sum(row['Emotion'] == str(label) for row in rows) for label in range(6)}
    return dict(valid=True, rows=len(rows), columns=['ID', 'Emotion'], id_order_matches=True, label_counts=counts)
