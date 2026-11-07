import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from emotion_classifier.data import CLASSES, DEFAULT_DATA, read_manifest, load_arrays, save_json
from emotion_classifier.inference import load_checkpoint, predict_probabilities


def run_evaluation(checkpoint, output=None, manifest='outputs/data/manifest.csv', data_root=DEFAULT_DATA,
                   cache_dir='outputs/data', device='cuda', batch_size=128):
    torch.set_num_threads(4)
    output = Path(output or Path(checkpoint).parent / 'evaluation')
    output.mkdir(parents=True, exist_ok=True)
    model, config, state = load_checkpoint(checkpoint, device)
    if config.get('final_training'):
        raise ValueError('Full-data model cannot produce an independent holdout validation score')
    if config['data_signature'] != hashlib.sha256(Path(manifest).read_bytes()).hexdigest():
        raise ValueError('Evaluation manifest differs from training')
    rows = read_manifest(manifest)
    x, y, ids = load_arrays(rows, data_root, 'val', cache_dir)
    size = config.get('preprocessing', {}).get('size')
    base = predict_probabilities(model, x, config['model'], device, batch_size, image_size=size)
    # Evaluate horizontal flip augmentation once on the selected holdout checkpoint.
    tta = predict_probabilities(model, x, config['model'], device, batch_size, tta=True, image_size=size)
    base_accuracy = float((base.argmax(1) == y).mean())
    tta_accuracy = float((tta.argmax(1) == y).mean())
    use_tta = tta_accuracy > base_accuracy
    probs = tta if use_tta else base
    prediction = probs.argmax(1)
    matrix = confusion_matrix(y, prediction, labels=list(range(6)))
    metrics = dict(checkpoint=str(Path(checkpoint).resolve()), best_epoch=state['best_epoch'], samples=len(y),
                   accuracy=base_accuracy, flip_tta_accuracy=tta_accuracy, selected_tta=use_tta,
                   selected_accuracy=max(base_accuracy, tta_accuracy),
                   per_class={name: dict(count=int(matrix[i].sum()), correct=int(matrix[i, i]),
                                         accuracy=float(matrix[i, i] / matrix[i].sum())) for i, name in enumerate(CLASSES)},
                   confusion_matrix=matrix.tolist(), class_to_label=config['class_to_label'],
                   cross_entropy=float(-np.log(np.clip(base[np.arange(len(y)), y], 1e-12, 1)).mean()),
                   flip_tta_cross_entropy=float(-np.log(np.clip(tta[np.arange(len(y)), y], 1e-12, 1)).mean()))
    save_json(metrics, output / 'metrics.json')
    np.savez_compressed(output / 'validation_probabilities.npz', ids=np.array(ids), targets=y,
                        probabilities=base, flip_tta_probabilities=tta, class_names=np.asarray(CLASSES))
    with (output / 'errors.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'true_label', 'predicted_label', 'confidence'])
        writer.writerows((ids[i], int(y[i]), int(prediction[i]), float(probs[i, prediction[i]]))
                         for i in np.flatnonzero(y != prediction))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(6), CLASSES, rotation=45, ha='right')
    ax.set_yticks(range(6), CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    fig.tight_layout(); fig.savefig(output / 'confusion_matrix.png', dpi=150); plt.close(fig)
    errors = np.flatnonzero(y != prediction)
    errors = errors[np.argsort(probs[errors].max(1))[::-1]][:24]
    fig, axes = plt.subplots(4, 6, figsize=(12, 9))
    for ax, i in zip(axes.flat, errors):
        ax.imshow(x[i, 0], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'T: {CLASSES[y[i]]}\nP: {CLASSES[prediction[i]]} {probs[i].max():.2f}', fontsize=8)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(output / 'errors.png', dpi=140); plt.close(fig)
    history = state['history']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row['epoch'] for row in history]
    for prefix in ['train', 'val']:
        axes[0].plot(epochs, [r[f'{prefix}_loss'] for r in history], label=prefix)
        axes[1].plot(epochs, [r[f'{prefix}_accuracy'] for r in history], label=prefix)
    axes[0].set_ylabel('Cross entropy'); axes[1].set_ylabel('Accuracy')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend()
    fig.tight_layout(); fig.savefig(output / 'learning_curves.png', dpi=150); plt.close(fig)
    print(metrics, flush=True)
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output')
    p.add_argument('--device', default='cuda')
    run_evaluation(**vars(p.parse_args()))
