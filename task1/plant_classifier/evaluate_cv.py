from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .core import build_class_mapping, discover_data
from .infer import checkpoint_probabilities
from .train import collect_samples


def validate_cv_checkpoints(checkpoints, expected_seed: int) -> None:
    expected_num_folds = len(checkpoints)
    for expected_fold, path in enumerate(checkpoints):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("args", {})
        actual_fold = metadata.get("fold")
        if actual_fold != expected_fold:
            raise ValueError(f"Checkpoint {path} has fold {actual_fold}; expected fold {expected_fold}")
        if metadata.get("num_folds") != expected_num_folds:
            raise ValueError(f"Checkpoint {path} num_folds does not match {expected_num_folds}")
        if metadata.get("seed") != expected_seed:
            raise ValueError(f"Checkpoint {path} seed does not match {expected_seed}")


def main():
    parser = argparse.ArgumentParser(description="Build out-of-fold diagnostics for CV checkpoints")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cv_summary"))
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_cv_checkpoints(args.checkpoints, args.seed)
    paths = discover_data(args.root); mapping = build_class_mapping(paths.train_dir); classes = list(mapping)
    samples = collect_samples(paths.train_dir, mapping); y_true = np.array([x[1] for x in samples]); n = len(samples)
    splitter = StratifiedKFold(len(args.checkpoints), shuffle=True, random_state=args.seed)
    oof_probs = np.zeros((n, len(classes)), dtype=np.float32); fold_column = np.full(n, -1); fold_metrics = []
    for fold, ((_, val_idx), checkpoint) in enumerate(zip(splitter.split(np.zeros(n), y_true), args.checkpoints)):
        val_samples = [samples[i] for i in val_idx]
        probs, checkpoint_classes = checkpoint_probabilities(checkpoint, val_samples, args.batch_size, 2, args.tta)
        if checkpoint_classes != classes: raise ValueError("Class mapping mismatch")
        oof_probs[val_idx] = probs; fold_column[val_idx] = fold
        pred = probs.argmax(1); truth = y_true[val_idx]
        fold_metrics.append({"fold": fold, "micro_f1": f1_score(truth, pred, average="micro"),
                             "accuracy": accuracy_score(truth, pred), "macro_f1": f1_score(truth, pred, average="macro")})
    pred = oof_probs.argmax(1)
    output = pd.DataFrame({"filename": [p.name for p, _ in samples], "true": [classes[i] for i in y_true],
                           "predicted": [classes[i] for i in pred], "confidence": oof_probs.max(1), "fold": fold_column})
    for class_index, class_name in enumerate(classes):
        output[f"prob_{class_name}"] = oof_probs[:, class_index]
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    report = classification_report(y_true, pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    (args.output_dir / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_true, pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right"); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(args.output_dir / "confusion_matrix.png", dpi=160); plt.close(fig)
    summary = {"tta": args.tta, "folds": fold_metrics, "mean_micro_f1": float(np.mean([x["micro_f1"] for x in fold_metrics])),
               "std_micro_f1": float(np.std([x["micro_f1"] for x in fold_metrics], ddof=1)),
               "oof_micro_f1": float(f1_score(y_true, pred, average="micro")),
               "oof_accuracy": float(accuracy_score(y_true, pred)), "oof_macro_f1": float(f1_score(y_true, pred, average="macro")),
               "confusion_matrix": cm.tolist()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
