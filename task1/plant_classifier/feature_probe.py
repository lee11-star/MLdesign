from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from .core import build_class_mapping, discover_data, image_files, seed_everything, validate_submission
from .modeling import PlantDataset, build_model, build_transforms, compatible_image_size
from .train import collect_samples

def cross_validate_features(features, labels, folds=5, seed=42):
    """Fit a fixed linear probe separately on each training fold."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if features.ndim != 2 or len(features) != len(labels) or len(classes) < 2:
        raise ValueError("Expected matching feature rows and at least two classes")
    probabilities = np.zeros((len(labels), len(classes)), dtype=np.float64)
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(features, labels)):
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        classifier.fit(features[train_idx], labels[train_idx])
        predicted = classifier.predict_proba(features[val_idx])
        for column, class_name in enumerate(classifier.classes_):
            probabilities[val_idx, np.where(classes == class_name)[0][0]] = predicted[:, column]
        fold_ids[val_idx] = fold
    return probabilities, fold_ids


def extract_embeddings(model, dataset, batch_size, device):
    """Capture the input to a torchvision model's final linear layer."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        last_layer = model.fc
    elif hasattr(model, "classifier") and isinstance(model.classifier[-1], nn.Linear):
        last_layer = model.classifier[-1]
    else:
        raise ValueError("Could not locate final linear classification layer")
    captured = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().cpu().numpy())

    handle = last_layer.register_forward_pre_hook(capture)
    try:
        model = model.to(device).eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for batch in loader:
                images = batch[0].to(device)
                model(images)
    finally:
        handle.remove()
    if not captured:
        raise ValueError("Cannot extract features from an empty dataset")
    return np.concatenate(captured, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen pretrained image encoder with five-fold linear probes")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["tiny_cnn", "resnet18", "efficientnet_b2", "convnext_tiny", "dinov2_vits14", "dinov2_vitb14"], required=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--submission", type=Path, help="Fit the full-data linear probe and generate a test submission")
    args = parser.parse_args()
    args.image_size = compatible_image_size(args.model, args.image_size)
    seed_everything(args.seed)
    data_paths = discover_data(args.root)
    mapping = build_class_mapping(data_paths.train_dir)
    classes = list(mapping)
    samples = collect_samples(data_paths.train_dir, mapping)
    labels = np.array([label for _, label in samples])
    dataset = PlantDataset(samples, build_transforms(args.image_size, False, mode="fine_grained"))
    model = build_model(args.model, len(classes), args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = extract_embeddings(model, dataset, args.batch_size, device)
    probabilities, fold_ids = cross_validate_features(embeddings, labels, args.folds, args.seed)
    predictions = probabilities.argmax(axis=1)
    output = pd.DataFrame({
        "filename": [path.name for path, _ in samples],
        "true": [classes[index] for index in labels],
        "predicted": [classes[index] for index in predictions],
        "confidence": probabilities.max(axis=1),
        "fold": fold_ids,
    })
    for index, name in enumerate(classes):
        output[f"prob_{name}"] = probabilities[:, index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "train_embeddings.npy", embeddings)
    output.to_csv(args.output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    report = classification_report(labels, predictions, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "model": args.model, "pretrained": args.pretrained, "image_size": args.image_size,
        "folds": args.folds, "seed": args.seed, "feature_dim": embeddings.shape[1],
        "oof_micro_f1": float(f1_score(labels, predictions, average="micro")),
        "oof_macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=range(len(classes))).tolist(),
    }
    if args.submission is not None:
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        classifier.fit(embeddings, labels)
        joblib.dump(classifier, args.output_dir / "feature_classifier.joblib")
        torch.save({
            "model_state": model.state_dict(), "model_name": args.model,
            "image_size": args.image_size, "class_to_idx": mapping,
            "transform_mode": "fine_grained", "pretrained": args.pretrained,
        }, args.output_dir / "encoder.pt")
        ids = pd.read_csv(data_paths.submission_template)["ID"].astype(str).tolist()
        lookup = {path.name: path for path in image_files(data_paths.test_dir)}
        test_samples = [(lookup[name], -1) for name in ids]
        test_dataset = PlantDataset(test_samples, build_transforms(args.image_size, False, mode="fine_grained"))
        test_embeddings = extract_embeddings(model, test_dataset, args.batch_size, device)
        test_probabilities = classifier.predict_proba(test_embeddings)
        test_predictions = [classes[index] for index in test_probabilities.argmax(axis=1)]
        args.submission.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": ids, "Category": test_predictions}).to_csv(args.submission, index=False, encoding="utf-8")
        diagnostic = pd.DataFrame({"ID": ids, "Prediction": test_predictions,
                                   "Confidence": test_probabilities.max(axis=1)})
        for index, name in enumerate(classes):
            diagnostic[f"prob_{name}"] = test_probabilities[:, index]
        diagnostic.to_csv(args.output_dir / "test_probabilities.csv", index=False, encoding="utf-8")
        summary["submission"] = str(args.submission.resolve())
        summary["submission_validation"] = validate_submission(args.submission, ids, classes)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
