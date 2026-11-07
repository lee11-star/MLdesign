from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .core import discover_data, image_files, validate_submission
from .modeling import PlantDataset, build_model, build_transforms


def checkpoint_probabilities(path: Path, samples, batch_size: int, workers: int, tta: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    mapping = checkpoint["class_to_idx"]; classes = [name for name, _ in sorted(mapping.items(), key=lambda x: x[1])]
    model = build_model(checkpoint["model_name"], len(classes), False).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    dataset = PlantDataset(
        samples,
        build_transforms(checkpoint["image_size"], False, mode=checkpoint.get("transform_mode", "crop")),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    all_probs = []
    with torch.inference_mode():
        for images, _, _ in loader:
            images = images.to(device)
            views = [images]
            if tta: views.extend([torch.flip(images, [3]), torch.flip(images, [2])])
            probs = torch.stack([torch.softmax(model(view), dim=1) for view in views]).mean(0)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs), classes


def main():
    parser = argparse.ArgumentParser(description="Ensemble inference and submission generation")
    parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument("--diagnostics", type=Path, default=Path("outputs/test_predictions_with_probabilities.csv"))
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--workers", type=int, default=2); parser.add_argument("--tta", action="store_true")
    args = parser.parse_args(); paths = discover_data(args.root); template = pd.read_csv(paths.submission_template)
    test_lookup = {p.name: p for p in image_files(paths.test_dir)}
    ordered_ids = template["ID"].astype(str).tolist(); samples = [(test_lookup[name], -1) for name in ordered_ids]
    probabilities, classes = [], None
    for checkpoint in args.checkpoints:
        probs, checkpoint_classes = checkpoint_probabilities(checkpoint, samples, args.batch_size, args.workers, args.tta)
        if classes is not None and classes != checkpoint_classes: raise ValueError("Checkpoint class mappings differ")
        classes = checkpoint_classes; probabilities.append(probs)
    mean_probs = np.mean(probabilities, axis=0); pred_idx = mean_probs.argmax(1); predictions = [classes[i] for i in pred_idx]
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": ordered_ids, "Category": predictions}).to_csv(args.submission, index=False, encoding="utf-8")
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diag = pd.DataFrame({"ID": ordered_ids, "Prediction": predictions, "Confidence": mean_probs.max(1)})
    for index, name in enumerate(classes): diag[f"prob_{name}"] = mean_probs[:, index]
    diag.sort_values("Confidence").to_csv(args.diagnostics, index=False, encoding="utf-8")
    result = validate_submission(args.submission, ordered_ids, classes)
    result.update({"submission": str(args.submission.resolve()), "diagnostics": str(args.diagnostics.resolve()), "tta": args.tta, "models": len(args.checkpoints), "lowest_confidence": diag.nsmallest(10, "Confidence")[["ID", "Prediction", "Confidence"]].to_dict("records")})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
