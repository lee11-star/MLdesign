from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from .core import build_class_mapping, discover_data, image_files, save_mapping, seed_everything, stratified_split
from .metric_learning import convnext_logits_and_features, supervised_contrastive_loss
from .modeling import PlantDataset, build_model, build_transforms, compatible_image_size


def collect_samples(train_dir: Path, mapping: dict[str, int]) -> list[tuple[Path, int]]:
    return [(path, mapping[name]) for name in mapping for path in image_files(train_dir / name)]


def remap_class_subset(samples, candidate_indices, included_labels, full_classes):
    included_labels = list(included_labels)
    old_to_new = {old: new for new, old in enumerate(included_labels)}
    mapping = {full_classes[old]: new for old, new in old_to_new.items()}
    kept_indices = [i for i in candidate_indices if samples[i][1] in old_to_new]
    remapped = [(samples[i][0], old_to_new[samples[i][1]]) for i in kept_indices]
    return remapped, kept_indices, mapping


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), generator=generator,
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0,
    )


def classification_head(model: nn.Module) -> nn.Module:
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "fc"):
        return model.fc
    raise ValueError(f"Cannot identify classification head for {type(model).__name__}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze every parameter except the classification head."""
    head = classification_head(model)
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in head.parameters():
        parameter.requires_grad = True
    model._backbone_frozen = not trainable


def optimizer_parameter_groups(model: nn.Module, base_lr: float, head_lr_multiplier: float):
    """Build exhaustive, disjoint backbone/head groups for discriminative learning rates."""
    head_ids = {id(parameter) for parameter in classification_head(model).parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    head = [parameter for parameter in model.parameters() if id(parameter) in head_ids]
    return [
        {"params": backbone, "lr": base_lr},
        {"params": head, "lr": base_lr * head_lr_multiplier},
    ]


def load_compatible_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    expected_model: str,
    expected_fold: int,
    expected_seed: int,
    expected_num_folds: int | None,
) -> dict[str, object]:
    """Load matching tensors after validating split identity.

    This supports transferring a five-class backbone into a binary expert
    while deliberately leaving shape-incompatible classifier tensors at
    their freshly initialized values.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != expected_model:
        raise ValueError(
            f"Initialization model {checkpoint.get('model_name')} does not match {expected_model}"
        )
    metadata = checkpoint.get("args", {})
    checks = {
        "fold": (metadata.get("fold"), expected_fold),
        "seed": (metadata.get("seed"), expected_seed),
    }
    if expected_num_folds is not None:
        checks["num_folds"] = (metadata.get("num_folds"), expected_num_folds)
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"Initialization checkpoint {name}={actual}; expected {expected}")
    target_state = model.state_dict()
    source_state = checkpoint["model_state"]
    compatible = {
        name: tensor for name, tensor in source_state.items()
        if name in target_state and target_state[name].shape == tensor.shape
    }
    if not compatible:
        raise ValueError("Initialization checkpoint has no shape-compatible model tensors")
    model.load_state_dict(compatible, strict=False)
    skipped = sorted(name for name in source_state if name not in compatible)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": sorted(compatible),
        "skipped_keys": skipped,
    }


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, max_batches=0, grad_clip=0.0,
              contrastive_weight=0.0):
    training = optimizer is not None
    model.train(training)
    if training and getattr(model, "_backbone_frozen", False):
        model.eval()
        classification_head(model).train()
    losses, labels, preds = [], [], []
    for batch_index, (images, targets, _) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            if training and contrastive_weight:
                logits, features = convnext_logits_and_features(model, images)
            else:
                logits = model(images)
            loss = criterion(logits, targets)
            if training and contrastive_weight:
                loss = loss + contrastive_weight * supervised_contrastive_loss(features, targets)
        if training:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.item() * len(targets))
        labels.extend(targets.detach().cpu().tolist())
        preds.extend(logits.argmax(1).detach().cpu().tolist())
    count = len(labels)
    return {
        "loss": float(sum(losses) / count),
        "micro_f1": float(f1_score(labels, preds, average="micro")),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "labels": labels,
        "preds": preds,
    }


def save_diagnostics(output_dir: Path, history: list[dict], labels, preds, classes):
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([x["epoch"] for x in history], [x["train_loss"] for x in history], label="train")
    axes[0].plot([x["epoch"] for x in history], [x["val_loss"] for x in history], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot([x["epoch"] for x in history], [x["val_micro_f1"] for x in history], label="micro-F1")
    axes[1].plot([x["epoch"] for x in history], [x["val_macro_f1"] for x in history], label="macro-F1")
    axes[1].set_title("Validation metrics"); axes[1].legend(); fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160); plt.close(fig)
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(7, 6)); image = ax.imshow(cm, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right"); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close(fig)
    report = classification_report(
        labels, preds, labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0,
    )
    (output_dir / "classification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def train(args) -> dict:
    started = time.perf_counter(); seed_everything(args.seed)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint cannot be used together")
    if args.contrastive_weight < 0 or (args.contrastive_weight > 0 and args.model != "convnext_tiny"):
        raise ValueError("--contrastive-weight requires convnext_tiny and a nonnegative weight")
    paths = discover_data(args.root); full_mapping = build_class_mapping(paths.train_dir); full_classes = list(full_mapping)
    samples = collect_samples(paths.train_dir, full_mapping); labels = [label for _, label in samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        args.batch_size = min(args.batch_size, 8)
        args.image_size = min(args.image_size, 160)
        args.epochs = min(args.epochs, 5)
        args.workers = min(args.workers, 1)
    args.image_size = compatible_image_size(args.model, args.image_size)
    if args.fold >= 0:
        folds = list(StratifiedKFold(args.num_folds, shuffle=True, random_state=args.seed).split(np.zeros(len(labels)), labels))
        train_idx, val_idx = folds[args.fold]
        train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
    else:
        train_idx, val_idx = stratified_split(labels, args.val_size, args.seed)
    if args.include_classes:
        included_labels = [full_mapping[name] for name in args.include_classes]
        train_samples, train_idx, mapping = remap_class_subset(samples, train_idx, included_labels, full_classes)
        val_samples, val_idx, _ = remap_class_subset(samples, val_idx, included_labels, full_classes)
    else:
        mapping = full_mapping
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
    classes = list(mapping)
    output_dir = args.output_dir; output_dir.mkdir(parents=True, exist_ok=True); save_mapping(mapping, output_dir / "class_mapping.json")
    train_ds = PlantDataset(train_samples, build_transforms(args.image_size, True, args.strong_aug, args.transform_mode))
    val_ds = PlantDataset(val_samples, build_transforms(args.image_size, False, mode=args.transform_mode))
    workers = args.workers if device.type == "cuda" else min(args.workers, 1)
    model = build_model(args.model, len(classes), args.pretrained).to(device)
    initialization = None
    if args.init_checkpoint:
        initialization = load_compatible_checkpoint(
            model, args.init_checkpoint, args.model, args.fold, args.seed, args.num_folds
        )
        print(json.dumps({"initialization": initialization}, ensure_ascii=False))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.freeze_backbone_epochs:
        set_backbone_trainable(model, False)
    if args.head_lr_multiplier == 1.0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        parameter_groups = optimizer_parameter_groups(model, args.lr, args.head_lr_multiplier)
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_f1, bad_epochs = 1, -1.0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint: scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1; best_f1 = checkpoint["best_micro_f1"]
        if args.freeze_backbone_epochs:
            set_backbone_trainable(model, start_epoch > args.freeze_backbone_epochs)
    train_loader = make_loader(train_ds, args.batch_size, True, workers, args.seed)
    val_loader = make_loader(val_ds, args.batch_size, False, workers, args.seed)
    history, best_epoch = [], 0
    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, args.max_batches,
                                  args.grad_clip, args.contrastive_weight)
        val_metrics = run_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_metrics["loss"],
               "train_micro_f1": train_metrics["micro_f1"], "val_loss": val_metrics["loss"],
               "val_micro_f1": val_metrics["micro_f1"], "val_accuracy": val_metrics["accuracy"],
               "val_macro_f1": val_metrics["macro_f1"]}
        history.append(row); print(json.dumps(row)); scheduler.step()
        state = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "epoch": epoch,
                 "best_micro_f1": max(best_f1, val_metrics["micro_f1"]), "class_to_idx": mapping,
                 "model_name": args.model, "image_size": args.image_size, "pretrained": args.pretrained,
                 "transform_mode": args.transform_mode,
                 "initialization": initialization,
                 "args": vars(args) | {"root": str(args.root), "output_dir": str(args.output_dir), "resume": str(args.resume) if args.resume else None}}
        torch.save(state, output_dir / "last.pt")
        if val_metrics["micro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val_metrics["micro_f1"], epoch, 0
            torch.save(state, output_dir / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience: break
    best = torch.load(output_dir / "best.pt", map_location=device, weights_only=False); model.load_state_dict(best["model_state"])
    final_val = run_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
    save_diagnostics(output_dir, history, final_val["labels"], final_val["preds"], classes)
    summary = {"experiment": output_dir.name, "model": args.model, "pretrained": args.pretrained,
               "image_size": args.image_size, "strong_aug": args.strong_aug, "transform_mode": args.transform_mode, "optimizer": "AdamW",
               "lr": args.lr, "head_lr_multiplier": args.head_lr_multiplier,
               "freeze_backbone_epochs": args.freeze_backbone_epochs,
               "contrastive_weight": args.contrastive_weight,
               "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
               "scheduler": "CosineAnnealingLR", "batch_size": args.batch_size,
               "epochs_requested": args.epochs, "epochs_run": len(history), "best_epoch": best_epoch,
               "seed": args.seed, "fold": args.fold, "train_samples": len(train_idx), "val_samples": len(val_idx),
               "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None,
               "elapsed_seconds": time.perf_counter() - started, "val_loss": final_val["loss"],
               "val_micro_f1": final_val["micro_f1"], "val_accuracy": final_val["accuracy"], "val_macro_f1": final_val["macro_f1"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False)); return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train plant classifier")
    parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=[
            "tiny_cnn", "resnet18", "efficientnet_b0", "efficientnet_b2", "convnext_tiny", "dinov2_vits14", "dinov2_vitb14",
            "dual_resnet18", "dual_convnext_tiny",
        ],
        default="resnet18",
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=224); parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=.1); parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--transform-mode", choices=["crop", "fine_grained", "dual_foreground"], default="crop")
    parser.add_argument("--head-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-size", type=float, default=.2); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4); parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--fold", type=int, default=-1); parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--include-classes", nargs="+", choices=["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"])
    parser.add_argument("--max-batches", type=int, default=0, help="Smoke-test limit; 0 uses all batches")
    return parser.parse_args()


if __name__ == "__main__": train(parse_args())
