from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn

from .core import build_class_mapping, discover_data, save_mapping, seed_everything
from .modeling import PlantDataset, build_model, build_transforms, compatible_image_size
from .train import (
    collect_samples,
    load_compatible_checkpoint,
    make_loader,
    optimizer_parameter_groups,
    remap_class_subset,
    run_epoch,
    set_backbone_trainable,
)


CLASS_NAMES = ["Black-grass", "Common wheat", "Loose Silky-bent", "Scentless Mayweed", "Sugar beet"]


def _serializable_args(args) -> dict:
    output = {}
    for name, value in vars(args).items():
        output[name] = str(value) if isinstance(value, Path) else value
    output.update({"fold": -1, "num_folds": 0, "full_data": True})
    return output


def build_full_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    mapping: dict[str, int],
    args,
    train_samples: int,
    initialization: dict | None = None,
) -> dict:
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "class_to_idx": mapping,
        "model_name": args.model,
        "image_size": args.image_size,
        "pretrained": args.pretrained,
        "transform_mode": args.transform_mode,
        "train_samples": train_samples,
        "full_data": True,
        "initialization": initialization,
        "args": _serializable_args(args),
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    return state


def train_full(args) -> dict:
    started = time.perf_counter()
    seed_everything(args.seed)
    if args.contrastive_weight < 0 or (args.contrastive_weight > 0 and args.model != "convnext_tiny"):
        raise ValueError("--contrastive-weight requires convnext_tiny and a nonnegative weight")
    if args.scheduler_epochs is not None and args.scheduler_epochs < args.epochs:
        raise ValueError('--scheduler-epochs must cover all requested training epochs')
    paths = discover_data(args.root)
    full_mapping = build_class_mapping(paths.train_dir)
    samples = collect_samples(paths.train_dir, full_mapping)
    if args.include_classes:
        included = [full_mapping[name] for name in args.include_classes]
        train_samples, _, mapping = remap_class_subset(
            samples, list(range(len(samples))), included, list(full_mapping)
        )
    else:
        train_samples, mapping = samples, full_mapping
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_mapping(mapping, args.output_dir / "class_mapping.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        args.batch_size = min(args.batch_size, 8)
        args.image_size = min(args.image_size, 160)
        args.epochs = min(args.epochs, 5)
        args.workers = min(args.workers, 1)
    args.image_size = compatible_image_size(args.model, args.image_size)
    dataset = PlantDataset(
        train_samples,
        build_transforms(args.image_size, True, args.strong_aug, args.transform_mode),
    )
    loader = make_loader(dataset, args.batch_size, True, args.workers, args.seed)
    model = build_model(args.model, len(mapping), args.pretrained).to(device)
    initialization = None
    if args.init_checkpoint:
        initialization = load_compatible_checkpoint(
            model, args.init_checkpoint, args.model, -1, args.seed, 0
        )
        print(json.dumps({"initialization": initialization}, ensure_ascii=False))
    if args.freeze_backbone_epochs:
        set_backbone_trainable(model, False)
    if args.head_lr_multiplier == 1.0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, args.lr, args.head_lr_multiplier),
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.scheduler_epochs or args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
        metrics = run_epoch(
            model, loader, criterion, device, optimizer, scaler, args.max_batches, args.grad_clip,
            args.contrastive_weight,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": metrics["loss"],
            "train_micro_f1": metrics["micro_f1"],
            "train_macro_f1": metrics["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row))
        scheduler.step()
        state = build_full_checkpoint(
            model, optimizer, scheduler, scaler, epoch, mapping, args, len(train_samples), initialization
        )
        torch.save(state, args.output_dir / "last.pt")
    torch.save(state, args.output_dir / "final.pt")
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "experiment": args.output_dir.name,
        "model": args.model,
        "pretrained": args.pretrained,
        "image_size": args.image_size,
        "transform_mode": args.transform_mode,
        "contrastive_weight": args.contrastive_weight,
        "seed": args.seed,
        "epochs": args.epochs,
        "scheduler_epochs": args.scheduler_epochs or args.epochs,
        "train_samples": len(train_samples),
        "class_to_idx": mapping,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_micro_f1": history[-1]["train_micro_f1"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - started,
        "validation_metrics": None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train a plant classifier on all available training images")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["tiny_cnn", "resnet18", "efficientnet_b0", "efficientnet_b2", "convnext_tiny", "dinov2_vits14", "dinov2_vitb14"],
        default="convnext_tiny",
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument('--scheduler-epochs', type=int, help='Keep the CV cosine horizon when stopping full-data training earlier')
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=.05)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--transform-mode", choices=["crop", "fine_grained"], default="fine_grained")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--include-classes", nargs="+", choices=CLASS_NAMES)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train_full(parse_args())
