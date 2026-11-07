from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


class PlantDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        return tensor, label, path.name


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


DINO_HUB_REPOSITORY = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def compatible_image_size(model_name: str, image_size: int) -> int:
    """Keep DINO inputs on its 14-pixel patch grid, including CPU fallback."""
    if model_name in {"dinov2_vits14", "dinov2_vitb14"}:
        if image_size < 14:
            raise ValueError("DINOv2 requires an image size of at least 14 pixels")
        return image_size // 14 * 14
    return image_size


class DinoClassifier(nn.Module):
    """DINOv2 encoder with a small trainable classification head."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, num_classes))

    def forward(self, images):
        return self.classifier(self.backbone(images))


class PadToSquare:
    """Center-pad an image without changing its aspect ratio."""

    def __init__(self, fill: tuple[int, int, int] = (0, 0, 0)):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(image, (left, top, side - width - left, side - height - top), fill=self.fill)


def green_foreground_bbox(
    image: Image.Image, margin: float = 0.15, min_pixels: int = 16
) -> tuple[int, int, int, int]:
    """Return a green-foreground bounding box, or the full image when detection is weak.

    The rule uses excess green rather than a fixed hue so it remains useful across
    moderate brightness changes. Coordinates follow PIL's right/bottom-exclusive
    convention.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excess_green = 2 * green - red - blue
    mask = (excess_green > 24) & (green > 35) & (green > red) & (green > blue)
    ys, xs = np.nonzero(mask)
    width, height = image.size
    if len(xs) < min_pixels:
        return 0, 0, width, height
    # Ignore the outermost one percent of green pixels so isolated moss/noise
    # near an image edge does not turn the foreground crop back into a full view.
    left = int(np.floor(np.quantile(xs, 0.01)))
    right = int(np.ceil(np.quantile(xs, 0.99))) + 1
    top = int(np.floor(np.quantile(ys, 0.01)))
    bottom = int(np.ceil(np.quantile(ys, 0.99))) + 1
    pad_x = int(round((right - left) * margin))
    pad_y = int(round((bottom - top) * margin))
    return max(0, left - pad_x), max(0, top - pad_y), min(width, right + pad_x), min(height, bottom + pad_y)


class DualViewTransform:
    """Return channel-concatenated full-image and green-foreground views."""

    def __init__(self, image_size: int, training: bool, strong: bool = False):
        self.foreground = GreenForegroundCrop()
        self.full_transform = _fine_grained_transform(image_size, training, strong)
        self.local_transform = _fine_grained_transform(image_size, training, strong)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        local = self.foreground(image)
        full_tensor = self.full_transform(image)
        local_tensor = self.local_transform(local)
        return torch.cat((full_tensor, local_tensor), dim=0)


class GreenForegroundCrop:
    """Pickle-safe callable used by Windows DataLoader workers."""

    def __init__(self, margin: float = 0.15, min_pixels: int = 16):
        self.margin = margin
        self.min_pixels = min_pixels

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.crop(green_foreground_bbox(image, self.margin, self.min_pixels))


class DualViewClassifier(nn.Module):
    """Shared-backbone classifier for a full image and a foreground crop."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(feature_dim * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Dual-view models require N×6×H×W input, got {tuple(x.shape)}")
        full_features = self.backbone(x[:, :3])
        local_features = self.backbone(x[:, 3:])
        return self.classifier(torch.cat((full_features, local_features), dim=1))


def _fine_grained_transform(image_size: int, training: bool, strong: bool = False):
    ops = [PadToSquare(), transforms.Resize((image_size, image_size))]
    if training:
        ops.extend([
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(18 if strong else 12),
            transforms.ColorJitter(
                brightness=.12 if strong else .08,
                contrast=.12 if strong else .08,
                saturation=.09 if strong else .06,
                hue=.015 if strong else .01,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(.035, .035) if strong else (.025, .025),
                scale=(.95, 1.05) if strong else (.97, 1.03),
            ),
        ])
    ops.extend([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    if training and strong:
        ops.append(transforms.RandomErasing(p=.12, scale=(.01, .06)))
    return transforms.Compose(ops)


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name in {"dinov2_vits14", "dinov2_vitb14"}:
        hub_dir = Path(__file__).resolve().parents[1] / "outputs" / "torch_cache" / "hub"
        torch.hub.set_dir(str(hub_dir))
        backbone = torch.hub.load(
            DINO_HUB_REPOSITORY, name, pretrained=pretrained,
            trust_repo=True, skip_validation=True,
        )
        return DinoClassifier(backbone, backbone.embed_dim, num_classes)
    if name == "tiny_cnn":
        return TinyCNN(num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if name == "dual_resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return DualViewClassifier(backbone, feature_dim, num_classes)
    if name == "dual_convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Identity()
        return DualViewClassifier(backbone, feature_dim, num_classes)
    raise ValueError(f"Unknown model: {name}")


def build_transforms(image_size: int, training: bool, strong: bool = False, mode: str = "crop"):
    if mode == "dual_foreground":
        return DualViewTransform(image_size, training, strong)
    if mode == "fine_grained":
        return _fine_grained_transform(image_size, training, strong)
    if mode != "crop":
        raise ValueError(f"Unknown transform mode: {mode}")
    if training:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.72 if strong else 0.82, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20 if strong else 12),
        ]
        if strong:
            ops.extend([
                transforms.ColorJitter(brightness=.18, contrast=.18, saturation=.15, hue=.03),
                transforms.RandomAffine(degrees=0, translate=(.05, .05), scale=(.95, 1.05)),
            ])
        ops.extend([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        if strong:
            ops.append(transforms.RandomErasing(p=.2, scale=(.02, .1)))
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
