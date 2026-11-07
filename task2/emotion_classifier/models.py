import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, convnext_tiny, efficientnet_b2


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks, channels = [], 1
        for width in [32, 64, 128, 256]:
            blocks += [nn.Conv2d(channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
                       nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(),
                       nn.MaxPool2d(2), nn.Dropout2d(0.1)]
            channels = width
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256 * 3 * 3, 256), nn.ReLU(),
                                        nn.Dropout(0.4), nn.Linear(256, 6))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(architecture, pretrained=None, head_dropout=0.0, trainable_blocks=4, dinov2_repo=None):
    if architecture in ('dan', 'dan_confidence'):
        from .facial import DAN
        return DAN(pretrained, use_confidence=architecture == 'dan_confidence')
    if architecture == 'dinov2_vits14':
        from .dinov2 import DinoClassifier
        return DinoClassifier(pretrained, head_dropout, trainable_blocks, dinov2_repo)
    if architecture == 'cnn':
        if head_dropout:
            raise ValueError('The small CNN already has a fixed dropout head')
        return SmallCNN()
    builders = {'resnet18': resnet18, 'convnext_tiny': convnext_tiny, 'efficientnet_b2': efficientnet_b2}
    if architecture not in builders:
        raise ValueError(f'Unknown architecture: {architecture}')
    model = builders[architecture](weights=None)
    if pretrained:
        # Read the existing local ImageNet state only; no download or cache writes.
        model.load_state_dict(torch.load(Path(pretrained), map_location='cpu', weights_only=True), strict=True)
    if architecture == 'resnet18':
        model.fc = nn.Linear(model.fc.in_features, 6)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
    if head_dropout:
        if architecture == 'resnet18':
            model.fc = nn.Sequential(nn.Dropout(head_dropout), model.fc)
        else:
            layers = list(model.classifier.children())
            model.classifier = nn.Sequential(*layers[:-1], nn.Dropout(head_dropout), layers[-1])
    return model


def preprocessing_config(architecture, image_size=None, geometry='standard'):
    size = image_size or (48 if architecture == 'cnn' else 224 if architecture in ('dinov2_vits14', 'dan', 'dan_confidence') else 96)
    if architecture == 'cnn' and size != 48:
        raise ValueError('SmallCNN requires 48x48 input')
    if size < 32:
        raise ValueError('Input size must be at least 32')
    if architecture == 'dinov2_vits14' and size % 14:
        raise ValueError('DINOv2 input size must be a multiple of 14')
    if geometry not in ('standard', 'flip_only'):
        raise ValueError(f'Unknown geometry recipe: {geometry}')
    return dict(size=size, channels=1 if architecture == 'cnn' else 3,
                mean=[0.5] if architecture == 'cnn' else [0.485, 0.456, 0.406],
                std=[0.5] if architecture == 'cnn' else [0.229, 0.224, 0.225],
                training_augmentation=dict(horizontal_flip=0.5, rotation_degrees=10 if geometry == 'standard' else 0,
                                           translation_fraction=0.08 if geometry == 'standard' else 0,
                                           scale=[0.9, 1.1] if geometry == 'standard' else [1.0, 1.0]))


def preprocess(images, architecture, training=False, flip=False, image_size=None, geometry='standard'):
    x = images.float() / 255.0
    if training and geometry == 'flip_only':
        reflected = torch.rand(len(x), device=x.device) < 0.5
        x = torch.where(reflected[:, None, None, None], x.flip(-1), x)
    elif training:
        n = len(x)
        angles = (torch.rand(n, device=x.device) * 2 - 1) * (10 * math.pi / 180)
        scales = torch.rand(n, device=x.device) * 0.2 + 0.9
        signs = torch.where(torch.rand(n, device=x.device) < 0.5, -1.0, 1.0)
        cosine, sine = angles.cos() / scales, angles.sin() / scales
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 0, 1] = cosine * signs, -sine
        theta[:, 1, 0], theta[:, 1, 1] = sine * signs, cosine
        theta[:, :, 2] = (torch.rand(n, 2, device=x.device) * 2 - 1) * 0.16
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, padding_mode='border', align_corners=False)
    elif flip:
        x = x.flip(-1)
    if architecture == 'cnn':
        return (x - 0.5) / 0.5
    size = image_size or (224 if architecture in ('dinov2_vits14', 'dan', 'dan_confidence') else 96)
    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False, antialias=True).expand(-1, 3, -1, -1)
    mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std
