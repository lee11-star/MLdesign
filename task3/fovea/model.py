from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, convnext_tiny, efficientnet_b2


class FoveaNet(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.encoder = resnet18(weights=None)
        if pretrained:
            self.encoder.load_state_dict(torch.load(Path(pretrained), map_location='cpu', weights_only=True))
        self.encoder.fc = nn.Identity()
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in [64, 128, 256, 512]])
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # BatchNorm uses ImageNet running statistics with this small dataset.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images):
        enc = self.encoder
        x = (images - self.mean) / self.std
        x = enc.maxpool(enc.relu(enc.bn1(enc.conv1(x))))
        features = []
        for layer in [enc.layer1, enc.layer2, enc.layer3, enc.layer4]:
            x = layer(x)
            features.append(x)
        x = self.lateral[-1](features[-1])
        for i in [2, 1, 0]:
            x = F.interpolate(x, size=features[i].shape[-2:], mode='bilinear', align_corners=False) + self.lateral[i](features[i])
        return self.head(x).squeeze(1)


class ConvNeXtFoveaNet(nn.Module):
    """Alternative ImageNet encoder with the same spatial FPN head."""
    def __init__(self, pretrained=None):
        super().__init__()
        self.encoder = convnext_tiny(weights=None)
        if pretrained:
            self.encoder.load_state_dict(torch.load(Path(pretrained), map_location='cpu', weights_only=True))
        self.encoder.classifier = nn.Identity()
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in [96, 192, 384, 768]])
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        x = (images-self.mean)/self.std
        features = []
        for index, layer in enumerate(self.encoder.features):
            x = layer(x)
            if index % 2 == 1:
                features.append(x)
        x = self.lateral[-1](features[-1])
        for i in [2, 1, 0]:
            x = F.interpolate(x, size=features[i].shape[-2:], mode='bilinear', align_corners=False) + self.lateral[i](features[i])
        return self.head(x).squeeze(1)


class EfficientNetFoveaNet(FoveaNet):
    """EfficientNet features at strides 4/8/16/32 with the existing FPN contract."""
    def __init__(self, pretrained=None):
        nn.Module.__init__(self)
        self.encoder = efficientnet_b2(weights=None)
        if pretrained:
            self.encoder.load_state_dict(torch.load(Path(pretrained), map_location='cpu', weights_only=True))
        self.encoder.classifier = nn.Identity()
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in [24, 48, 120, 352]])
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        x = (images-self.mean)/self.std
        features = []
        for index, layer in enumerate(self.encoder.features):
            x = layer(x)
            if index in [2, 3, 5, 7]:
                features.append(x)
            if index == 7:
                break
        x = self.lateral[-1](features[-1])
        for i in [2, 1, 0]:
            x = F.interpolate(x, size=features[i].shape[-2:], mode='bilinear', align_corners=False) + self.lateral[i](features[i])
        return self.head(x).squeeze(1)


def create_model(architecture='resnet18', pretrained=None):
    if architecture == 'convnext_contrast':
        from .contrast import ContrastConvNeXtFoveaNet
        return ContrastConvNeXtFoveaNet(pretrained)
    if architecture == 'dinov2_vits14':
        from .dino_model import DinoFoveaNet
        return DinoFoveaNet(pretrained)
    return {'resnet18': FoveaNet, 'convnext_tiny': ConvNeXtFoveaNet,
            'efficientnet_b2': EfficientNetFoveaNet}[architecture](pretrained)


def heatmap_loss(logits, xy, sigma=.025):
    height, width = logits.shape[-2:]
    ys = torch.linspace(0, 1, height, device=logits.device)
    xs = torch.linspace(0, 1, width, device=logits.device)
    distance = (xs[None, None, :] - xy[:, 0, None, None]) ** 2 + (ys[None, :, None] - xy[:, 1, None, None]) ** 2
    target = (-distance / (2 * sigma ** 2)).flatten(1).softmax(1)
    return -(target * logits.float().flatten(1).log_softmax(1)).sum(1).mean()


def decode_heatmaps(logits, radius=3):
    # Local expectation around the peak avoids averaging two distant modes.
    height, width = logits.shape[-2:]
    flat = logits.flatten(1)
    peak = flat.argmax(1)
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device), torch.arange(width, device=logits.device), indexing='ij')
    keep = ((xx[None] - (peak % width)[:, None, None]).abs() <= radius) & ((yy[None] - (peak // width)[:, None, None]).abs() <= radius)
    probability = logits.float().masked_fill(~keep, -1e9).flatten(1).softmax(1).reshape_as(logits)
    x = (probability * xx / (width - 1)).sum((1, 2))
    y = (probability * yy / (height - 1)).sum((1, 2))
    return torch.stack([x, y], 1)


def augment_batch(images, xy, return_transform=False):
    batch = len(images)
    device = images.device
    angle = (torch.rand(batch, device=device) * 2 - 1) * .12
    scale = .92 + .16 * torch.rand(batch, device=device)
    flip = torch.where(torch.rand(batch, device=device) < .5, -1., 1.)
    matrix = torch.zeros(batch, 2, 2, device=device)
    matrix[:, 0, 0] = angle.cos() * scale * flip
    matrix[:, 0, 1] = -angle.sin() * scale
    matrix[:, 1, 0] = angle.sin() * scale * flip
    matrix[:, 1, 1] = angle.cos() * scale
    translation = (torch.rand(batch, 2, device=device) * 2 - 1) * .08
    inverse = torch.linalg.inv(matrix)
    theta = torch.cat([inverse, -(inverse @ translation.unsqueeze(-1))], dim=-1)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, align_corners=False)
    xy = (matrix @ (xy * 2 - 1).unsqueeze(-1)).squeeze(-1) + translation
    xy = (xy + 1) / 2
    brightness = .85 + .3 * torch.rand(batch, 3, 1, 1, device=device)
    images = (images * brightness).clamp(0, 1)
    if return_transform:
        return images, xy, torch.cat([matrix, translation.unsqueeze(-1)], dim=2)
    return images, xy
