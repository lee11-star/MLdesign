"""Spatial localization using the existing local DINOv2 source and weights."""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class DinoFoveaNet(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        source = (Path(__file__).resolve().parents[2]/'task1/outputs/torch_cache/hub'/
                  'facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8')
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        try:
            from dinov2.hub.backbones import dinov2_vits14
            self.encoder = dinov2_vits14(pretrained=False)
        finally:
            sys.path.remove(str(source))
            sys.dont_write_bytecode = previous
        if pretrained:
            self.encoder.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True))
        self.lateral = nn.ModuleList([nn.Conv2d(384, 64, 1) for _ in range(4)])
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        features = self.encoder.get_intermediate_layers((images-self.mean)/self.std,
                                                        n=[2, 5, 8, 11], reshape=True)
        x = sum(layer(feature) for layer, feature in zip(self.lateral, features))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return self.head(x).squeeze(1)
