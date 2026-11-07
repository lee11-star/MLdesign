"""DAN transfer and label-preserving SCN-style confidence learning.

Attention modules adapted from https://github.com/yaoing/DAN (MIT).
Copyright (c) 2021 Zhengyao Wen. Full notice: licenses/DAN-MIT.txt.
The original seven-class output head is replaced, never remapped by guessing labels.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256))
        self.conv_3x3 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512))
        self.conv_1x3 = nn.Sequential(nn.Conv2d(256, 512, (1, 3), padding=(0, 1)), nn.BatchNorm2d(512))
        self.conv_3x1 = nn.Sequential(nn.Conv2d(256, 512, (3, 1), padding=(1, 0)), nn.BatchNorm2d(512))
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        return x * y.sum(dim=1, keepdim=True)


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(nn.Linear(512, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                                       nn.Linear(32, 512), nn.Sigmoid())

    def forward(self, x):
        pooled = self.gap(x).flatten(1)
        return pooled * self.attention(pooled)


class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa, self.ca = SpatialAttention(), ChannelAttention()
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.ca(self.sa(x))


class DAN(nn.Module):
    def __init__(self, pretrained=None, use_confidence=False):
        super().__init__()
        self.features = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
        self.num_head = 4
        for i in range(self.num_head):
            setattr(self, f'cat_head{i}', CrossAttentionHead())
        self.classifier = nn.Sequential(nn.Linear(512, 6), nn.BatchNorm1d(6))
        self.use_confidence = use_confidence
        if use_confidence:
            # Keep the common classifier initialization and augmentation RNG
            # identical between the two controlled recipes.
            with torch.random.fork_rng(devices=[]):
                self.confidence = nn.Linear(512, 1)
        if pretrained:
            source = torch.load(pretrained, map_location='cpu', weights_only=True)['model_state_dict']
            state = self.state_dict()
            transferred = {k: v for k, v in source.items() if k.startswith(('features.', 'cat_head'))}
            expected = {k for k in state if k.startswith(('features.', 'cat_head'))}
            if set(transferred) != expected:
                raise ValueError('Official DAN feature/attention keys differ from this model')
            state.update(transferred)
            self.load_state_dict(state, strict=True)

    def _representations(self, x):
        features = self.features(x)
        heads = torch.stack([getattr(self, f'cat_head{i}')(features) for i in range(self.num_head)], dim=1)
        fused = F.log_softmax(heads, dim=1).sum(dim=1)
        return self.classifier(fused), features.mean(dim=(-2, -1))

    def forward_with_confidence(self, x):
        logits, pooled = self._representations(x)
        attention = self.confidence(pooled).sigmoid()
        return logits * attention, attention.flatten()

    def forward(self, x):
        if self.use_confidence:
            return self.forward_with_confidence(x)[0]
        return self._representations(x)[0]


def confidence_rank_loss(attention, fraction=0.7, margin=0.15):
    values = attention.float().flatten().sort(descending=True).values
    boundary = max(1, min(len(values) - 1, int(len(values) * fraction)))
    return F.relu(values.new_tensor(margin) - (values[:boundary].mean() - values[boundary:].mean()))
