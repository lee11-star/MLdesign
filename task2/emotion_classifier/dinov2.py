"""Use the existing local DINOv2 source and weights without hub downloads."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


TASK1_CACHE = Path(__file__).resolve().parents[2] / 'task1/outputs/torch_cache/hub'
DEFAULT_REPO = TASK1_CACHE / 'facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DEFAULT_WEIGHTS = TASK1_CACHE / 'checkpoints/dinov2_vits14_pretrain.pth'


class DinoClassifier(nn.Module):
    def __init__(self, pretrained=None, head_dropout=0.0, trainable_blocks=4, repo=None):
        super().__init__()
        # task1 is read-only, including Python bytecode caches.
        source = str(Path(repo or DEFAULT_REPO).resolve())
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, source)
        try:
            from dinov2.hub.backbones import dinov2_vits14
            self.backbone = dinov2_vits14(pretrained=False)
        finally:
            sys.path.remove(source)
            sys.dont_write_bytecode = previous
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True), strict=True)
        if not 0 <= trainable_blocks <= len(self.backbone.blocks):
            raise ValueError('trainable_blocks must be between 0 and 12')
        self.backbone.requires_grad_(False)
        if trainable_blocks:
            for block in self.backbone.blocks[-trainable_blocks:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
        self.feature_dim = self.backbone.embed_dim * 5
        self.classifier = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(self.feature_dim, 6))

    def features(self, inputs):
        layers = self.backbone.get_intermediate_layers(inputs, n=4, return_class_token=True)
        return torch.cat([item[1] for item in layers] + [layers[-1][0].mean(1)], dim=1)

    def forward(self, inputs):
        features = self.features(inputs)
        # Preserve the fitted linear probe's float32 logits under backbone AMP.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return self.classifier(features.float())


def fit_linear_probe(features, labels, regularization):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    # Each image contributes its original and flipped view; neither validation
    # statistics nor validation labels are used to fit the scaler or classifier.
    x = np.asarray(features).reshape(-1, features.shape[-1])
    y = np.repeat(labels, features.shape[1])
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, max_iter=400, tol=1e-4))
    with threadpool_limits(limits=4):
        probe.fit(x, y)
    return probe


def probe_head_state(probe):
    scaler, linear = probe.steps[0][1], probe.steps[1][1]
    weights = linear.coef_ / scaler.scale_[None]
    bias = linear.intercept_ - weights @ scaler.mean_
    return {'1.weight': torch.from_numpy(weights.astype(np.float32)),
            '1.bias': torch.from_numpy(bias.astype(np.float32))}
