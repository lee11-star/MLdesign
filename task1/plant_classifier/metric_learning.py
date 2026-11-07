from __future__ import annotations

import torch
from torch.nn import functional as F


def convnext_logits_and_features(model, images):
    """Expose the normalized penultimate ConvNeXt vector without changing its logits."""
    if not hasattr(model, "features") or not hasattr(model, "avgpool") or len(model.classifier) != 3:
        raise ValueError("Contrastive training requires a torchvision ConvNeXt classifier")
    features = model.features(images)
    features = model.avgpool(features)
    features = model.classifier[0](features)
    features = model.classifier[1](features)
    return model.classifier[2](features), features


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Supervised contrastive loss for anchors with another same-class item in the batch."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("Expected [batch, features] and [batch] labels")
    same_class = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = same_class & ~diagonal
    counts = positives.sum(dim=1)
    if not torch.any(counts):
        return features.sum() * 0
    normalized = F.normalize(features.float(), dim=1)
    logits = normalized @ normalized.T / temperature
    logits = logits.masked_fill(diagonal, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positives, log_prob, 0).sum(dim=1)
    valid = counts > 0
    return -(positive_log_prob[valid] / counts[valid]).mean()
