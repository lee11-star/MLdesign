import torch
from torch import nn
from torch.utils.data import TensorDataset

from plant_classifier import modeling
from plant_classifier.feature_probe import extract_embeddings
from plant_classifier.train import set_backbone_trainable


def test_dino_image_size_respects_patch_grid_after_cpu_limit():
    assert modeling.compatible_image_size('dinov2_vits14', 160) == 154
    assert modeling.compatible_image_size('dinov2_vitb14', 384) == 378
    assert modeling.compatible_image_size('dinov2_vits14', 392) == 392
    assert modeling.compatible_image_size('convnext_tiny', 160) == 160


def test_dino_classifier_keeps_backbone_frozen_and_trains_five_class_head():
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
    model = modeling.DinoClassifier(backbone, 8, 5)
    set_backbone_trainable(model, False)

    logits = model(torch.randn(2, 3, 28, 28))
    logits.sum().backward()

    assert logits.shape == (2, 5)
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert model.classifier[-1].weight.grad is not None


def test_dino_embedding_probe_excludes_classification_head():
    backbone = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8))
    model = modeling.DinoClassifier(backbone, 8, 5)
    dataset = TensorDataset(torch.randn(3, 3, 28, 28), torch.zeros(3, dtype=torch.long))

    embeddings = extract_embeddings(model, dataset, 2, torch.device('cpu'))

    assert embeddings.shape == (3, 8)
