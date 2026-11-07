import tempfile
import unittest
from pathlib import Path

import torch

from fovea.model import create_model
from fovea.optimized import load_model


class BackboneContracts(unittest.TestCase):
    def test_dino_spatial_gradient_and_reload(self):
        torch.set_num_threads(2)
        model = create_model('dinov2_vits14')
        image = torch.rand(1, 3, 56, 56)
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 16, 16))
        logits.square().mean().backward()
        self.assertGreater(float(model.encoder.patch_embed.proj.weight.grad.abs().sum()), 0.)
        model.eval()
        with torch.no_grad():
            expected = model(image)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/'model.pt'
            torch.save({'state_dict': model.state_dict(),
                        'config': {'architecture': 'dinov2_vits14'}, 'training_ids': [1]}, path)
            reloaded, _ = load_model(path, 'cpu')
            with torch.no_grad():
                torch.testing.assert_close(reloaded(image), expected)

    def test_efficientnet_heatmap_gradient_bn_and_reload(self):
        torch.set_num_threads(2)
        model = create_model('efficientnet_b2')
        model.train()
        self.assertTrue(all(not layer.training for layer in model.encoder.modules()
                            if isinstance(layer, torch.nn.BatchNorm2d)))
        image = torch.rand(1, 3, 64, 64)
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 16, 16))
        logits.square().mean().backward()
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()), 0.)
        model.eval()
        with torch.no_grad():
            expected = model(image)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory)/'model.pt'
            torch.save({'state_dict': model.state_dict(),
                        'config': {'architecture': 'efficientnet_b2'}, 'training_ids': [1]}, path)
            reloaded, _ = load_model(path, 'cpu')
            with torch.no_grad():
                torch.testing.assert_close(reloaded(image), expected)

    def test_default_factory_preserves_original_resnet_state_keys(self):
        from fovea.model import FoveaNet
        torch.set_num_threads(2)
        self.assertEqual(set(create_model().state_dict()), set(FoveaNet().state_dict()))

    def test_convnext_heatmap_gradient_and_checkpoint_reload(self):
        torch.set_num_threads(2)
        model = create_model('convnext_tiny')
        image = torch.rand(1, 3, 64, 64)
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 16, 16))
        logits.square().mean().backward()
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()), 0.)
        model.eval()
        with torch.no_grad():
            expected = model(image)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory).resolve() / 'model.pt'
            self.assertTrue(path.is_relative_to(Path.cwd().resolve()))
            torch.save({'state_dict': model.state_dict(), 'config': {'architecture': 'convnext_tiny'},
                        'training_ids': [1, 2]}, path)
            reloaded, metadata = load_model(path, 'cpu')
            self.assertEqual(metadata['config']['architecture'], 'convnext_tiny')
            with torch.no_grad():
                torch.testing.assert_close(reloaded(image), expected)


if __name__ == '__main__':
    unittest.main()
