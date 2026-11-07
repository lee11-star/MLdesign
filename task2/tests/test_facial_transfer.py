"""Use only existing real task images to check facial transfer and confidence training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from emotion_classifier.data import DEFAULT_DATA, load_arrays, read_manifest
from emotion_classifier.inference import load_checkpoint
from emotion_classifier.models import build_model, preprocess
from emotion_classifier.training import train_run

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'outputs/round8/pretrained/dan_affectnet7.pth'


def real_images(split):
    rows = read_manifest(ROOT / 'outputs/data/manifest.csv')
    chosen = [next(r for r in rows if r['split'] == split and r['label'] == label) for label in range(6)]
    return load_arrays(chosen, DEFAULT_DATA, split, ROOT / 'outputs/data')[:2]


class FacialTransferContracts(unittest.TestCase):
    def test_completed_runner_recovers_interrupted_checkpoint_publication(self):
        import pipeline as runner
        torch.set_num_threads(2)
        x, y = real_images('train')
        vx, vy = real_images('val')
        for final in (False, True):
            config = dict(model='cnn', epochs=1, scheduler_epochs=1, batch_size=6,
                          lr=0.001, weight_decay=0.0, seed=42, device='cpu', pretrained=None,
                          final_training=final, data_signature='unit_check_real_manifest_subset')
            with tempfile.TemporaryDirectory(dir=ROOT / 'outputs/test_tmp') as tmp:
                folder = Path(tmp) / 'run'
                train_run(config, x, y, None if final else vx, None if final else vy, folder)
                selected = folder / ('final.pt' if final else 'best.pt')
                selected.unlink()  # Simulate interrupted publication of this disposable unit artifact.
                with patch.object(runner, 'OUT', Path(tmp)), patch.object(runner, 'configuration', return_value=config):
                    runner.fit('run', 'cnn', final=final)
                self.assertTrue(selected.exists())
                self.assertEqual(torch.load(selected, weights_only=False)['epoch'], 1)

    def test_official_backbone_is_loaded_and_new_head_has_six_classes(self):
        torch.set_num_threads(2)
        model = build_model('dan', pretrained=WEIGHTS).eval()
        source = torch.load(WEIGHTS, map_location='cpu', weights_only=True)['model_state_dict']
        self.assertTrue(torch.equal(model.features[0].weight, source['features.0.weight']))
        self.assertTrue(torch.equal(model.cat_head3.ca.attention[0].weight,
                                    source['cat_head3.ca.attention.0.weight']))
        x, _ = real_images('train')
        with torch.inference_mode():
            logits = model(preprocess(torch.from_numpy(x), 'dan', image_size=96))
        self.assertEqual(tuple(logits.shape), (6, 6))
        self.assertTrue(torch.isfinite(logits).all())

    def test_ranking_penalty_rewards_confidence_separation(self):
        from emotion_classifier.facial import confidence_rank_loss
        equal = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
        penalty = confidence_rank_loss(equal, fraction=0.7, margin=0.15)
        self.assertAlmostEqual(penalty.item(), 0.15, places=6)
        penalty.backward()
        self.assertTrue(torch.isfinite(equal.grad).all() and equal.grad.abs().sum() > 0)
        separated = torch.tensor([0.9, 0.8, 0.2, 0.1])
        self.assertEqual(confidence_rank_loss(separated, 0.7, 0.15).item(), 0.0)

    def test_confidence_updates_resumes_and_reloads_using_real_split_images(self):
        torch.set_num_threads(2)
        x, y = real_images('train')
        vx, vy = real_images('val')
        original_y, original_vy = y.copy(), vy.copy()
        config = dict(model='dan_confidence', image_size=96, epochs=2, scheduler_epochs=2,
                      batch_size=6, lr=0.001, backbone_lr_scale=0.1, weight_decay=1e-4,
                      seed=42, device='cpu', pretrained=None, final_training=False,
                      data_signature='unit_check_real_manifest_subset',
                      confidence_fraction=0.7, confidence_margin=0.15, confidence_regularization=1.0)
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs/test_tmp') as tmp:
            full, resumed = Path(tmp) / 'full', Path(tmp) / 'resumed'
            train_run(config, x, y, vx, vy, full)
            train_run(config, x, y, vx, vy, resumed, stop_after=1)
            before = torch.load(resumed / 'last.pt', weights_only=False)['model']['confidence.weight'].clone()
            train_run(config, x, y, vx, vy, resumed, resume=resumed / 'last.pt')
            a = torch.load(full / 'last.pt', weights_only=False)
            b = torch.load(resumed / 'last.pt', weights_only=False)
            self.assertEqual(a['history'], b['history'])
            self.assertTrue(all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']))
            self.assertFalse(torch.equal(before, b['model']['confidence.weight']))
            model, saved, _ = load_checkpoint(resumed / 'last.pt', 'cpu')
            with torch.inference_mode():
                self.assertEqual(tuple(model(preprocess(torch.from_numpy(vx), 'dan_confidence', image_size=96)).shape), (6, 6))
            np.testing.assert_array_equal(y, original_y)
            np.testing.assert_array_equal(vy, original_vy)


if __name__ == '__main__':
    unittest.main()
