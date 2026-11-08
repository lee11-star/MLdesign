import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vessel.data import load_case, split_ids, training_weights
from vessel.model import UNet, predict_probability
from vessel.submission import encode_rle, make_submission_mask, write_submission
from vessel.augment import affine_batch


class PipelineTests(unittest.TestCase):
    def test_image_split_and_white_vessel_labels(self):
        ids = [str(i) for i in range(1, 21)]
        train_ids, val_ids = split_ids(ids)
        self.assertEqual((len(train_ids), len(val_ids)), (16, 4))
        self.assertFalse(set(train_ids) & set(val_ids))
        self.assertEqual(set(train_ids + val_ids), set(ids))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            (root / 'train/image').mkdir(parents=True)
            (root / 'train/label').mkdir(parents=True)
            Image.fromarray(np.full((24, 32, 3), 100, np.uint8)).save(root / 'train/image/1.png')
            mask = np.zeros((24, 32), np.uint8)
            mask[5:18, 11:13] = 255
            Image.fromarray(mask).save(root / 'train/label/1.png')
            features, field, label = load_case(root, 'train', '1')
            self.assertEqual(features.shape, (4, 24, 32))
            np.testing.assert_array_equal(label, mask > 128)
            self.assertTrue(field.all())
        label = np.zeros((33, 41), np.float32)
        label[:, 6] = 1
        label[8:25, 20:29] = 1
        weights = training_weights(label, thin_weight=2)
        np.testing.assert_array_equal(weights[label == 0], 1)
        self.assertTrue((weights[:, 6] == 3).all())
        self.assertEqual(weights[16, 24], 1)

    def test_unet_preserves_non_square_shape_and_gradients(self):
        torch.set_num_threads(2)
        model = UNet(width=4)
        logits = model(torch.randn(2, 4, 33, 41))
        self.assertEqual(tuple(logits.shape), (2, 1, 33, 41))
        logits.mean().backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())

    def test_d4_restores_non_square_pixel_positions(self):
        class PointwiseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return 2 * x[:, :1] - 0.3

        features = np.linspace(-1, 1, 4 * 33 * 41, dtype=np.float32).reshape(4, 33, 41)
        model = PointwiseModel()
        probability = predict_probability(model, features, torch.device('cpu'), tta='d4')
        self.assertEqual(model.calls, 8)
        expected = 1 / (1 + np.exp(-(2 * features[0] - 0.3)))
        np.testing.assert_allclose(probability, expected, atol=1e-6)

    def test_affine_keeps_images_labels_and_weights_aligned(self):
        torch.manual_seed(7)
        target = torch.zeros(3, 1, 48, 48)
        target[:, :, :, 24:] = 1
        features = target.repeat(1, 4, 1, 1)
        weights = 1 + 2 * target
        images, labels, transformed_weights = affine_batch(features, target, weights, output_size=24)
        self.assertEqual(tuple(images.shape), (3, 4, 24, 24))
        self.assertEqual(set(labels.unique().tolist()), {0, 1})
        torch.testing.assert_close((images[:, :1] > 0.5).float(), labels)
        torch.testing.assert_close(transformed_weights, 1 + 2 * labels)

    def test_selected_dice_matches_exported_threshold_mask(self):
        from train import dice_score, score_probabilities
        probability = np.array([[0.5, 0.49], [0.49, 0.9]], dtype=np.float32)
        label = np.array([[1, 0], [0, 1]], dtype=bool)
        result, _, targets = score_probabilities([probability], [label], ['1'])
        exported = make_submission_mask(probability > result['threshold'])
        self.assertAlmostEqual(result['mean_dice'], dice_score(exported, targets[0]), places=12)

    def test_row_major_one_based_rle_and_sample_order(self):
        expanded = make_submission_mask(np.array([[1, 0], [0, 0]], bool), size=4)
        np.testing.assert_array_equal(expanded, [[1, 1, 1, 0], [1, 1, 1, 0],
                                                [1, 1, 1, 0], [0, 0, 0, 0]])
        mask = np.array([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=bool)
        self.assertEqual(encode_rle(mask), '1 2 5 1 7 2')
        self.assertEqual(encode_rle(np.zeros((2, 4), bool)), '')
        self.assertEqual(encode_rle(np.ones((2, 4), bool)), '1 8')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            for name, image in [('2', mask), ('1', ~mask)]:
                Image.fromarray(image.astype(np.uint8) * 255).save(root / f'{name}.png')
            (root / 'sample.csv').write_text('Id,Predicted\n2,\n1,\n', encoding='utf-8')
            write_submission(root, root / 'sample.csv', root / 'result.csv')
            lines = (root / 'result.csv').read_text().splitlines()
            self.assertEqual(lines, ['Id,Predicted', '2,1 2 5 1 7 2', '1,3 2 6 1'])


if __name__ == '__main__':
    unittest.main()
