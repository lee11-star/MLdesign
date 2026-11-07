import unittest

import torch

from fovea.model import augment_batch, decode_heatmaps, heatmap_loss


class HeatmapContracts(unittest.TestCase):
    def test_peak_decoding_and_horizontal_flip_use_same_coordinates(self):
        ys, xs = torch.meshgrid(torch.linspace(0, 1, 96), torch.linspace(0, 1, 96), indexing='ij')
        logits = -((xs - .37) ** 2 + (ys - .54) ** 2)[None] / (2 * .018 ** 2)
        prediction = decode_heatmaps(logits)
        torch.testing.assert_close(prediction, torch.tensor([[.37, .54]]), atol=.002, rtol=0)
        mirrored = decode_heatmaps(logits.flip(-1))
        torch.testing.assert_close(mirrored[:, 0], 1 - prediction[:, 0])
        torch.testing.assert_close(mirrored[:, 1], prediction[:, 1])
        self.assertLess(float(heatmap_loss(logits, torch.tensor([[.37, .54]]))),
                        float(heatmap_loss(logits, torch.tensor([[.6, .7]]))))

    def test_geometric_augmentation_moves_image_and_label_together(self):
        torch.manual_seed(42)
        size = 128
        x, y = 46, 70
        images = torch.zeros(1, 3, size, size)
        images[:, :, y-2:y+3, x-2:x+3] = 1
        label = torch.tensor([[(x+.5)/size, (y+.5)/size]])
        transformed, moved = augment_batch(images, label)
        weights = transformed.sum(1)[0]
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        centroid = torch.stack([((xx+.5)*weights).sum(), ((yy+.5)*weights).sum()]) / weights.sum() / size
        torch.testing.assert_close(centroid, moved[0], atol=.003, rtol=0)


if __name__ == '__main__':
    unittest.main()
