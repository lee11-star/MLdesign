import unittest

import torch

from fovea.metric_objective import expectation_xy, infer_expectation, pixel_coordinate_mse


class MetricObjectiveContracts(unittest.TestCase):
    def test_augmented_error_is_restored_before_original_pixel_weighting(self):
        sizes = torch.tensor([[2992., 2000.], [1956., 1934.]])
        original_error = torch.tensor([[.01, .02], [-.02, .01]])
        angle = torch.tensor(.12)
        matrix = 1.08 * torch.tensor([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
        matrix = matrix.expand(2, -1, -1).clone()
        matrix[1, :, 0] *= -1
        transform = torch.cat([matrix, torch.tensor([[[.04], [-.03]], [[-.02], [.01]]])], dim=2)
        target = torch.tensor([[.5, .5], [.55, .5]])
        predicted = (target + torch.einsum('bij,bj->bi', matrix, original_error)).requires_grad_()
        expected = ((original_error*sizes)**2).mean()
        actual = pixel_coordinate_mse(predicted, target, sizes, transform=transform)
        torch.testing.assert_close(actual, expected, atol=.003, rtol=0)
        actual.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertGreater(float(predicted.grad.abs().sum()), 0.)

    def test_mse_uses_original_pixel_sizes(self):
        prediction = torch.tensor([[.51, .51], [.51, .51]])
        target = torch.full((2, 2), .5)
        sizes = torch.tensor([[3000., 2000.], [2000., 2000.]])
        self.assertAlmostEqual(float(pixel_coordinate_mse(prediction, target, sizes)), 525., places=2)

    def test_expectation_is_differentiable_and_flip_consistent(self):
        logits = torch.zeros(1, 32, 32, requires_grad=True)
        images = torch.ones(1, 3, 128, 128) * .5
        region = [.2, .3, .8, .7]
        target = torch.tensor([[.6, .53]])
        sizes = torch.tensor([[3000., 2000.]])
        xy = expectation_xy(logits, images, region)
        loss = pixel_coordinate_mse(xy, target, sizes) / 2500
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        updated = logits.detach() - .001 * logits.grad
        updated_xy = expectation_xy(updated, images, region)
        self.assertLess(float(pixel_coordinate_mse(updated_xy, target, sizes) / 2500), float(loss))
        mirrored = expectation_xy(updated.flip(-1), images.flip(-1), region)
        torch.testing.assert_close(mirrored[:, 0], 1-updated_xy[:, 0])
        torch.testing.assert_close(mirrored[:, 1], updated_xy[:, 1])

    def test_training_support_follows_geometric_transform(self):
        logits = torch.zeros(1, 101, 101)
        images = torch.ones(1, 3, 202, 202) * .5
        transform = torch.tensor([[[1., 0., .2], [0., 1., -.1]]])
        xy = expectation_xy(logits, images, [.4, .45, .6, .55], transform)
        torch.testing.assert_close(xy, torch.tensor([[.6, .45]]), atol=.006, rtol=0)

    def test_temperature_reduces_secondary_peak_bias_and_keeps_gradients(self):
        logits = torch.full((1, 101, 101), -20.)
        logits[0, 50, 40] = 10.
        logits[0, 50, 60] = 9.
        logits.requires_grad_()
        images = torch.ones(1, 3, 202, 202) * .5
        region = [.2, .3, .8, .7]
        broad = expectation_xy(logits, images, region)
        concentrated = expectation_xy(logits, images, region, temperature=.25)
        self.assertLess(abs(float(concentrated[0, 0])-.4), abs(float(broad[0, 0])-.4))
        loss = pixel_coordinate_mse(concentrated, torch.tensor([[.4, .5]]), torch.tensor([[3000., 2000.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_shift_views_are_restored_to_original_coordinate_frame(self):
        class ImageHeatmap(torch.nn.Module):
            def forward(self, images):
                return images[:, 0]*100
        size = 128
        yy, xx = torch.meshgrid((torch.arange(size)+.5)/size, (torch.arange(size)+.5)/size, indexing='ij')
        intensity = (.9-3*((xx-.47)**2+(yy-.53)**2)).clamp(0, 1)
        images = (intensity[None, None].repeat(1, 3, 1, 1)*255).round().to(torch.uint8)
        model = ImageHeatmap()
        original = infer_expectation(model, images, [.2, .3, .8, .7], temperature=.25)
        shifted = infer_expectation(model, images, [.2, .3, .8, .7], temperature=.25,
                                    shifts=[[0, 0], [2, 2], [2, -2], [-2, 2], [-2, -2]])
        torch.testing.assert_close(torch.from_numpy(shifted), torch.from_numpy(original), atol=.0005, rtol=0)


if __name__ == '__main__':
    unittest.main()
