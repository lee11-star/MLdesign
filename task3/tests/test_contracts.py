import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fovea.data import normalize_xy, restore_xy, pixel_metrics, write_submission


class Contracts(unittest.TestCase):
    def test_coordinates_roundtrip_for_both_image_sizes(self):
        sizes = np.array([[2992, 2000], [1956, 1934]])
        xy = np.array([[1470.2, 1033.1], [734.0, 1016.8]])
        np.testing.assert_allclose(restore_xy(normalize_xy(xy, sizes), sizes), xy)

    def test_metric_matches_single_value_column_kaggle_format(self):
        # The submission is one value column with alternating X/Y rows.
        actual = np.array([[100., 200.], [300., 400.]])
        predicted = actual + np.array([[3., 4.], [0., 0.]])
        result = pixel_metrics(actual, predicted)
        self.assertAlmostEqual(result['rmse'], 2.5)
        self.assertAlmostEqual(result['mean_distance'], 2.5)

    def test_submission_obeys_template_order_and_preserves_zero(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(Path.cwd().resolve()))
            root = Path(directory)
            template = root / 'sample.csv'
            pd.DataFrame({'ImageID': ['82_Fovea_Y', '81_Fovea_X', '82_Fovea_X', '81_Fovea_Y'],
                          'value': [None] * 4}).to_csv(template, index=False)
            result = write_submission(template, {81: [0., 0.], 82: [321., 456.]}, root / 'out.csv')
            self.assertEqual(result.ImageID.tolist(), pd.read_csv(template).ImageID.tolist())
            self.assertEqual(result.value.tolist(), [456., 0., 321., 0.])

    def test_submission_rejects_missing_or_nonfinite_predictions(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(Path.cwd().resolve()))
            root = Path(directory)
            template = root / 'sample.csv'
            pd.DataFrame({'ImageID': ['81_Fovea_X', '81_Fovea_Y'], 'value': [0, 0]}).to_csv(template, index=False)
            with self.assertRaises(ValueError):
                write_submission(template, {81: [np.nan, 10.]}, root / 'out.csv')
            with self.assertRaises(ValueError):
                write_submission(template, {}, root / 'out.csv')


if __name__ == '__main__':
    unittest.main()
