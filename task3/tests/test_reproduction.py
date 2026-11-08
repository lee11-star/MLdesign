import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import export_optimized


class Reproduction(unittest.TestCase):
    def test_cached_optimization_exports_summary_without_platform_feedback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            out = root / 'outputs/optimization'
            name = 'res640_sigma012'
            def dump(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding='utf-8')
            val = pd.DataFrame({'id': [3, 4], 'Fovea_X': [100., 200.], 'Fovea_Y': [100., 200.],
                                'pred_x': [110., 210.], 'pred_y': [100., 200.], 'distance': [10., 10.]})
            (root / 'outputs/validation').mkdir(parents=True)
            val.to_csv(root / 'outputs/validation/predictions.csv', index=False)
            (root / 'submission.csv').write_text('baseline', encoding='utf-8')
            first = {'config': {'seed': 42}, 'metrics': {'rmse': 0.7071}, 'best_epoch': 1, 'checkpoint': 'first.pt'}
            second = {'config': {'seed': 2026}, 'metrics': {'rmse': 2.1213}, 'best_epoch': 1, 'checkpoint': 'second.pt'}
            dump(out / 'screening.json', {'baseline': {'metrics': {'rmse': 7.0711}},
                'experiments': {name: first}, 'candidates': {name: first['metrics']}})
            dump(out / 'refinement/result.json', {'metrics': {'rmse': 3.}})
            dump(out / f'{name}_seed2026/result.json', second)
            dump(out / f'full_{name}_seed42/result.json', {'config': first['config'], 'epochs': 1, 'checkpoint': 'full.pt'})
            for sub, offset in [(name, 1.), (f'{name}_seed2026', 3.)]:
                (out / sub).mkdir(exist_ok=True)
                frame = val.copy()
                frame['pred_x'] = frame.Fovea_X + offset
                frame.to_csv(out / sub / 'predictions.csv', index=False)
            with patch.multiple(export_optimized, ROOT=root, OUTPUT=out), \
                 patch.object(export_optimized, 'load_split', return_value=(val, val.iloc[:1], val)), \
                 patch.object(export_optimized, 'fit', side_effect=AssertionError('Cached run should not train')), \
                 patch.object(export_optimized, 'run_prediction', return_value={'rows': 4}):
                export_optimized.finalize()
            result = json.loads((out / 'optimization_summary.json').read_text())
            self.assertEqual(result['selection']['stage'], name)
            self.assertAlmostEqual(result['selection']['metrics']['rmse'], 2 ** -0.5)
            self.assertEqual((root / 'submission.csv').read_text(), 'baseline')
            self.assertFalse((out / 'kaggle_feedback.json').exists())


if __name__ == '__main__':
    unittest.main()
