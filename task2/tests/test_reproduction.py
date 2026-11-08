import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import regularization


class Reproduction(unittest.TestCase):
    def test_prepares_both_regularized_models_from_generated_baseline(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            old, out = root / 'outputs/optimization', root / 'outputs/kaggle80'
            (old / 'convnext_tiny').mkdir(parents=True)
            (root / 'outputs/data').mkdir()
            (root / 'outputs/data/manifest.csv').write_text('manifest', encoding='utf-8')
            for name in ['submission.csv', 'submission_optimized.csv']:
                (root / name).write_text('ID,Emotion\n1,2\n', encoding='utf-8')
            signature = hashlib.sha256(b'manifest').hexdigest()
            (old / 'selection.json').write_text(json.dumps({'accuracy': 0.7, 'data_signature': signature}))
            (old / 'convnext_tiny/config.json').write_text(json.dumps({'model': 'convnext_tiny', 'epochs': 30,
                'scheduler_epochs': 30, 'data_signature': signature, 'final_training': False}))
            trained = []
            def train_stage(path, config, device):
                trained.append((path.name, config, device))
            with patch.multiple(regularization, ROOT=root, OLD=old, OUT=out), \
                 patch.object(regularization, 'train_stage', side_effect=train_stage), \
                 patch.object(regularization, 'run_evaluation') as evaluate:
                regularization.prepare_models()
            self.assertEqual([r[1]['image_size'] for r in trained], [96, 128])
            self.assertTrue(all(r[1]['epochs'] == 40 and r[1]['weight_decay'] == 0.01 and
                                r[1]['label_smoothing'] == 0.1 for r in trained))
            self.assertEqual(evaluate.call_count, 2)
            baseline = json.loads((out / 'baseline.json').read_text())
            self.assertEqual(baseline['data_signature'], signature)
            self.assertEqual(baseline['validation_accuracy'], 0.7)
            self.assertEqual(set(baseline['original_submissions']), {'submission.csv', 'submission_optimized.csv'})


if __name__ == '__main__':
    unittest.main()
