"""Protect label semantics, duplicate isolation, ID alignment and checkpoint reuse."""
import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from emotion_classifier.data import CLASS_TO_LABEL, audit_dataset, split_records, load_arrays
from emotion_classifier.inference import write_submission, validate_submission, load_checkpoint
from emotion_classifier.models import build_model, preprocess
from emotion_classifier.training import train_run
from emotion_classifier import training

TEST_TEMP = Path(__file__).resolve().parents[1] / 'outputs' / 'test_tmp'
TEST_TEMP.mkdir(parents=True, exist_ok=True)


class DataContracts(unittest.TestCase):
    def test_folder_labels_follow_competition_and_audit_reads_images(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            expected = {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5}
            for name, label in expected.items():
                target = root / 'train' / name
                target.mkdir(parents=True)
                for index in range(10):
                    Image.new('L', (48, 48), label * 20 + index).save(target / f'{name}{index}.jpg')
            (root / 'test').mkdir()
            Image.new('L', (48, 48), 240).save(root / 'test' / 'b.jpg')
            Image.new('L', (48, 48), 241).save(root / 'test' / 'a.jpg')
            template = root / 'submission.csv'
            template.write_text('ID,Emotion\nb.jpg,0\na.jpg,0\n', encoding='utf-8')
            records, summary = audit_dataset(root, template)
            self.assertEqual(CLASS_TO_LABEL, expected)
            self.assertEqual(summary['train_count'], 60)
            self.assertEqual(summary['test_count'], 2)
            for row in records:
                if row['kind'] == 'train':
                    self.assertEqual(row['label'], expected[Path(row['path']).parent.name])
            pixels, labels, ids = load_arrays(records, root, 'test')
            self.assertEqual(pixels.shape, (2, 1, 48, 48))
            self.assertEqual(ids, ['a.jpg', 'b.jpg'])

    def test_split_is_reproducible_excludes_conflicts_and_keeps_duplicate_groups(self):
        rows = []
        for label in range(6):
            for group in range(20):
                for copy in range(1 + (group % 3 == 0)):
                    rows.append(dict(path=f'{label}/{group}_{copy}.jpg', label=label,
                                     sha256=f'{label}-{group}', kind='train', id=f'{label}-{group}-{copy}'))
        rows.extend([dict(path='conflict1.jpg', label=0, sha256='conflict', kind='train', id='c1'),
                     dict(path='conflict2.jpg', label=5, sha256='conflict', kind='train', id='c2')])
        result = split_records(rows, seed=42)
        self.assertEqual(result, split_records(rows, seed=42))
        train = [r for r in result if r['split'] == 'train']
        val = [r for r in result if r['split'] == 'val']
        self.assertFalse({r['sha256'] for r in train} & {r['sha256'] for r in val})
        self.assertEqual({r['label'] for r in val}, set(range(6)))
        self.assertEqual([r['split'] for r in result if r['sha256'] == 'conflict'], ['excluded', 'excluded'])
        self.assertTrue(0.15 < len(val) / (len(train) + len(val)) < 0.25)


class SubmissionContracts(unittest.TestCase):
    def test_shuffled_predictions_are_aligned_to_template(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            template, output = Path(folder) / 'template.csv', Path(folder) / 'out.csv'
            template.write_text('ID,Emotion\nb.jpg,0\na.jpg,0\nc.jpg,0\n')
            write_submission(template, {'c.jpg': 5, 'a.jpg': 4, 'b.jpg': 1}, output)
            self.assertEqual(output.read_text().splitlines(), ['ID,Emotion', 'b.jpg,1', 'a.jpg,4', 'c.jpg,5'])
            self.assertEqual(validate_submission(output, template)['rows'], 3)

    def test_bad_labels_missing_ids_and_reordered_rows_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            template, output = Path(folder) / 'template.csv', Path(folder) / 'out.csv'
            template.write_text('ID,Emotion\nb.jpg,0\na.jpg,0\n')
            with self.assertRaises(ValueError):
                write_submission(template, {'a.jpg': 1}, output)
            for bad in ['ID,Emotion\nb.jpg,1.0\na.jpg,2\n',
                        'ID,Emotion\nb.jpg,6\na.jpg,2\n',
                        'ID,Emotion\na.jpg,2\nb.jpg,1\n']:
                output.write_text(bad)
                with self.assertRaises(ValueError):
                    validate_submission(output, template)


class ModelContracts(unittest.TestCase):
    def test_interruption_after_last_checkpoint_recovers_selected_artifact_and_metadata(self):
        # Catch a committed last.pt whose selected best/final artifact was never published.
        torch.set_num_threads(2)
        x = np.random.default_rng(8).integers(0, 256, (12, 1, 48, 48), dtype=np.uint8)
        y = np.tile(np.arange(6, dtype=np.int64), 2)
        config = dict(model='cnn', epochs=1, batch_size=6, lr=0.001, weight_decay=0.0001,
                      seed=42, device='cpu', num_workers=0, pretrained=None, data_signature='fixture')
        original_save = training.atomic_save
        def interrupted_save(value, path):
            original_save(value, path)
            if Path(path).name == 'last.pt':
                raise InterruptedError('Injected interruption after committing last.pt')
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            for validation in [False, True]:
                output = Path(folder) / str(validation)
                vx, vy = (x, y) if validation else (None, None)
                with patch.object(training, 'atomic_save', side_effect=interrupted_save):
                    with self.assertRaises(InterruptedError):
                        train_run(config, x, y, vx, vy, output)
                # Also recreate the previous writer's lost-selected-artifact state.
                selected = output / ('best.pt' if validation else 'final.pt')
                if selected.exists():
                    selected.unlink()
                train_run(config, x, y, vx, vy, output, resume=output / 'last.pt')
                self.assertTrue(selected.is_file(), 'Resume must repair the selected artifact even at the final epoch')
                state = torch.load(selected, weights_only=False)
                self.assertEqual(state['epoch'], 1)
                self.assertTrue((output / 'summary.json').is_file(), 'Resume must repair completion metadata')

    def test_architectures_preprocess_backward_and_reload(self):
        torch.set_num_threads(2)
        for architecture in ['cnn', 'resnet18']:
            model = build_model(architecture)
            images = torch.randint(0, 256, (4, 1, 48, 48), dtype=torch.uint8)
            inputs = preprocess(images, architecture, training=False)
            logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (4, 6))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 4, 5])).backward()
            model.eval()
            with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
                checkpoint = Path(folder) / 'model.pt'
                torch.save({'model': model.state_dict(), 'config': {'model': architecture},
                            'class_to_label': CLASS_TO_LABEL}, checkpoint)
                restored, config, _ = load_checkpoint(checkpoint, 'cpu')
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(inputs), restored(inputs)))

    def test_resume_reproduces_uninterrupted_training(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(11)
        x = rng.integers(0, 256, (12, 1, 48, 48), dtype=np.uint8)
        y = np.tile(np.arange(6, dtype=np.int64), 2)
        config = dict(model='cnn', epochs=2, batch_size=6, lr=0.001, weight_decay=0.0001,
                      seed=42, device='cpu', num_workers=0, pretrained=None, data_signature='fixture')
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            full = Path(folder) / 'full'
            resumed = Path(folder) / 'resumed'
            train_run(config, x, y, x, y, full)
            train_run(config, x, y, x, y, resumed, stop_after=1)
            train_run(config, x, y, x, y, resumed, resume=resumed / 'last.pt')
            a = torch.load(full / 'last.pt', weights_only=False)
            b = torch.load(resumed / 'last.pt', weights_only=False)
            self.assertEqual(a['history'], b['history'])
            self.assertTrue(all(torch.equal(a['model'][key], b['model'][key]) for key in a['model']))


if __name__ == '__main__':
    unittest.main()
