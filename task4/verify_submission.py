"""Check all predictions against the unmodified organizer's RLE encoder."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('segmentation'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('organizer_encoder', args.data / 'segmentation_to_csv.py')
    organizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(organizer)
    with (args.data / 'sample_submission.csv').open(newline='', encoding='utf-8') as handle:
        expected_ids = [row['Id'] for row in csv.DictReader(handle)]
    with (args.output / 'submission.csv').open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['Id', 'Predicted'], reader.fieldnames
        rows = list(reader)
    assert [row['Id'] for row in rows] == expected_ids
    paths = [str(args.output / 'masks' / f'{case_id}.png') for case_id in expected_ids]
    native_paths = [str(args.output / 'native_masks' / f'{case_id}.png') for case_id in expected_ids]
    official_rles = organizer.turn_to_str(native_paths)
    details = []
    for row, path, official in zip(rows, paths, official_rles):
        assert row['Predicted'] == official, f'Official RLE differs for {row["Id"]}'
        mask = np.asarray(Image.open(path))
        assert mask.shape == (512, 512)
        assert set(np.unique(mask)) <= {0, 255}
        inverse = np.asarray(Image.open(args.output / 'masks_black_vessels' / f'{row["Id"]}.png'))
        np.testing.assert_array_equal(mask, 255 - inverse)
        pairs = np.array([int(x) for x in row['Predicted'].split()], dtype=np.int64).reshape(-1, 2)
        assert (pairs > 0).all()
        decoded = np.zeros(mask.size, dtype=bool)
        previous_end = 0
        for start, length in pairs:
            begin, end = start - 1, start - 1 + length
            assert begin >= previous_end and end <= mask.size
            decoded[begin:end] = True
            previous_end = end
        np.testing.assert_array_equal(decoded.reshape(mask.shape), mask > 0)
        details.append({'Id': row['Id'], 'runs': len(pairs), 'vessel_pixels': int(decoded.sum())})
    result = {'images': len(rows), 'official_encoder_matches': len(official_rles),
              'round_trip_matches': len(details), 'size_hw': [512, 512], 'details': details}
    (args.output / 'submission_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))


if __name__ == '__main__':
    main()
